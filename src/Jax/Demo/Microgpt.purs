-- | Reusable training pipeline for the microGPT-spirit demo.
-- |
-- | Lifted from `examples/microgpt/jax/Main.purs` so both the CLI
-- | demo and the in-browser worker can drive it. The `Params` record
-- | is the entire configuration surface; `Callbacks` is the
-- | output channel (per-step loss, per-sample text, completion).
-- |
-- | The pipeline:
-- |
-- |   1. Tokenize the corpus (CharTokenizer).
-- |   2. Init parameters Glorot-normal via `Random.normal` + key
-- |      splitting from a single seed (reproducible).
-- |   3. Build a 1-layer transformer cross-entropy loss over the
-- |      first `trainWindow` chars of the corpus (one fixed window).
-- |   4. Train via `Train.stepN` with adam ⨉ linearDecay schedule.
-- |   5. Sample `numSamples` continuations from a single-char prompt
-- |      via `generateTemperature`.
-- |
-- | Callbacks fire as the pipeline progresses; the function is
-- | otherwise pure-effect (no console writes, no global state).
module Jax.Demo.Microgpt
  ( Params
  , Callbacks
  , defaultParams
  , defaultCorpus
  , runMicrogpt
  ) where

import Prelude

import Data.Array as Array
import Data.Foldable (for_)
import Data.Int (toNumber)
import Data.Number as N
import Data.Traversable (traverse)
import Effect (Effect)
import Effect.Ref as Ref
import Jax.Autodiff (valueAndGradT)
import Jax.Core (D1, D2, NDArray, arrayInt1D, dispose, mulScalar, ones, ref)
import Jax.Loaders.CharTokenizer as CharTokenizer
import Jax.NN.Block (LayerWeights, ModelConfig, ModelWeights, refModelWeights)
import Jax.NN.Generate (generateTemperature)
import Jax.NN.RoPE (precomputeRoPE)
import Jax.NN.Train (makeCrossEntropyLoss)
import Jax.Optax as Optax
import Jax.Optax.Schedule as Schedule
import Jax.Random (Key)
import Jax.Random as Random
import Jax.Train as Train

-- | All user-facing knobs. The model-shape fields (hidden, nHeads, …)
-- | are mostly fixed in practice; what callers usually vary are
-- | `corpus`, `numSteps`, `lr`, `temperature`, `numSamples`.
type Params =
  { corpus :: String
  , trainWindow :: Int
  , hidden :: Int
  , nHeads :: Int
  , nKvHeads :: Int
  , headDim :: Int
  , intermediate :: Int
  , nLayers :: Int
  , numSteps :: Int
  , lr :: Number
  , seed :: Int
  , temperature :: Number
  , numSamples :: Int
  , maxSampleLen :: Int
  }

-- | Output channel for the pipeline. All three are optional — pass
-- | `pure unit` for any callback you don't care about.
type Callbacks =
  { onStart    :: { paramCount :: Int, vocabSize :: Int } -> Effect Unit
  , onProgress :: Int -> Number -> Effect Unit  -- step, loss
  , onSampled  :: Int -> String -> Effect Unit   -- 0-based index, decoded text
  , onDone     :: Number -> Effect Unit          -- final loss
  }

-- | A small subset of Karpathy's names.txt — enough characters to
-- | exercise the pipeline without external IO. Callers can swap in a
-- | larger corpus via `Params.corpus`.
defaultCorpus :: String
defaultCorpus = """ada
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
evelyn
abigail
emily
elizabeth
mila
ella
avery
sofia
camila
aria
scarlett
victoria
madison
luna
grace
chloe
penelope
layla
riley
zoey
"""

-- | A reasonable starting point. Bump `numSteps` for longer training.
defaultParams :: Params
defaultParams =
  { corpus: defaultCorpus
  , trainWindow: 32
  , hidden: 32
  , nHeads: 4
  , nKvHeads: 4
  , headDim: 8
  , intermediate: 64
  , nLayers: 1
  , numSteps: 100
  , lr: 5.0e-3
  , seed: 1337
  , temperature: 0.8
  , numSamples: 5
  , maxSampleLen: 16
  }

-- | Run the full pipeline: tokenize → init → train → sample. Calls
-- | the relevant callback at each phase boundary; returns when
-- | sampling is done. Throws via Effect's exception channel on
-- | shape mismatches (only possible if `corpus` is empty).
runMicrogpt :: Params -> Callbacks -> Effect Unit
runMicrogpt p cb = do
  -- Section 1: tokenizer.
  let tok = CharTokenizer.fromText p.corpus
      vocabSize = CharTokenizer.size tok
  -- Section 2: weights + RoPE tables.
  let
    cfg :: ModelConfig
    cfg =
      { hidden: p.hidden
      , nHeads: p.nHeads
      , nKvHeads: p.nKvHeads
      , headDim: p.headDim
      , intermediate: p.intermediate
      , nLayers: p.nLayers
      , maxSeqLen: p.trainWindow
      , vocabSize
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
  -- Train window: prompt = first N chars, target = shift-by-1.
  let
    allIds = CharTokenizer.encode tok p.corpus
    promptIds = Array.take p.trainWindow allIds
    targetIds = Array.take p.trainWindow (Array.drop 1 allIds)
  rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
  keyInit <- Random.mkKey p.seed
  weights0 <- buildWeights keyInit cfg
  cb.onStart { paramCount: countParams cfg, vocabSize }
  -- Section 3-4: training. Build loss + valueAndGrad once with the
  -- fixed prompt window; iterate via Train.stepN.
  promptT <- arrayInt1D promptIds
  targetT <- arrayInt1D targetIds
  let lossFn = makeCrossEntropyLoss cfg rope promptT targetT
  vagFn <- valueAndGradT lossFn
  baseAdam <- Optax.adam p.lr
  decay <- Schedule.scaleBySchedule (Schedule.linearDecay p.numSteps)
  opt <- Schedule.chain [ baseAdam, decay ]
  weightsForInit <- refModelWeights weights0
  state0 <- Optax.initT opt weightsForInit
  bootstrap <- Train.initial vagFn weights0 state0
  cb.onProgress 0 bootstrap.lastLoss
  finalState <- Train.stepN opt vagFn p.numSteps
    (\i s -> cb.onProgress i s.lastLoss)
    bootstrap
  -- Section 5: sample N continuations from a single-char prompt.
  -- Default start char is the first vocab entry (whatever the corpus
  -- starts with — for the names corpus that's 'a').
  let startId = 0
  for_ (Array.range 0 (p.numSamples - 1)) \i -> do
    keyS <- Random.mkKey (p.seed + 1 + i)
    out <- generateTemperature keyS p.temperature cfg finalState.weights rope
      [ startId ] p.maxSampleLen
    cb.onSampled i (CharTokenizer.decode tok out)
  cb.onDone finalState.lastLoss

-- | Total parameter count for the given config. Closed-form so the
-- | UI can show "training N parameters" before allocations happen;
-- | mirrors what `Pytree.countParams` would tell us after the build.
countParams :: ModelConfig -> Int
countParams cfg =
  cfg.vocabSize * cfg.hidden               -- embedding
    + cfg.hidden                            -- finalNorm
    + cfg.nLayers *
        ( cfg.hidden                                     -- attnNorm
            + cfg.hidden * cfg.nHeads * cfg.headDim      -- wq
            + cfg.hidden * cfg.nKvHeads * cfg.headDim    -- wk
            + cfg.hidden * cfg.nKvHeads * cfg.headDim    -- wv
            + cfg.nHeads * cfg.headDim * cfg.hidden      -- wo
            + cfg.hidden                                 -- mlpNorm
            + cfg.hidden * cfg.intermediate              -- gateProj
            + cfg.hidden * cfg.intermediate              -- upProj
            + cfg.intermediate * cfg.hidden              -- downProj
        )

-- =============================================================================
-- Internals (formerly inline in the CLI demo).
-- =============================================================================

-- | A `KeySource` returns a fresh derived Key on each call. Pure JAX
-- | discipline (no Effect.Random); reproducible from a single seed.
mkKeySource :: Key -> Effect (Effect Key)
mkKeySource k0 = do
  cell <- Ref.new k0
  pure do
    cur <- Ref.read cell
    { a, b } <- Random.splitKey2 cur
    Ref.write b cell
    pure a

-- | Glorot-scaled normal for a 2D linear weight.
glorotMat
  :: Effect Key
  -> Int -> Int
  -> Effect (NDArray D2)
glorotMat nextKey rows cols = do
  k <- nextKey
  raw <- Random.normal k [ rows, cols ]
  rawR <- ref raw
  scaled <- mulScalar rawR (N.sqrt (2.0 / toNumber (rows + cols)))
  dispose raw
  pure scaled

buildWeights :: Key -> ModelConfig -> Effect ModelWeights
buildWeights k0 cfg = do
  nextKey <- mkKeySource k0
  embedding <- glorotMat nextKey cfg.vocabSize cfg.hidden
  finalNorm <- ones [ cfg.hidden ] :: Effect (NDArray D1)
  layers <- traverse (\_ -> buildLayer nextKey cfg) (Array.range 1 cfg.nLayers)
  pure { embedding, layers, finalNorm }

buildLayer :: Effect Key -> ModelConfig -> Effect LayerWeights
buildLayer nextKey cfg = do
  attnNorm <- ones [ cfg.hidden ] :: Effect (NDArray D1)
  wq <- glorotMat nextKey cfg.hidden (cfg.nHeads * cfg.headDim)
  wk <- glorotMat nextKey cfg.hidden (cfg.nKvHeads * cfg.headDim)
  wv <- glorotMat nextKey cfg.hidden (cfg.nKvHeads * cfg.headDim)
  wo <- glorotMat nextKey (cfg.nHeads * cfg.headDim) cfg.hidden
  mlpNorm <- ones [ cfg.hidden ] :: Effect (NDArray D1)
  gateProj <- glorotMat nextKey cfg.hidden cfg.intermediate
  upProj <- glorotMat nextKey cfg.hidden cfg.intermediate
  downProj <- glorotMat nextKey cfg.intermediate cfg.hidden
  pure
    { attnNorm
    , attn: { wq, wk, wv, wo }
    , mlpNorm
    , mlp: { gateProj, upProj, downProj }
    }
