-- | Reusable training pipeline for the microGPT-spirit demo.
-- |
-- | Lifted from `examples/microgpt/jax/Main.purs` so both the CLI
-- | demo and the in-browser worker can drive it. The `Params` record
-- | is the entire configuration surface; `Callbacks` is the
-- | output channel.
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
-- | Train + sample are split (`trainOnly` returns a `Trained` handle;
-- | `sampleFrom` consumes one) so the browser UI can re-sample with
-- | new temperature/length settings without retraining. `runMicrogpt`
-- | is the back-compat one-shot that does both.
module Jax.Demo.Microgpt
  ( Params
  , TrainCallbacks
  , Callbacks
  , Trained
  , defaultParams
  , defaultCorpus
  , trainOnly
  , sampleFrom
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
import Jax.Loaders.CharTokenizer (CharTokenizer)
import Jax.Loaders.CharTokenizer as CharTokenizer
import Jax.NN.Block (LayerWeights, ModelConfig, ModelWeights, refModelWeights)
import Jax.NN.Generate (generateTemperature)
import Jax.NN.RoPE (RoPETables, precomputeRoPE)
import Jax.NN.Train (makeCrossEntropyLoss)
import Jax.Shape (S2)
import Jax.Shape.Tensor (Tensor, unsafeAssumeShape)
import Jax.Shape.Tensor.Op (onesWith) as Op
import Jax.Optax as Optax
import Jax.Optax.Schedule as Schedule
import Jax.Random (Key)
import Jax.Random as Random
import Jax.Train as Train

-- | All user-facing knobs. The model-shape fields (hidden, nHeads, …)
-- | are mostly fixed in practice; what callers usually vary are
-- | `corpus`, `numSteps`, `lr`, `temperature`, `numSamples`.
-- |
-- | `trainOnly` reads the train-relevant fields (corpus, numSteps,
-- | lr, seed, plus model shape). `sampleFrom` reads the sample-
-- | relevant fields (temperature, numSamples, maxSampleLen, seed).
-- | The shared shape is convenience; nothing prevents you from
-- | passing two different `Params` records into the two functions
-- | (e.g. to re-sample with a different temperature).
type Params =
  { corpus :: String
  , trainWindow :: Int
  , maxSeqLen :: Int
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

-- | Train-phase callbacks. `onStart` fires once before the loop with
-- | the parameter count (closed-form from the config) and the vocab
-- | size (extracted from the tokenizer). `onProgress` fires once per
-- | step with the new loss; the bootstrap pass before the loop fires
-- | as step 0.
type TrainCallbacks =
  { onStart    :: { paramCount :: Int, vocabSize :: Int } -> Effect Unit
  , onProgress :: Int -> Number -> Effect Unit  -- step, loss
  }

-- | Combined train + sample callbacks for the one-shot `runMicrogpt`.
type Callbacks =
  { onStart    :: { paramCount :: Int, vocabSize :: Int } -> Effect Unit
  , onProgress :: Int -> Number -> Effect Unit
  , onSampled  :: Int -> String -> Effect Unit
  , onDone     :: Number -> Effect Unit          -- final loss
  }

-- | Output of training: everything `sampleFrom` needs to produce
-- | continuations. Holds NDArrays whose lifetime is the caller's
-- | responsibility. Re-use as many times as you like before
-- | discarding (the underlying buffers are not consumed by sampling).
type Trained =
  { weights :: ModelWeights
  , cfg :: ModelConfig
  , rope :: RoPETables
  , tokenizer :: CharTokenizer
  , finalLoss :: Number
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
  , maxSeqLen: 64
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

-- | Phase 1: train. Returns a `Trained` handle the caller can pass to
-- | `sampleFrom` (zero or more times). Throws via Effect's exception
-- | channel on shape mismatches (e.g. empty corpus).
trainOnly :: Params -> TrainCallbacks -> Effect Trained
trainOnly p cb = do
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
      , maxSeqLen: max p.trainWindow p.maxSeqLen
      , vocabSize
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
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
  pure
    { weights: finalState.weights
    , cfg
    , rope
    , tokenizer: tok
    , finalLoss: finalState.lastLoss
    }

-- | Phase 2: sample. Reads from a `Trained` handle (built by
-- | `trainOnly`) — the original tensors are *not* consumed, so the
-- | caller can call this repeatedly with different parameters.
-- |
-- | Only the sample-relevant fields of `Params` are consulted:
-- | `temperature`, `numSamples`, `maxSampleLen`, `seed`. The other
-- | fields (corpus, model shape, lr, …) are ignored — passing the
-- | same `Params` you trained with is fine, but you can also build
-- | a fresh record if you only want to vary the sampling settings.
sampleFrom
  :: Trained
  -> Params
  -> (Int -> String -> Effect Unit)   -- ^ per-sample callback (idx, text)
  -> Effect Unit
sampleFrom t p emit = do
  let
    startId = 0   -- corpus first-char index
    promptLen = 1
    -- Hard cap: every position read must exist in the RoPE table /
    -- KV cache, both sized at cfg.maxSeqLen. Generation produces
    -- `maxSampleLen` new tokens after a 1-token prompt, hitting
    -- positions 0..maxSampleLen, so the cap is `maxSeqLen - promptLen`.
    cap = max 1 (t.cfg.maxSeqLen - promptLen)
    safeLen = min p.maxSampleLen cap
  for_ (Array.range 0 (p.numSamples - 1)) \i -> do
    keyS <- Random.mkKey (p.seed + 1 + i)
    out <- generateTemperature keyS p.temperature t.cfg t.weights t.rope
      [ startId ] safeLen
    emit i (CharTokenizer.decode t.tokenizer out)

-- | Back-compat: train then sample in one call. Same shape as before
-- | the train/sample split landed; mainly for the CLI demo.
runMicrogpt :: Params -> Callbacks -> Effect Unit
runMicrogpt p cb = do
  trained <- trainOnly p
    { onStart: cb.onStart, onProgress: cb.onProgress }
  sampleFrom trained p cb.onSampled
  cb.onDone trained.finalLoss

-- | Total parameter count for the given config. Closed-form so the
-- | UI can show "training N parameters" before allocations happen.
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

-- | Glorot-scaled normal for a 2D linear weight. Returns a typed
-- | `Tensor (S2 a b)` directly — the unsafe shape claim is confined
-- | inside this helper rather than spread across each caller.
glorotMat
  :: forall a b
   . Effect Key
  -> Int
  -> Int
  -> Effect (Tensor (S2 a b))
glorotMat nextKey rows cols = do
  k <- nextKey
  raw <- Random.normal k [ rows, cols ]
  rawR <- ref raw
  scaled <- mulScalar rawR (N.sqrt (2.0 / toNumber (rows + cols)))
  dispose raw
  -- The single shape claim for the helper. Callers consume the
  -- typed result and the type system threads it onward.
  pure (unsafeAssumeShape (scaled :: NDArray D2))

buildWeights :: Key -> ModelConfig -> Effect ModelWeights
buildWeights k0 cfg = do
  nextKey <- mkKeySource k0
  embedding <- glorotMat nextKey cfg.vocabSize cfg.hidden
  finalNorm <- Op.onesWith [ cfg.hidden ]
  layers <- traverse (\_ -> buildLayer nextKey cfg) (Array.range 1 cfg.nLayers)
  pure { embedding, layers, finalNorm }

buildLayer :: Effect Key -> ModelConfig -> Effect LayerWeights
buildLayer nextKey cfg = do
  attnNorm <- Op.onesWith [ cfg.hidden ]
  wq <- glorotMat nextKey cfg.hidden (cfg.nHeads * cfg.headDim)
  wk <- glorotMat nextKey cfg.hidden (cfg.nKvHeads * cfg.headDim)
  wv <- glorotMat nextKey cfg.hidden (cfg.nKvHeads * cfg.headDim)
  wo <- glorotMat nextKey (cfg.nHeads * cfg.headDim) cfg.hidden
  mlpNorm <- Op.onesWith [ cfg.hidden ]
  gateProj <- glorotMat nextKey cfg.hidden cfg.intermediate
  upProj <- glorotMat nextKey cfg.hidden cfg.intermediate
  downProj <- glorotMat nextKey cfg.intermediate cfg.hidden
  pure
    { attnNorm
    , attn: { wq, wk, wv, wo }
    , mlpNorm
    , mlp: { gateProj, upProj, downProj }
    }
