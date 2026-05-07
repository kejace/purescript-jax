-- | Reimagined microGPT-spirit port — single file, end-to-end,
-- | top-to-bottom, real transformer powered by purescript-jax.
-- |
-- | The Faithful sibling at ../faithful/ does the same training task
-- | (next-character prediction on a names corpus) but with from-scratch
-- | scalar autograd. Here, JAX takes over: tensor ops vectorize the
-- | per-position math, and `valueAndGradT` walks the computation graph
-- | for us instead of `Value.backward`.
-- |
-- | Section structure mirrors the Faithful demo so you can read both
-- | side by side and see exactly what JAX hides.
-- |
-- | Run from this directory: `bunx spago run`.
module Microgpt.Jax.Main where

import Prelude

import Data.Array as Array
import Data.Foldable (for_)
import Data.Int (toNumber)
import Data.Number as N
import Data.Traversable (traverse)
import Effect (Effect)
import Effect.Console (log)
import Effect.Ref as Ref
import Jax.Autodiff (valueAndGradT)
import Jax.Core (D1, D2, NDArray, arrayInt1D, dispose, init, mulScalar, ones, ref
                , setDefaultDevice)
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

-- =============================================================================
-- Section 0: corpus (inline; Faithful demo uses the exact same string)
-- =============================================================================

namesCorpus :: String
namesCorpus = """ada
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

-- =============================================================================
-- Section 1: tokenizer
-- =============================================================================
--
-- Faithful demo hand-rolls buildVocab + charToId + idToChar (~15 lines).
-- We use the framework module — three lines.

-- =============================================================================
-- Section 2: parameter init
-- =============================================================================

-- A `KeySource` returns a fresh derived Key on each call. Sub-keys are
-- generated via `splitKey2`: the source's stored key is split into
-- (a, b) pairs; we hand out `a`, keep `b` for next time. Pure JAX
-- discipline (no Effect.Random); reproducible from a single seed.
mkKeySource :: Key -> Effect (Effect Key)
mkKeySource k0 = do
  cell <- Ref.new k0
  pure do
    cur <- Ref.read cell
    { a, b } <- Random.splitKey2 cur
    Ref.write b cell
    pure a

-- Glorot-scaled normal: σ = sqrt(2 / (fan_in + fan_out)). The 2D
-- variant is the one we need for every linear weight matrix.
glorotMat
  :: Effect Key      -- key source: each call returns a fresh sub-key
  -> Int -> Int      -- rows × cols
  -> Effect (NDArray D2)
glorotMat nextKey rows cols = do
  k <- nextKey
  raw <- Random.normal k [ rows, cols ]
  rawR <- ref raw
  scaled <- mulScalar rawR (N.sqrt (2.0 / toNumber (rows + cols)))
  dispose raw
  pure scaled

-- Build a fresh `ModelWeights` for the given config. Linear weights are
-- Glorot-init via `Random.normal` + scalar scaling; RMSNorm γ tensors
-- initialized to ones. The key tree is split once per leaf via
-- `mkKeySource` for reproducibility from a single seed.
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

-- =============================================================================
-- Section 3: model + loss
-- =============================================================================
--
-- Both the forward pass and cross-entropy loss live in the framework:
--
--   transformerStack / forwardLogits  (Jax.NN.Block)
--   makeCrossEntropyLoss              (Jax.NN.Train)
--
-- The Faithful sibling builds these by hand on top of `Value`. See its
-- Section 3 (`forward`, `crossEntropy`) and Section 4 (`backward` then
-- the manual Adam loop). What you'd write yourself, JAX writes for you
-- via `valueAndGradT`:
--
--   vagFn <- valueAndGradT lossFn
--   -- vagFn :: EffectFn1 Weights { value :: scalar loss; grad :: ∂loss/∂Weights }
--
-- This is the autograd boundary. The Faithful demo's `Value.backward`
-- is what's hidden inside.

-- =============================================================================
-- Section 4: training loop
-- =============================================================================

-- For the demo: train on the FIRST `trainWindow` characters of the
-- corpus, treated as one long sequence (one (prompt, target) pair,
-- where target is prompt shifted by 1). Real training would window
-- over the whole corpus with random offsets — see README.

trainWindow :: Int
trainWindow = 32

-- =============================================================================
-- Section 5: sampling
-- =============================================================================
--
-- `generateTemperature` from Jax.NN.Generate runs a KV-cached decode
-- loop with temperature softmax + categorical sampling. The Faithful
-- sibling re-implements this loop scalar-by-scalar in Section 5 of
-- its Main.purs.

-- =============================================================================
-- main: end-to-end glue.
-- =============================================================================

main :: Effect Unit
main = do
  init
  _ <- setDefaultDevice "wasm"
  log "[microgpt-jax] start"

  -- Section 1.
  let
    tok = CharTokenizer.fromText namesCorpus
    vocabSize = CharTokenizer.size tok
  log $ "[microgpt-jax] vocab=" <> show vocabSize

  let
    allIds = CharTokenizer.encode tok namesCorpus
    promptIds = Array.take trainWindow allIds
    targetIds = Array.take trainWindow (Array.drop 1 allIds)

  -- Section 2.
  let
    cfg :: ModelConfig
    cfg =
      { hidden: 32, nHeads: 4, nKvHeads: 4, headDim: 8
      , intermediate: 64, nLayers: 1, maxSeqLen: trainWindow
      , vocabSize, ropeTheta: 10000.0, normEps: 1.0e-6
      }
  log $ "[microgpt-jax] hidden=" <> show cfg.hidden
    <> " nHeads=" <> show cfg.nHeads
    <> " nLayers=" <> show cfg.nLayers
  rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
  keyInit <- Random.mkKey 1337
  weights0 <- buildWeights keyInit cfg

  -- Section 3-4.
  promptT <- arrayInt1D promptIds
  targetT <- arrayInt1D targetIds
  let lossFn = makeCrossEntropyLoss cfg rope promptT targetT
  vagFn <- valueAndGradT lossFn
  let numSteps = 100

  -- Optimizer = chain(adam(lr), scaleBySchedule(linearDecay)). The
  -- chain composes left-to-right: adam computes the update from the
  -- gradient, scaleBySchedule then rescales by the per-step factor.
  baseAdam <- Optax.adam 5.0e-3
  decay <- Schedule.scaleBySchedule (Schedule.linearDecay numSteps)
  opt <- Schedule.chain [ baseAdam, decay ]
  -- Optax.initT consumes its argument. Ref-bump every leaf of weights0
  -- so the original tree stays alive for `Train.initial` below.
  weightsForInit <- refModelWeights weights0
  state0 <- Optax.initT opt weightsForInit
  -- `Train.initial` runs the first valueAndGrad pass and bootstraps a
  -- TrainState carrying { weights, optState, grad, lastLoss }.
  bootstrap <- Train.initial vagFn weights0 state0
  log $ "  step 0 · loss " <> show bootstrap.lastLoss

  -- Run training: framework collapses the per-step Adam dance to one
  -- call. See Section 4 of ../faithful/src/Microgpt/Faithful/Main.purs
  -- for the unwrapped version.
  finalState <- Train.stepN opt vagFn numSteps
    (\i s -> when (mod i 10 == 0 || i == 1) do
        log $ "  step " <> show i <> " · loss " <> show s.lastLoss)
    bootstrap
  log $ "[microgpt-jax] training done · final loss " <> show finalState.lastLoss

  -- Section 5: sample 5 continuations from prompt "a".
  let startId = 0  -- 'a' is at index 0 since corpus starts with "ada"
  log "[microgpt-jax] sampling 5 continuations:"
  for_ (Array.range 1 5) \i -> do
    keyS <- Random.mkKey (1000 + i)
    out <- generateTemperature keyS 0.8 cfg finalState.weights rope [ startId ] 16
    log $ "  → " <> CharTokenizer.decode tok out

  log "[microgpt-jax] done"
