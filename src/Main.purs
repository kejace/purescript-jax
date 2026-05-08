module Main where

import Prelude

import Control.Monad.Trans.Class (lift)
import Effect (Effect)
import Effect.Console (log)
import Jax.Core
  ( D1
  , D2
  , NDArray
  , ones
  , setDefaultDevice
  )
import Jax.Managed (Managed, allocate, allocateT, runManaged)
import Jax.NN.Block (LayerWeights, ModelConfig, ModelWeights)
import Jax.NN.Generate (generateGreedyCached)
import Jax.NN.RoPE (RoPETables, precomputeRoPE)
import Jax.Shape.Tensor.Op (onesWith) as Op

-- | End-to-end demo: build a tiny synthetic model with all-ones weights,
-- | run greedy autoregressive generation, log the result.
-- |
-- | This isn't a useful model — uniform weights produce uniform-ish
-- | logits — but it exercises the entire pipeline end-to-end on the
-- | wasm backend: embedding → n × (RMSNorm + GQA-with-RoPE + SwiGLU)
-- | → final-norm → LM-head → argmax-sample, with refcount discipline
-- | enforced by `Managed`.
-- For a real-model demo: replace `buildModel` with a load step that
-- 1. fetches a safetensors checkpoint over HTTP,
-- 2. calls `Jax.Loaders.Safetensors.parseSafetensors` on the bytes,
-- 3. calls `Jax.Loaders.LlamaAdapter.loadLlamaWeights cfg parsed`,
-- 4. precomputes the RoPE tables for the model's `headDim`/`maxSeqLen`.
-- The resulting `ModelWeights` plugs into the existing `generateGreedyCached`.
main :: Effect Unit
main = do
  log "[purescript-jax] initializing wasm backend"
  setDefaultDevice "wasm"
  log "[purescript-jax] building synthetic model (all-ones weights)"
  let
    cfg :: ModelConfig
    cfg =
      { hidden: 8
      , nHeads: 2
      , nKvHeads: 2
      , headDim: 4
      , intermediate: 16
      , nLayers: 2
      , maxSeqLen: 16
      , vocabSize: 5
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
    prompt = [ 0, 1, 2 ]
    maxNew = 5
  log $ "[purescript-jax] prompt: " <> show prompt
  log $ "[purescript-jax] max new tokens: " <> show maxNew
  runManaged (buildModel cfg) \{ weights, rope } -> do
    log "[purescript-jax] running KV-cached greedy decode"
    out <- generateGreedyCached cfg weights rope prompt maxNew
    log $ "[purescript-jax] generated tokens: " <> show out

-- | Allocate a synthetic model + RoPE tables inside a Managed scope,
-- | so all weights are released when the scope exits.
buildModel
  :: ModelConfig
  -> Managed { weights :: ModelWeights, rope :: RoPETables }
buildModel cfg = do
  emb <- allocateT (Op.onesWith [ cfg.vocabSize, cfg.hidden ])
  layer0 <- buildLayer cfg
  layer1 <- buildLayer cfg
  fn <- allocateT (Op.onesWith [ cfg.hidden ])
  rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
  cosT <- allocate (pure rope.cos)
  sinT <- allocate (pure rope.sin)
  let
    weights :: ModelWeights
    weights =
      { embedding: emb
      , layers: [ layer0, layer1 ]
      , finalNorm: fn
      }
    ropeTables = { cos: cosT, sin: sinT }
  pure { weights, rope: ropeTables }

buildLayer
  :: ModelConfig
  -> Managed LayerWeights
buildLayer cfg = do
  attnNorm <- allocateT (Op.onesWith [ cfg.hidden ])
  wq <- allocateT (Op.onesWith [ cfg.hidden, cfg.nHeads * cfg.headDim ])
  wk <- allocateT (Op.onesWith [ cfg.hidden, cfg.nKvHeads * cfg.headDim ])
  wv <- allocateT (Op.onesWith [ cfg.hidden, cfg.nKvHeads * cfg.headDim ])
  wo <- allocateT (Op.onesWith [ cfg.nHeads * cfg.headDim, cfg.hidden ])
  mlpNorm <- allocateT (Op.onesWith [ cfg.hidden ])
  gp <- allocateT (Op.onesWith [ cfg.hidden, cfg.intermediate ])
  up <- allocateT (Op.onesWith [ cfg.hidden, cfg.intermediate ])
  dp <- allocateT (Op.onesWith [ cfg.intermediate, cfg.hidden ])
  pure
    { attnNorm
    , attn: { wq, wk, wv, wo }
    , mlpNorm
    , mlp: { gateProj: gp, upProj: up, downProj: dp }
    }
