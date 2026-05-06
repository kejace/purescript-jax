module Jax.Loaders.LlamaAdapter
  ( LoadedCheckpoint
  , loadLlamaWeights
  ) where

import Prelude

import Data.Array (elem, range)
import Data.Traversable (traverse)
import Effect (Effect)
import Jax.Core (D1, D2, NDArray, ref, transpose)
import Jax.Loaders.Safetensors (SafetensorsMap, getTensor, tensorNames)
import Jax.NN.Block (LayerWeights, ModelConfig, ModelWeights)

-- | A loaded checkpoint pairs the autodiff-friendly `ModelWeights`
-- | pytree with the LM-head projection. For tied-embedding checkpoints
-- | `lmHead` is a `ref`-bumped handle to `weights.embedding`; for
-- | untied checkpoints it is the explicit `lm_head.weight` tensor.
type LoadedCheckpoint =
  { weights :: ModelWeights
  , lmHead :: NDArray D2  -- [vocab, hidden]
  }

-- | Adapter that maps a Llama-style safetensors checkpoint into our
-- | `ModelWeights` record. Llama names look like:
-- |
-- |   model.embed_tokens.weight
-- |   model.layers.0.input_layernorm.weight
-- |   model.layers.0.self_attn.q_proj.weight
-- |   model.layers.0.self_attn.k_proj.weight
-- |   model.layers.0.self_attn.v_proj.weight
-- |   model.layers.0.self_attn.o_proj.weight
-- |   model.layers.0.post_attention_layernorm.weight
-- |   model.layers.0.mlp.gate_proj.weight
-- |   model.layers.0.mlp.up_proj.weight
-- |   model.layers.0.mlp.down_proj.weight
-- |   model.norm.weight
-- |   lm_head.weight     (when not weight-tied)
-- |
-- | PyTorch stores linear weights as `[out, in]`; our model expects
-- | `[in, out]` for `x @ w`, so we transpose on load.
-- |
-- | LM head: if the checkpoint contains `lm_head.weight` (tie_word_embeddings
-- | is false), we load it as the head's projection matrix. Otherwise we
-- | reuse the embedding table (weight-tied, the Llama-2 default). Both
-- | are stored unchanged in PyTorch `[vocab, hidden]` orientation;
-- | `unembed` transposes internally.
loadLlamaWeights
  :: ModelConfig
  -> SafetensorsMap
  -> Effect LoadedCheckpoint
loadLlamaWeights cfg st = do
  embedding <- getTensor st "model.embed_tokens.weight" :: Effect (NDArray D2)
  finalNorm <- getTensor st "model.norm.weight" :: Effect (NDArray D1)
  names <- tensorNames st
  lmHead <-
    if elem "lm_head.weight" names then
      getTensor st "lm_head.weight" :: Effect (NDArray D2)
    else
      ref embedding
  layers <- traverse (loadLayer st) (range 0 (cfg.nLayers - 1))
  pure { weights: { embedding, layers, finalNorm }, lmHead }

loadLayer :: SafetensorsMap -> Int -> Effect LayerWeights
loadLayer st i = do
  let prefix = "model.layers." <> show i <> "."
  attnNorm <- getTensor st (prefix <> "input_layernorm.weight") :: Effect (NDArray D1)
  -- Linear weights stored as [out, in] in PyTorch; transpose to [in, out].
  wq <- getAndTranspose st (prefix <> "self_attn.q_proj.weight")
  wk <- getAndTranspose st (prefix <> "self_attn.k_proj.weight")
  wv <- getAndTranspose st (prefix <> "self_attn.v_proj.weight")
  wo <- getAndTranspose st (prefix <> "self_attn.o_proj.weight")
  mlpNorm <- getTensor st (prefix <> "post_attention_layernorm.weight") :: Effect (NDArray D1)
  gp <- getAndTranspose st (prefix <> "mlp.gate_proj.weight")
  up <- getAndTranspose st (prefix <> "mlp.up_proj.weight")
  dp <- getAndTranspose st (prefix <> "mlp.down_proj.weight")
  pure
    { attnNorm
    , attn: { wq, wk, wv, wo }
    , mlpNorm
    , mlp: { gateProj: gp, upProj: up, downProj: dp }
    }
  where
  getAndTranspose s name = do
    w <- getTensor s name :: Effect (NDArray D2)
    wR <- ref w
    transpose wR
