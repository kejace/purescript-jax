module Jax.NN.Attention
  ( KVCache
  , KVCacheStack
  , attention
  , attentionNoMask
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn4, runEffectFn4)
import Jax.Core (D3, NDArray, ref)

-- | Per-layer key/value cache. Shapes:
-- |   k :: NDArray D3 — [seq_len, n_kv_heads, head_dim]
-- |   v :: NDArray D3 — same shape as k
-- |
-- | These are written-once-per-decode-step and held across decode steps.
-- | Phase 3 inference loop will append fresh K/V to the cache after each
-- | step. For Phase 2 we only define the type.
type KVCache = { k :: NDArray D3, v :: NDArray D3 }

-- | One KVCache per transformer layer.
type KVCacheStack = Array KVCache

foreign import dotProductAttentionImpl
  :: forall d
   . EffectFn4 (NDArray d) (NDArray d) (NDArray d) Boolean (NDArray d)

-- | Scaled dot-product attention with causal mask.
-- |
-- | Inputs (rank 3 or 4 — jax-js auto-handles batched [B, L, N, H] vs
-- | unbatched [L, N, H]):
-- |   q — [seq_q, n_heads, head_dim]
-- |   k — [seq_kv, n_kv_heads, head_dim]
-- |   v — [seq_kv, n_kv_heads, head_dim]
-- |
-- | When n_kv_heads < n_heads, jax-js performs GQA broadcast across the
-- | head axis automatically (no manual `repeat` needed).
-- |
-- | Output: same shape as q.
-- |
-- | The default scale (1/sqrt(head_dim)) is applied. Inputs are *consumed*
-- | by jax-js per the standard convention; we ref-bump on entry so callers
-- | can keep using q/k/v afterwards (e.g. to fold k/v into the KVCache).
attention
  :: forall d
   . NDArray d
  -> NDArray d
  -> NDArray d
  -> Effect (NDArray d)
attention q k v = do
  qR <- ref q
  kR <- ref k
  vR <- ref v
  runEffectFn4 dotProductAttentionImpl qR kR vR true

-- | Same as `attention` but with NO mask. Use for KV-cached decode
-- | steps where `q` is a single new token and `k`/`v` cover the cache
-- | + the new token: there is nothing to mask (the lone query is at
-- | the latest position and may attend to all of K/V).
-- |
-- | Plain `attention` (causal) only does the right thing when
-- | `seq_q == seq_kv`, because jax-js's `isCausal` aligns `q` and `k`
-- | at position 0. For decode with cached K/V the new query would be
-- | masked off everything but position 0 — the model loses context
-- | and degenerates to a one-token loop.
attentionNoMask
  :: forall d
   . NDArray d
  -> NDArray d
  -> NDArray d
  -> Effect (NDArray d)
attentionNoMask q k v = do
  qR <- ref q
  kR <- ref k
  vR <- ref v
  runEffectFn4 dotProductAttentionImpl qR kR vR false
