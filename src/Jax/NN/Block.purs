module Jax.NN.Block
  ( ModelConfig
  , AttentionWeights
  , MLPWeights
  , LayerWeights
  , ModelWeights
  , transformerBlock
  , transformerStack
  , transformerBlocksAndNorm
  , forwardLogits
  , forwardLogitsWithHead
  , forwardCachedWithHead
  , emptyKVCacheStack
  , forwardCached
  ) where

import Prelude hiding (add)

import Data.Array (head, snoc, zip) as Array
import Data.Foldable (foldM)
import Data.Maybe (fromMaybe)
import Data.Tuple (Tuple(..))
import Effect (Effect)
import Jax.Core
  ( D1
  , D2
  , D3
  , NDArray
  , add
  , concatAxis
  , matmul
  , ref
  , repeatAxis
  , reshape
  , shape
  , sliceAxis
  , zeros
  )
import Jax.NN.Attention (KVCache, KVCacheStack, attention, attentionNoMask)
import Jax.NN.Embed (embed, unembed)
import Jax.NN.MLP (mlp)
import Jax.NN.RMSNorm (rmsnorm)
import Jax.NN.RoPE (RoPETables, applyRoPE)

-- | Static model hyperparameters (no tensors here).
type ModelConfig =
  { hidden :: Int
  , nHeads :: Int
  , nKvHeads :: Int
  , headDim :: Int
  , intermediate :: Int
  , nLayers :: Int
  , maxSeqLen :: Int
  , vocabSize :: Int
  , ropeTheta :: Number
  , normEps :: Number
  }

-- | Linear projection weights for one attention layer.
type AttentionWeights =
  { wq :: NDArray D2  -- [hidden, n_heads * head_dim]
  , wk :: NDArray D2  -- [hidden, n_kv_heads * head_dim]
  , wv :: NDArray D2  -- [hidden, n_kv_heads * head_dim]
  , wo :: NDArray D2  -- [n_heads * head_dim, hidden]
  }

-- | SwiGLU MLP weights.
type MLPWeights =
  { gateProj :: NDArray D2  -- [hidden, intermediate]
  , upProj :: NDArray D2    -- [hidden, intermediate]
  , downProj :: NDArray D2  -- [intermediate, hidden]
  }

-- | All weights for one transformer layer.
type LayerWeights =
  { attnNorm :: NDArray D1  -- [hidden]
  , attn :: AttentionWeights
  , mlpNorm :: NDArray D1
  , mlp :: MLPWeights
  }

-- | All model weights. The LM head is taken to be weight-tied to
-- | `embedding`; for *untied* checkpoints (`tie_word_embeddings: false`
-- | in HF configs) thread the separate head tensor through
-- | `forwardLogitsWith` / `forwardCachedWith` instead of putting it on
-- | this record. Keeping `ModelWeights` rectangular means it stays a
-- | clean autodiff pytree for training.
type ModelWeights =
  { embedding :: NDArray D2  -- [vocab, hidden]
  , layers :: Array LayerWeights
  , finalNorm :: NDArray D1  -- [hidden]
  }

-- | Repeat-interleave kv heads up to n_q heads. Shape transformation:
-- |   `[seq, n_kv, head_dim]` → `[seq, n_q, head_dim]`
-- | where each kv head is duplicated G = n_q / n_kv times consecutively.
-- | No-op when n_kv == n_q. Consumes its argument.
expandKV :: ModelConfig -> NDArray D3 -> Effect (NDArray D3)
expandKV cfg x =
  if cfg.nHeads == cfg.nKvHeads then pure x
  else repeatAxis x (cfg.nHeads / cfg.nKvHeads) 1

-- | Attention sub-block: norm + Q/K/V projection + RoPE + SDPA + output
-- | projection. Returns the same shape as the (already-normed) input.
attentionForward
  :: ModelConfig
  -> AttentionWeights
  -> Int          -- seqLen
  -> NDArray D3   -- cos slice, broadcast-shaped [seq, 1, head_dim/2]
  -> NDArray D3   -- sin slice, same
  -> NDArray D2   -- x_normed [seq, hidden]
  -> Effect (NDArray D2)
attentionForward cfg w seqLen cosSlice sinSlice xNormed = do
  let halfDim = cfg.headDim / 2
      qDim = cfg.nHeads * cfg.headDim
  -- Q projection + reshape to multi-head
  xR1 <- ref xNormed
  wqR <- ref w.wq
  qFlat <- matmul xR1 wqR
  q <- reshape qFlat [ seqLen, cfg.nHeads, cfg.headDim ]
  -- K projection + reshape
  xR2 <- ref xNormed
  wkR <- ref w.wk
  kFlat <- matmul xR2 wkR
  k <- reshape kFlat [ seqLen, cfg.nKvHeads, cfg.headDim ]
  -- V projection + reshape
  xR3 <- ref xNormed
  wvR <- ref w.wv
  vFlat <- matmul xR3 wvR
  v <- reshape vFlat [ seqLen, cfg.nKvHeads, cfg.headDim ]
  -- RoPE on Q and K (rank-polymorphic; cos/sin already shaped to broadcast)
  cR1 <- ref cosSlice
  sR1 <- ref sinSlice
  qRot <- applyRoPE halfDim q cR1 sR1
  cR2 <- ref cosSlice
  sR2 <- ref sinSlice
  kRot <- applyRoPE halfDim k cR2 sR2
  -- GQA expansion: pre-repeat-interleave kv heads up to n_q heads so
  -- jax-js's SDPA sees N == K and skips its internal `tile`-based
  -- expansion. The internal tile uses the wrong head pairing for HF
  -- Llama (which trains q_head_i ↔ kv_head_(i // G), matching
  -- repeat_interleave, not tile).
  kFull <- expandKV cfg kRot
  vFull <- expandKV cfg v
  -- SDPA
  attnOut <- attention qRot kFull vFull
  -- Reshape multi-head output back to flat dim and project
  attnOutFlat <- reshape attnOut [ seqLen, qDim ]
  woR <- ref w.wo
  matmul attnOutFlat woR

-- | One transformer block with residuals:
-- |   h = x + attention(rmsnorm(x))
-- |   y = h + mlp(rmsnorm(h))
transformerBlock
  :: ModelConfig
  -> LayerWeights
  -> Int           -- seqLen
  -> NDArray D3    -- cos slice
  -> NDArray D3    -- sin slice
  -> NDArray D2    -- x [seq, hidden]
  -> Effect (NDArray D2)
transformerBlock cfg lw seqLen cosSlice sinSlice x = do
  -- Pre-attention norm
  xR1 <- ref x
  anR <- ref lw.attnNorm
  xNormed <- rmsnorm cfg.normEps xR1 anR
  -- Attention sub-block
  attnOut <- attentionForward cfg lw.attn seqLen cosSlice sinSlice xNormed
  -- Residual: h = x + attn_out
  xR2 <- ref x
  h <- add xR2 attnOut
  -- Pre-MLP norm
  hR1 <- ref h
  mnR <- ref lw.mlpNorm
  hNormed <- rmsnorm cfg.normEps hR1 mnR
  -- MLP
  mlpOut <- mlp hNormed lw.mlp.gateProj lw.mlp.upProj lw.mlp.downProj
  -- Residual: out = h + mlp_out
  add h mlpOut

-- | Full stack: embed → n × transformerBlock → finalNorm. Returns the
-- | final hidden states (not yet projected to logits).
transformerStack
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> NDArray D1   -- token IDs [seq]
  -> Effect (NDArray D2)
transformerStack cfg w rope ids = do
  idsShape <- shape ids
  let seqLen = fromMaybe 0 (Array.head idsShape)
  hidden0 <- embed w.embedding ids
  transformerBlocksAndNorm cfg w rope seqLen hidden0

-- | Variant that takes already-embedded `hidden0 :: [seq, hidden]` and
-- | runs only the post-embedding pipeline (transformer blocks + final
-- | norm). Useful when the caller wants a custom embedding path — e.g.
-- | autodiff-friendly `one_hot @ embedding_table` instead of `take`,
-- | since jax-js's gather transpose rule is not yet implemented.
transformerBlocksAndNorm
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Int           -- seqLen (caller knows this from hidden0's shape)
  -> NDArray D2    -- hidden0 [seq, hidden]
  -> Effect (NDArray D2)
transformerBlocksAndNorm cfg w rope seqLen hidden0 = do
  let halfDim = cfg.headDim / 2
  cosFullR <- ref rope.cos
  cosSlice2 <- sliceAxis cosFullR 0 0 seqLen
  cosSlice3 <- reshape cosSlice2 [ seqLen, 1, halfDim ]
  sinFullR <- ref rope.sin
  sinSlice2 <- sliceAxis sinFullR 0 0 seqLen
  sinSlice3 <- reshape sinSlice2 [ seqLen, 1, halfDim ]
  hiddenN <- foldM
    (\h lw -> transformerBlock cfg lw seqLen cosSlice3 sinSlice3 h)
    hidden0
    w.layers
  hR <- ref hiddenN
  fnR <- ref w.finalNorm
  rmsnorm cfg.normEps hR fnR

-- | Full forward including LM head: returns logits over the vocabulary.
forwardLogits
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> NDArray D1
  -> Effect (NDArray D2)
forwardLogits cfg w rope ids =
  forwardLogitsWithHead cfg w w.embedding rope ids

-- | `forwardLogits` variant that takes an explicit LM-head projection
-- | matrix `[vocab, hidden]` separate from `weights.embedding`. Use this
-- | for checkpoints where `tie_word_embeddings: false` (e.g. some HF
-- | Llama-arch models): the loader supplies the explicit `lm_head.weight`
-- | tensor. For tied checkpoints, `head` is just `weights.embedding`.
forwardLogitsWithHead
  :: ModelConfig
  -> ModelWeights
  -> NDArray D2     -- ^ LM-head projection [vocab, hidden]
  -> RoPETables
  -> NDArray D1
  -> Effect (NDArray D2)
forwardLogitsWithHead cfg w head rope ids = do
  hidden <- transformerStack cfg w rope ids
  unembed hidden head

-- =============================================================================
-- KVCache-accelerated forward (Phase 3 fast path)
-- =============================================================================
--
-- The non-cached `forwardLogits` recomputes the full attention each step
-- (O(n²) decode). With KVCache, decode steps take O(n) by reusing K/V
-- from earlier positions. The cache is a per-layer pair of tensors:
--   k :: [seq_so_far, n_kv_heads, head_dim]
--   v :: [seq_so_far, n_kv_heads, head_dim]
-- Initial cache is length-0 along the seq axis.

-- | Build an empty KVCacheStack — `n_layers` entries, each with shape
-- | `[0, n_kv_heads, head_dim]`. The empty leading dim works because
-- | `concatAxis` with a zero-length tensor and a fresh tensor yields the
-- | fresh tensor verbatim.
emptyKVCacheStack :: ModelConfig -> Effect KVCacheStack
emptyKVCacheStack cfg = traverseN cfg.nLayers \_ -> do
  k <- zeros [ 0, cfg.nKvHeads, cfg.headDim ] :: Effect (NDArray D3)
  v <- zeros [ 0, cfg.nKvHeads, cfg.headDim ] :: Effect (NDArray D3)
  pure { k, v }

-- | Allocate `n` items via `Effect`. Local helper; avoids pulling in
-- | `Data.Traversable` for one use.
traverseN :: forall a. Int -> (Int -> Effect a) -> Effect (Array a)
traverseN n f = go 0 []
  where
  go i acc
    | i >= n = pure acc
    | otherwise = do
        x <- f i
        go (i + 1) (Array.snoc acc x)

-- | Attention sub-block with KVCache. Computes new K/V for the new tokens,
-- | concatenates with the cached K/V along the seq axis, runs causal SDPA
-- | over the full K/V, and projects back. Returns the attention output
-- | and the *new* cache (the previous cache's K/V tensors are consumed
-- | by the concat operations and become part of the returned cache).
attentionForwardCached
  :: ModelConfig
  -> AttentionWeights
  -> Int           -- newSeq
  -> NDArray D3    -- cos slice for the new positions
  -> NDArray D3    -- sin slice for the new positions
  -> KVCache       -- previous cache
  -> NDArray D2    -- x_normed [newSeq, hidden]
  -> Effect { newCache :: KVCache, output :: NDArray D2 }
attentionForwardCached cfg w newSeq cosSlice sinSlice prevCache xNormed = do
  let halfDim = cfg.headDim / 2
      qDim = cfg.nHeads * cfg.headDim
  -- Q projection
  xR1 <- ref xNormed
  wqR <- ref w.wq
  qFlat <- matmul xR1 wqR
  q <- reshape qFlat [ newSeq, cfg.nHeads, cfg.headDim ]
  -- K projection
  xR2 <- ref xNormed
  wkR <- ref w.wk
  kFlat <- matmul xR2 wkR
  newK <- reshape kFlat [ newSeq, cfg.nKvHeads, cfg.headDim ]
  -- V projection
  xR3 <- ref xNormed
  wvR <- ref w.wv
  vFlat <- matmul xR3 wvR
  newV <- reshape vFlat [ newSeq, cfg.nKvHeads, cfg.headDim ]
  -- RoPE on Q and (new) K only — cached K already had RoPE applied
  cR1 <- ref cosSlice
  sR1 <- ref sinSlice
  qRot <- applyRoPE halfDim q cR1 sR1
  cR2 <- ref cosSlice
  sR2 <- ref sinSlice
  newKRot <- applyRoPE halfDim newK cR2 sR2
  -- Concat with cache along axis 0 (the seq axis)
  fullK <- concatAxis [ prevCache.k, newKRot ] 0
  fullV <- concatAxis [ prevCache.v, newV ] 0
  -- We'll need fullK and fullV both for attention AND to retain in the new
  -- cache, so bump them once before handing into attention.
  fullKRA <- ref fullK
  fullVRA <- ref fullV
  -- GQA expansion: see `attentionForward` for why we pre-expand here.
  fullKExp <- expandKV cfg fullKRA
  fullVExp <- expandKV cfg fullVRA
  -- Mask choice: when the cache is empty (prefill, newSeq == kvSeq) we
  -- want a standard causal mask. For single-token decode (newSeq == 1
  -- with non-empty cache) the single new query is at the latest
  -- position and must attend to all of K/V — *no* mask. jax-js's
  -- `isCausal` aligns Q/K at position 0 and would mask the decode
  -- query down to seeing only K[0], giving a degenerate "pu pu pu"
  -- loop.
  fullKShape <- shape fullK
  let kvSeq = fromMaybe 0 (Array.head fullKShape)
  attnOut <-
    if newSeq == kvSeq
      then attention qRot fullKExp fullVExp
      else attentionNoMask qRot fullKExp fullVExp
  -- Project output
  attnOutFlat <- reshape attnOut [ newSeq, qDim ]
  woR <- ref w.wo
  output <- matmul attnOutFlat woR
  pure { newCache: { k: fullK, v: fullV }, output }

-- | One transformer block, KVCache-aware.
transformerBlockCached
  :: ModelConfig
  -> LayerWeights
  -> Int
  -> NDArray D3
  -> NDArray D3
  -> KVCache
  -> NDArray D2
  -> Effect { newCache :: KVCache, output :: NDArray D2 }
transformerBlockCached cfg lw newSeq cosSlice sinSlice prevCache x = do
  xR1 <- ref x
  anR <- ref lw.attnNorm
  xNormed <- rmsnorm cfg.normEps xR1 anR
  { newCache, output: attnOut } <-
    attentionForwardCached cfg lw.attn newSeq cosSlice sinSlice prevCache xNormed
  xR2 <- ref x
  h <- add xR2 attnOut
  hR1 <- ref h
  mnR <- ref lw.mlpNorm
  hNormed <- rmsnorm cfg.normEps hR1 mnR
  mlpOut <- mlp hNormed lw.mlp.gateProj lw.mlp.upProj lw.mlp.downProj
  out <- add h mlpOut
  pure { newCache, output: out }

-- | Cached forward pass over the full layer stack. Handles both prefill
-- | (initial empty cache + full prompt) and decode (populated cache + one
-- | new token), distinguished only by the caller's `startPos` and the
-- | length of `newIds`.
forwardCached
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> KVCacheStack
  -> Int             -- ^ startPos (the position of the first new token)
  -> NDArray D1      -- ^ newIds (token IDs for the new positions)
  -> Effect { newCache :: KVCacheStack, logits :: NDArray D2 }
forwardCached cfg w rope cache startPos newIds =
  forwardCachedWithHead cfg w w.embedding rope cache startPos newIds

-- | `forwardCached` with an explicit LM-head projection. See
-- | `forwardLogitsWithHead` for when to use this.
forwardCachedWithHead
  :: ModelConfig
  -> ModelWeights
  -> NDArray D2     -- ^ LM-head projection [vocab, hidden]
  -> RoPETables
  -> KVCacheStack
  -> Int
  -> NDArray D1
  -> Effect { newCache :: KVCacheStack, logits :: NDArray D2 }
forwardCachedWithHead cfg w head rope cache startPos newIds = do
  newIdsShape <- shape newIds
  let
    newSeq = fromMaybe 0 (Array.head newIdsShape)
    halfDim = cfg.headDim / 2
  -- Embed
  hidden0 <- embed w.embedding newIds
  -- RoPE slices for the new positions only
  cosFullR <- ref rope.cos
  cosSlice2 <- sliceAxis cosFullR 0 startPos (startPos + newSeq)
  cosSlice3 <- reshape cosSlice2 [ newSeq, 1, halfDim ]
  sinFullR <- ref rope.sin
  sinSlice2 <- sliceAxis sinFullR 0 startPos (startPos + newSeq)
  sinSlice3 <- reshape sinSlice2 [ newSeq, 1, halfDim ]
  let layerPairs = Array.zip w.layers cache
  result <- foldM
    ( \acc (Tuple lw lc) -> do
        r <- transformerBlockCached cfg lw newSeq cosSlice3 sinSlice3 lc acc.hidden
        pure { hidden: r.output, cachesRev: Array.snoc acc.cachesRev r.newCache }
    )
    { hidden: hidden0, cachesRev: [] }
    layerPairs
  hR <- ref result.hidden
  fnR <- ref w.finalNorm
  hNormed <- rmsnorm cfg.normEps hR fnR
  logits <- unembed hNormed head
  pure { newCache: result.cachesRev, logits }
