-- | Transformer block + full-stack forwards. Internal helpers run in
-- | `Forward = ReaderT { cfg :: ModelConfig } Effect`, removing
-- | `cfg` from every internal signature; public functions
-- | (`forwardLogits`, `forwardCached`, etc.) keep their Effect-typed
-- | API and `runReaderT` once at the entry point.
-- |
-- | `T.run` is `MonadEffect`-polymorphic, so DSL operations don't need
-- | `liftEffect` inside `Forward`.
module Jax.NN.Block
  ( ModelConfig
  , AttentionWeights
  , MLPWeights
  , LayerWeights
  , ModelWeights
  -- * Dim symbols (re-exported for typed callers)
  , Hidden
  , NHeads
  , NKvHeads
  , HeadDim
  , QDim
  , KvDim
  , Intermediate
  , Vocab
  , transformerBlock
  , transformerStack
  , transformerBlocksAndNorm
  , forwardLogits
  , forwardLogitsWithHead
  , forwardCachedWithHead
  , emptyKVCacheStack
  , forwardCached
  , refModelWeights
  ) where

import Prelude hiding (add)

import Control.Monad.Reader.Trans (ReaderT, ask, runReaderT)
import Data.Array (snoc, zip) as Array
import Data.Foldable (foldM)
import Data.Traversable (traverse)
import Data.Tuple (Tuple(..))
import Effect (Effect)
import Effect.Class (liftEffect)
import Jax.Core
  ( D1
  , D2
  , D3
  , NDArray
  , concatAxis
  , dimAt
  , ref
  , repeatAxis
  , reshape
  , sliceAxis
  , zeros
  )
import Jax.NN.Attention (KVCache, KVCacheStack, attention, attentionNoMask)
import Jax.NN.Embed (embed, unembed)
import Jax.NN.MLP (mlp)
import Jax.NN.RMSNorm (rmsnorm)
import Jax.NN.RoPE (RoPETables, applyRoPE)
import Jax.Shape (class RankOf, Var, S1, S2)
import Jax.Shape.Tensor (Tensor, refT, unsafeAssumeShape, withRank)
import Jax.Tensor as T

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

-- =============================================================================
-- Dim symbols for the weight types
-- =============================================================================
--
-- These are `Var Symbol` aliases — same-named Vars unify across the
-- weight record so role-swap bugs (e.g. passing wq where wk is
-- expected, or `embedding` where `lm_head` is expected) become
-- compile errors.
--
-- The multiplicative invariants (`QDim = NHeads * HeadDim`, `KvDim =
-- NKvHeads * HeadDim`) are NOT proved at the type level — those would
-- require lifting `ModelConfig` to type-level Ints, a separate
-- refactor. Here `QDim` and `KvDim` are just distinct Symbols, so
-- the type system enforces "this is the q-projection's column dim"
-- without proving it equals `nHeads * headDim`.

type Hidden       = Var "hidden"
type NHeads       = Var "nHeads"
type NKvHeads     = Var "nKvHeads"
type HeadDim      = Var "headDim"
type QDim         = Var "qDim"          -- = nHeads * headDim (unproved)
type KvDim        = Var "kvDim"         -- = nKvHeads * headDim (unproved)
type Intermediate = Var "intermediate"
type Vocab        = Var "vocab"

-- | Linear projection weights for one attention layer.
type AttentionWeights =
  { wq :: Tensor (S2 Hidden QDim)
  , wk :: Tensor (S2 Hidden KvDim)
  , wv :: Tensor (S2 Hidden KvDim)
  , wo :: Tensor (S2 QDim Hidden)
  }

-- | SwiGLU MLP weights.
type MLPWeights =
  { gateProj :: Tensor (S2 Hidden Intermediate)
  , upProj   :: Tensor (S2 Hidden Intermediate)
  , downProj :: Tensor (S2 Intermediate Hidden)
  }

-- | All weights for one transformer layer.
type LayerWeights =
  { attnNorm :: Tensor (S1 Hidden)
  , attn :: AttentionWeights
  , mlpNorm :: Tensor (S1 Hidden)
  , mlp :: MLPWeights
  }

-- | All model weights. The LM head is taken to be weight-tied to
-- | `embedding`; for *untied* checkpoints (`tie_word_embeddings: false`
-- | in HF configs) thread the separate head tensor through
-- | `forwardLogitsWith` / `forwardCachedWith` instead of putting it on
-- | this record. Keeping `ModelWeights` rectangular means it stays a
-- | clean autodiff pytree for training.
type ModelWeights =
  { embedding :: Tensor (S2 Vocab Hidden)
  , layers :: Array LayerWeights
  , finalNorm :: Tensor (S1 Hidden)
  }

-- =============================================================================
-- Internal helpers: bridge typed weight tensors to rank-only NDArray.
-- =============================================================================
--
-- The rank-only forward path (which uses Jax.Tensor's T DSL, jax-js
-- ops, attention / mlp / rmsnorm, applyRoPE) consumes `NDArray d`.
-- Typed weight fields (`Tensor (S2 ...)`) share a runtime
-- representation, so the bridge is the `withRank` typed helper —
-- rank-checked at the type level via `RankOf`, phantom-only at
-- runtime. `asD2` / `asD1` are local aliases that pin the result rank
-- where the surrounding context is rank-polymorphic (e.g. inside a
-- `T.lit ...` whose result rank is determined elsewhere).
--
-- Going the other direction (NDArray d → Tensor s) is needed by
-- `unsafeAssumeShape`, used at boundaries that *produce* weight
-- values (loaders, allocators).

asD2 :: forall s. RankOf s D2 => Tensor s -> NDArray D2
asD2 = withRank

asD1 :: forall s. RankOf s D1 => Tensor s -> NDArray D1
asD1 = withRank

-- =============================================================================
-- Internal: Forward = ReaderT { cfg } Effect
-- =============================================================================

type Env = { cfg :: ModelConfig }
type Forward = ReaderT Env Effect

-- | Repeat-interleave kv heads up to n_q heads. Shape transformation:
-- |   `[seq, n_kv, head_dim]` → `[seq, n_q, head_dim]`
-- | where each kv head is duplicated G = n_q / n_kv times consecutively.
-- | No-op when n_kv == n_q. Consumes its argument.
expandKV :: NDArray D3 -> Forward (NDArray D3)
expandKV x = do
  { cfg } <- ask
  if cfg.nHeads == cfg.nKvHeads then pure x
  else liftEffect $ repeatAxis x (cfg.nHeads / cfg.nKvHeads) 1

-- | Attention sub-block: norm + Q/K/V projection + RoPE + SDPA + output
-- | projection. Returns the same shape as the (already-normed) input.
attentionForward
  :: AttentionWeights
  -> Int          -- seqLen
  -> NDArray D3   -- cos slice, broadcast-shaped [seq, 1, head_dim/2]
  -> NDArray D3   -- sin slice, same
  -> NDArray D2   -- x_normed [seq, hidden]
  -> Forward (NDArray D2)
attentionForward w seqLen cosSlice sinSlice xNormed = do
  { cfg } <- ask
  let halfDim = cfg.headDim / 2
      qDim = cfg.nHeads * cfg.headDim
      xT = T.lit xNormed
  q <- T.run (T.reshapeT (xT T.**. T.lit (asD2 w.wq)) [ seqLen, cfg.nHeads, cfg.headDim ])
  k <- T.run (T.reshapeT (xT T.**. T.lit (asD2 w.wk)) [ seqLen, cfg.nKvHeads, cfg.headDim ])
  v <- T.run (T.reshapeT (xT T.**. T.lit (asD2 w.wv)) [ seqLen, cfg.nKvHeads, cfg.headDim ])
  qRot <- liftEffect $ applyRoPE halfDim q cosSlice sinSlice
  kRot <- liftEffect $ applyRoPE halfDim k cosSlice sinSlice
  kFull <- expandKV kRot
  vFull <- expandKV v
  attnOut <- liftEffect $ attention qRot kFull vFull
  T.run (T.reshapeT (T.lit attnOut) [ seqLen, qDim ] T.**. T.lit (asD2 w.wo))

-- | One transformer block with residuals:
-- |   h = x + attention(rmsnorm(x))
-- |   y = h + mlp(rmsnorm(h))
transformerBlock
  :: LayerWeights
  -> Int           -- seqLen
  -> NDArray D3    -- cos slice
  -> NDArray D3    -- sin slice
  -> NDArray D2    -- x [seq, hidden]
  -> Forward (NDArray D2)
transformerBlock lw seqLen cosSlice sinSlice x = do
  { cfg } <- ask
  xNormed <- liftEffect $ rmsnorm cfg.normEps x (asD1 lw.attnNorm)
  attnOut <- attentionForward lw.attn seqLen cosSlice sinSlice xNormed
  h <- T.run (T.lit x T.+. T.lit attnOut)
  hNormed <- liftEffect $ rmsnorm cfg.normEps h (asD1 lw.mlpNorm)
  mlpOut <- liftEffect $ mlp hNormed (asD2 lw.mlp.gateProj) (asD2 lw.mlp.upProj) (asD2 lw.mlp.downProj)
  T.run (T.lit h T.+. T.lit mlpOut)

-- | Full stack: embed → n × transformerBlock → finalNorm. Returns the
-- | final hidden states (not yet projected to logits).
transformerStack
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> NDArray D1   -- token IDs [seq]
  -> Effect (NDArray D2)
transformerStack cfg w rope ids = do
  seqLen <- dimAt ids 0
  hidden0 <- embed (asD2 w.embedding) ids
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
  -- Slice axis 0 (seq dim) of the precomputed RoPE tables and reshape
  -- to broadcast against the per-head tensor `[seq, n_heads, head_dim]`.
  cosFullR <- ref rope.cos
  cosSlice2 <- sliceAxis cosFullR 0 0 seqLen
  cosSlice3 <- reshape cosSlice2 [ seqLen, 1, halfDim ]
  sinFullR <- ref rope.sin
  sinSlice2 <- sliceAxis sinFullR 0 0 seqLen
  sinSlice3 <- reshape sinSlice2 [ seqLen, 1, halfDim ]
  hiddenN <- foldM
    (\h lw -> runReaderT
        (transformerBlock lw seqLen cosSlice3 sinSlice3 h)
        { cfg })
    hidden0
    w.layers
  rmsnorm cfg.normEps hiddenN (asD1 w.finalNorm)

-- | Full forward including LM head: returns logits over the vocabulary.
forwardLogits
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> NDArray D1
  -> Effect (NDArray D2)
forwardLogits cfg w rope ids =
  forwardLogitsWithHead cfg w (asD2 w.embedding) rope ids

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
-- | over the full K/V, and projects back.
attentionForwardCached
  :: AttentionWeights
  -> Int           -- newSeq
  -> NDArray D3    -- cos slice for the new positions
  -> NDArray D3    -- sin slice for the new positions
  -> KVCache       -- previous cache
  -> NDArray D2    -- x_normed [newSeq, hidden]
  -> Forward { newCache :: KVCache, output :: NDArray D2 }
attentionForwardCached w newSeq cosSlice sinSlice prevCache xNormed = do
  { cfg } <- ask
  let halfDim = cfg.headDim / 2
      qDim = cfg.nHeads * cfg.headDim
      xT = T.lit xNormed
  q <- T.run (T.reshapeT (xT T.**. T.lit (asD2 w.wq)) [ newSeq, cfg.nHeads, cfg.headDim ])
  newK <- T.run (T.reshapeT (xT T.**. T.lit (asD2 w.wk)) [ newSeq, cfg.nKvHeads, cfg.headDim ])
  newV <- T.run (T.reshapeT (xT T.**. T.lit (asD2 w.wv)) [ newSeq, cfg.nKvHeads, cfg.headDim ])
  qRot <- liftEffect $ applyRoPE halfDim q cosSlice sinSlice
  newKRot <- liftEffect $ applyRoPE halfDim newK cosSlice sinSlice
  fullK <- liftEffect $ concatAxis [ prevCache.k, newKRot ] 0
  fullV <- liftEffect $ concatAxis [ prevCache.v, newV ] 0
  fullKRA <- liftEffect $ ref fullK
  fullVRA <- liftEffect $ ref fullV
  fullKExp <- expandKV fullKRA
  fullVExp <- expandKV fullVRA
  -- Mask choice: prefill (newSeq == kvSeq) gets a standard causal mask;
  -- decode (newSeq < kvSeq) needs *no* mask, since jax-js's isCausal
  -- aligns Q/K at position 0 and would mask the decode query down to
  -- seeing only K[0] — a known degenerate loop.
  kvSeq <- liftEffect $ dimAt fullK 0
  attnOut <- liftEffect $
    if newSeq == kvSeq
      then attention qRot fullKExp fullVExp
      else attentionNoMask qRot fullKExp fullVExp
  output <- T.run (T.reshapeT (T.lit attnOut) [ newSeq, qDim ] T.**. T.lit (asD2 w.wo))
  pure { newCache: { k: fullK, v: fullV }, output }

-- | One transformer block, KVCache-aware.
transformerBlockCached
  :: LayerWeights
  -> Int
  -> NDArray D3
  -> NDArray D3
  -> KVCache
  -> NDArray D2
  -> Forward { newCache :: KVCache, output :: NDArray D2 }
transformerBlockCached lw newSeq cosSlice sinSlice prevCache x = do
  { cfg } <- ask
  xNormed <- liftEffect $ rmsnorm cfg.normEps x (asD1 lw.attnNorm)
  { newCache, output: attnOut } <-
    attentionForwardCached lw.attn newSeq cosSlice sinSlice prevCache xNormed
  h <- T.run (T.lit x T.+. T.lit attnOut)
  hNormed <- liftEffect $ rmsnorm cfg.normEps h (asD1 lw.mlpNorm)
  mlpOut <- liftEffect $ mlp hNormed (asD2 lw.mlp.gateProj) (asD2 lw.mlp.upProj) (asD2 lw.mlp.downProj)
  out <- T.run (T.lit h T.+. T.lit mlpOut)
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
  forwardCachedWithHead cfg w (asD2 w.embedding) rope cache startPos newIds

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
  newSeq <- dimAt newIds 0
  let halfDim = cfg.headDim / 2
  hidden0 <- embed (asD2 w.embedding) newIds
  cosFullR <- ref rope.cos
  cosSlice2 <- sliceAxis cosFullR 0 startPos (startPos + newSeq)
  cosSlice3 <- reshape cosSlice2 [ newSeq, 1, halfDim ]
  sinFullR <- ref rope.sin
  sinSlice2 <- sliceAxis sinFullR 0 startPos (startPos + newSeq)
  sinSlice3 <- reshape sinSlice2 [ newSeq, 1, halfDim ]
  let layerPairs = Array.zip w.layers cache
  result <- foldM
    ( \acc (Tuple lw lc) -> do
        r <- runReaderT
          (transformerBlockCached lw newSeq cosSlice3 sinSlice3 lc acc.hidden)
          { cfg }
        pure { hidden: r.output, cachesRev: Array.snoc acc.cachesRev r.newCache }
    )
    { hidden: hidden0, cachesRev: [] }
    layerPairs
  hNormed <- rmsnorm cfg.normEps result.hidden (asD1 w.finalNorm)
  logits <- unembed hNormed head
  pure { newCache: result.cachesRev, logits }

-- | Refcount-bump every NDArray leaf in a `ModelWeights`. Returns a
-- | structurally-identical copy where each tensor is a fresh handle
-- | sharing the same underlying buffer (refcount up by 1 per leaf).
-- |
-- | Use this when you need to pass a `ModelWeights` to an op that
-- | consumes it (e.g. `Optax.initT`, which moves its argument into
-- | the optimizer state) but you also want to keep the original alive
-- | for the next call.
-- |
-- | The shape is hand-rolled rather than via heterogeneous-mapping
-- | because a generic `refTree` would need to thread `Effect` through
-- | the record/array traversals — significantly more wiring than the
-- | one ModelWeights instance buys.
refModelWeights :: ModelWeights -> Effect ModelWeights
refModelWeights w = do
  emb <- refT w.embedding
  fn <- refT w.finalNorm
  layers <- traverse refLayer w.layers
  pure { embedding: emb, layers, finalNorm: fn }
  where
  refLayer lw = do
    an <- refT lw.attnNorm
    wq <- refT lw.attn.wq
    wk <- refT lw.attn.wk
    wv <- refT lw.attn.wv
    wo <- refT lw.attn.wo
    mn <- refT lw.mlpNorm
    gp <- refT lw.mlp.gateProj
    up <- refT lw.mlp.upProj
    dp <- refT lw.mlp.downProj
    pure
      { attnNorm: an
      , attn: { wq, wk, wv, wo }
      , mlpNorm: mn
      , mlp: { gateProj: gp, upProj: up, downProj: dp }
      }
