-- | NN-shaped end-to-end demo for the shape-typed Tensor pipeline.
-- |
-- | This is the "would the type system actually work for transformer
-- | code?" stress test. It walks a forward pass through a tiny
-- | 1-layer attention-block-shaped pipeline on fully concrete `Lit`
-- | dimensions:
-- |
-- |     hidden    = 8
-- |     nHeads    = 2
-- |     headDim   = 4    (so qDim = nHeads * headDim = 8 = hidden)
-- |     seq       = 4
-- |     vocab     = 16
-- |
-- | Every shape transformation — Q/K/V matmul, head reshape, output
-- | reshape, residual add, scalar normalization, the LM head
-- | projection back to vocab — gets a precise `Tensor (S_ ...)` type.
-- | If any wiring is wrong, it's a compile error, not a runtime shape
-- | mismatch.
-- |
-- | This DOES NOT prove the deeper invariant `nHeads * headDim ==
-- | hidden` for *runtime* configs — that requires lifting
-- | `ModelConfig` to type-level integers (a separate refactor). It
-- | proves the operations themselves are sound and the type-level
-- | machinery composes through realistic NN shapes.
module Test.ShapeNN
  ( spec
  ) where

import Prelude

import Data.Array (length, replicate, (!!))
import Data.Maybe (Maybe(..), fromMaybe)
import Effect (Effect)
import Effect.Class (liftEffect)
import Test.Spec (Spec, describe, it)
import Test.Spec.Assertions (shouldEqual)

import Jax.Coerce (asArray2D)
import Jax.Core as Core
import Jax.Shape (Lit, S2, S3)
import Jax.Shape.Proxy (sProxy)
import Jax.Shape.Tensor (Tensor, unsafeAssumeShape, withRank)
import Jax.Shape.Tensor.Op as Op
import Jax.Shape.Tensor.Expr as TE

-- =============================================================================
-- Concrete shapes for the tiny transformer
-- =============================================================================

type Seq      = Lit 4
type Hidden   = Lit 8
type NHeads   = Lit 2
type HeadDim  = Lit 4
type QDim     = Lit 8     -- nHeads * headDim, statically equal to Hidden
type Vocab    = Lit 16

type SeqHidden    = S2 Seq Hidden                  -- [seq, hidden]
type SeqNhHd      = S3 Seq NHeads HeadDim          -- [seq, nHeads, headDim]
type HiddenQDim   = S2 Hidden QDim                 -- weights wq/wk/wv
type QDimHidden   = S2 QDim Hidden                 -- weights wo
type VocabHidden  = S2 Vocab Hidden                -- LM head
type SeqVocab     = S2 Seq Vocab                   -- logits

-- =============================================================================
-- Helpers
-- =============================================================================

mk2 :: forall m n. Array Number -> Array Int -> Effect (Tensor (S2 m n))
mk2 flat shape = do
  arr1 <- Core.array1D flat
  arr2 <- Core.reshape arr1 shape
  pure (unsafeAssumeShape (arr2 :: Core.NDArray Core.D2))

toArr2 :: forall m n. Tensor (S2 m n) -> Effect (Array (Array Number))
toArr2 t = do
  fr <- Core.toJs (withRank t :: Core.NDArray Core.D2)
  pure (asArray2D fr)

-- | Indexed lookup with a fallback for assertions.
at2 :: Array (Array Number) -> Int -> Int -> Number
at2 arr i j = fromMaybe 0.0 do
  row <- arr !! i
  row !! j

-- =============================================================================
-- The transformer-shaped pipeline
-- =============================================================================

-- | Q-projection then per-head reshape:
-- |   x : [seq, hidden] @ wq : [hidden, qDim] → [seq, qDim] → [seq, nh, hd]
-- |
-- | The reshape requires `Op.reshape` to discharge `Product s n` —
-- | both shapes have product 32 (= 4*8 = 4*2*4). With Lit dims, this
-- | is proved at compile time.
qProject
  :: Tensor SeqHidden
  -> Tensor HiddenQDim
  -> Effect (Tensor SeqNhHd)
qProject x wq = do
  qFlat <- TE.run (TE.lit x TE.**. TE.lit wq)   -- :: Tensor (S2 Seq QDim)
  Op.reshape (sProxy :: _ SeqNhHd) qFlat

-- | Inverse: head-reshape back to flat then o-projection.
oProject
  :: Tensor SeqNhHd
  -> Tensor QDimHidden
  -> Effect (Tensor SeqHidden)
oProject attnHeads wo = do
  flat <- Op.reshape (sProxy :: _ (S2 Seq QDim)) attnHeads
  TE.run (TE.lit flat TE.**. TE.lit wo)

-- | Residual skip + scalar scaling.
residualScale
  :: Tensor SeqHidden
  -> Tensor SeqHidden
  -> Number
  -> Effect (Tensor SeqHidden)
residualScale x attnOut scale =
  TE.run (TE.mulScalarT (TE.lit x TE.+. TE.lit attnOut) scale)

-- | Project to vocab logits via tied embedding (shape `[vocab, hidden]`).
unembed
  :: Tensor SeqHidden
  -> Tensor VocabHidden       -- embedding stored as [vocab, hidden]
  -> Effect (Tensor SeqVocab)
unembed y emb = do
  embT <- Op.transpose emb     -- :: Tensor (S2 Hidden Vocab)
  TE.run (TE.lit y TE.**. TE.lit embT)

-- =============================================================================
-- Spec
-- =============================================================================

spec :: Spec Unit
spec = describe "Shape-typed NN pipeline (Lit dimensions)" do
  it "qProject: matmul + reshape (Product 4*8 = 4*2*4 = 32)" do
    -- All-ones inputs make the matmul produce 8 in every cell of
    -- [seq, qDim] = [4, 8], which reshape into [4, 2, 4] preserves.
    topAxis <- liftEffect do
      x <- mk2 (replicate 32 1.0) [ 4, 8 ] :: Effect (Tensor SeqHidden)
      wq <- mk2 (replicate 64 1.0) [ 8, 8 ] :: Effect (Tensor HiddenQDim)
      qHeads <- qProject x wq
      Core.dimAt (withRank qHeads :: Core.NDArray Core.D3) 0
    topAxis `shouldEqual` 4

  it "oProject: head-reshape inverse + matmul" do
    rows <- liftEffect do
      attnHeads <- do
        flat1 <- Core.array1D (replicate 32 1.0)
        r3 <- Core.reshape flat1 [ 4, 2, 4 ]
        pure (unsafeAssumeShape (r3 :: Core.NDArray Core.D3) :: Tensor SeqNhHd)
      wo <- mk2 (replicate 64 1.0) [ 8, 8 ] :: Effect (Tensor QDimHidden)
      out <- oProject attnHeads wo
      toArr2 out
    -- 4x8 result; every cell = 8 (sum of 8 ones in matmul row)
    length rows `shouldEqual` 4
    at2 rows 0 0 `shouldEqual` 8.0
    at2 rows 3 7 `shouldEqual` 8.0

  it "residualScale: broadcast add + scalar mul" do
    rows <- liftEffect do
      x <- mk2 (replicate 32 2.0) [ 4, 8 ] :: Effect (Tensor SeqHidden)
      a <- mk2 (replicate 32 3.0) [ 4, 8 ] :: Effect (Tensor SeqHidden)
      r <- residualScale x a 0.5
      toArr2 r
    -- (2 + 3) * 0.5 = 2.5
    at2 rows 0 0 `shouldEqual` 2.5
    at2 rows 3 7 `shouldEqual` 2.5

  it "full pipeline: x @ wq → reshape → reshape → @ wo → @ embT" do
    -- End-to-end: types compose all the way through. If any link is
    -- wrong, this binding doesn't typecheck.
    rows <- liftEffect do
      x <- mk2 (replicate 32 1.0) [ 4, 8 ] :: Effect (Tensor SeqHidden)
      wq <- mk2 (replicate 64 1.0) [ 8, 8 ] :: Effect (Tensor HiddenQDim)
      wo <- mk2 (replicate 64 1.0) [ 8, 8 ] :: Effect (Tensor QDimHidden)
      emb <- mk2 (replicate 128 1.0) [ 16, 8 ] :: Effect (Tensor VocabHidden)
      qHeads <- qProject x wq
      flatBack <- oProject qHeads wo
      logits <- unembed flatBack emb
      toArr2 logits
    -- [seq, vocab] = [4, 16]
    length rows `shouldEqual` 4
    case rows !! 0 of
      Just row -> length row `shouldEqual` 16
      Nothing -> 0 `shouldEqual` 16   -- forced fail
