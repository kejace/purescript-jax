-- | End-to-end tests for the typed `Tensor s` operation surface
-- | (`Jax.Shape.Tensor.Op` and `Jax.Shape.Tensor.Expr`).
-- |
-- | These tests verify the *runtime correctness* of the shape-typed
-- | wrappers — the type-level identities are covered by the
-- | compile-time tests in `Test.Shape`. Here we allocate concrete
-- | tensors, run them through typed ops, and assert numerical output
-- | matches a hand-computed reference.
module Test.ShapeOps
  ( spec
  ) where

import Prelude

import Effect as Effect
import Effect.Class (liftEffect)
import Test.Spec (Spec, describe, it)
import Test.Spec.Assertions (shouldEqual)

import Jax.Coerce (asArray2D)
import Jax.Core as Core
import Jax.Shape (Lit, S2)
import Jax.Shape.Proxy (sProxy)
import Jax.Shape.Tensor (Tensor, unsafeAssumeShape, withRank)
import Jax.Shape.Tensor.Op as Op
import Jax.Shape.Tensor.Expr as TE

-- =============================================================================
-- Test helpers
-- =============================================================================

-- | Read a rank-2 typed tensor back as an Array (Array Number).
toArr2
  :: forall m n
   . Tensor (S2 m n) -> Effect.Effect (Array (Array Number))
toArr2 t = do
  fr <- Core.toJs (withRank t :: Core.NDArray Core.D2)
  pure (asArray2D fr)

-- | Build a literal 2D Tensor from a flat row-major array. The two
-- | shape vars `m` `n` are existential — caller supplies them via
-- | type ascription on the `Effect (Tensor (S2 m n))` result.
mk2
  :: forall m n
   . Array Number -> Array Int -> Effect.Effect (Tensor (S2 m n))
mk2 flat shape = do
  arr1 <- Core.array1D flat
  arr2 <- Core.reshape arr1 shape
  pure (unsafeAssumeShape (arr2 :: Core.NDArray Core.D2))

-- =============================================================================
-- Spec
-- =============================================================================

spec :: Spec Unit
spec = describe "Shape-typed Tensor ops" do
  describe "Op (direct primitives)" do
    it "matmul [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        b <- mk2 [ 5.0, 6.0, 7.0, 8.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- Op.matmul a b
        toArr2 c
      result `shouldEqual` [ [ 19.0, 22.0 ], [ 43.0, 50.0 ] ]

    it "transpose [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ] [ 2, 3 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 3)))
        c <- Op.transpose a
        toArr2 c
      result `shouldEqual`
        [ [ 1.0, 4.0 ], [ 2.0, 5.0 ], [ 3.0, 6.0 ] ]

    it "reshape [[1,2,3],[4,5,6]] from [2,3] to [3,2]" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ] [ 2, 3 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 3)))
        c <- Op.reshape (sProxy :: _ (S2 (Lit 3) (Lit 2))) a
        toArr2 c
      result `shouldEqual`
        [ [ 1.0, 2.0 ], [ 3.0, 4.0 ], [ 5.0, 6.0 ] ]

    it "addScalar [[1,2],[3,4]] 10 = [[11,12],[13,14]]" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- Op.addScalar a 10.0
        toArr2 c
      result `shouldEqual` [ [ 11.0, 12.0 ], [ 13.0, 14.0 ] ]

    it "mulScalar [[1,2],[3,4]] 3 = [[3,6],[9,12]]" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- Op.mulScalar a 3.0
        toArr2 c
      result `shouldEqual` [ [ 3.0, 6.0 ], [ 9.0, 12.0 ] ]

    it "square [[1,2],[3,4]] = [[1,4],[9,16]]" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- Op.square a
        toArr2 c
      result `shouldEqual` [ [ 1.0, 4.0 ], [ 9.0, 16.0 ] ]

    it "add (broadcast same shape) elementwise" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        b <- mk2 [ 10.0, 20.0, 30.0, 40.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- Op.add a b
        toArr2 c
      result `shouldEqual` [ [ 11.0, 22.0 ], [ 33.0, 44.0 ] ]

  describe "Expr (deferred DSL)" do
    it "(x **. y) chains correctly" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        b <- mk2 [ 5.0, 6.0, 7.0, 8.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- TE.run (TE.lit a TE.**. TE.lit b)
        toArr2 c
      result `shouldEqual` [ [ 19.0, 22.0 ], [ 43.0, 50.0 ] ]

    it "(x +. x) = 2*x via DSL" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- TE.run (TE.lit a TE.+. TE.lit a)
        toArr2 c
      result `shouldEqual` [ [ 2.0, 4.0 ], [ 6.0, 8.0 ] ]

    it "transposeT after matmul (chained)" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        b <- mk2 [ 5.0, 6.0, 7.0, 8.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        c <- TE.run (TE.transposeT (TE.lit a TE.**. TE.lit b))
        toArr2 c
      result `shouldEqual` [ [ 19.0, 43.0 ], [ 22.0, 50.0 ] ]

    it "ref-bumping: lit a used twice doesn't consume a" do
      result <- liftEffect do
        a <- mk2 [ 1.0, 2.0, 3.0, 4.0 ] [ 2, 2 ]
              :: _ (Tensor (S2 (Lit 2) (Lit 2)))
        -- Each `lit a` ref-bumps; running this twice in a row
        -- shouldn't leave `a` consumed.
        c1 <- TE.run (TE.lit a TE.+. TE.lit a)
        c2 <- TE.run (TE.lit a TE.*. TE.lit a)
        r1 <- toArr2 c1
        r2 <- toArr2 c2
        pure { r1, r2 }
      result.r1 `shouldEqual` [ [ 2.0, 4.0 ], [ 6.0, 8.0 ] ]
      result.r2 `shouldEqual` [ [ 1.0, 4.0 ], [ 9.0, 16.0 ] ]

  describe "Allocators" do
    it "zeros (S2 (Lit 2) (Lit 3)) produces 2x3 of zeros" do
      result <- liftEffect do
        z <- Op.zeros (sProxy :: _ (S2 (Lit 2) (Lit 3)))
        toArr2 z
      result `shouldEqual` [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]

    it "ones (S2 (Lit 2) (Lit 3)) produces 2x3 of ones" do
      result <- liftEffect do
        o <- Op.ones (sProxy :: _ (S2 (Lit 2) (Lit 3)))
        toArr2 o
      result `shouldEqual` [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
