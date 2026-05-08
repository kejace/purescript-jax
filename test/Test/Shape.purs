-- | Tests for `Jax.Shape` and friends.
-- |
-- | Two layers of test:
-- |
-- |   * **Compile-time** assertions — bindings whose type signatures
-- |     pin down a type-level equality. If the file compiles, the
-- |     equality holds; if it doesn't, the binding errors out at
-- |     `bun run build`. These are the workhorse for `RankOf`,
-- |     `Product`, `Replace`, `Append`, `Last`, `Head`, `Tail`,
-- |     `Init`, `Broadcast`. No runtime assertion needed.
-- |
-- |   * **Runtime** assertions — `reflectShape` reads the type-level
-- |     shape into an `Array Int`, then we compare to an expected
-- |     literal. Catches bugs in the reflection chain (the only piece
-- |     that's not pure compile-time).
module Test.Shape
  ( spec
  ) where

import Prelude

import Effect.Class (liftEffect)
import Test.Spec (Spec, describe, it)
import Test.Spec.Assertions (shouldEqual)
import Type.Proxy (Proxy(..))

import Jax.Core (D0, D1, D2, D3, D4)
import Jax.Shape
  ( SCons
  , SNil
  , Lit
  , Var
  , S2
  , S3
  , S4
  , class RankOf
  , class Product
  , class Append
  , class Last
  , class Init
  , class Head
  , class Tail
  , class Replace
  , class CountAxes
  )
import Jax.Shape.Broadcast (class Broadcast)
import Jax.Shape.Proxy (reflectShape)

-- =============================================================================
-- Compile-time assertions
-- =============================================================================
--
-- The pattern: a "checker" function whose constraint embodies the
-- equality we want; a binding that calls the checker on a Proxy of
-- the input. If the equality holds at the type level, the binding
-- typechecks. The body is `unit` — values are irrelevant.

-- | Assert that `RankOf s d` holds.
checkRank
  :: forall s d. RankOf s d => Proxy s -> Proxy d -> Unit
checkRank _ _ = unit

-- | Assert that `Product s n` holds.
checkProduct
  :: forall s n. Product s n => Proxy s -> Proxy n -> Unit
checkProduct _ _ = unit

-- | Assert that `CountAxes s n` holds.
checkCountAxes
  :: forall s n. CountAxes s n => Proxy s -> Proxy n -> Unit
checkCountAxes _ _ = unit

-- | Assert that `Append a b c` holds.
checkAppend
  :: forall a b c. Append a b c => Proxy a -> Proxy b -> Proxy c -> Unit
checkAppend _ _ _ = unit

-- | Assert that `Last s d` holds.
checkLast
  :: forall s d. Last s d => Proxy s -> Proxy d -> Unit
checkLast _ _ = unit

-- | Assert that `Init s s'` holds.
checkInit
  :: forall s s'. Init s s' => Proxy s -> Proxy s' -> Unit
checkInit _ _ = unit

-- | Assert that `Head s d` holds.
checkHead
  :: forall s d. Head s d => Proxy s -> Proxy d -> Unit
checkHead _ _ = unit

-- | Assert that `Tail s s'` holds.
checkTail
  :: forall s s'. Tail s s' => Proxy s -> Proxy s' -> Unit
checkTail _ _ = unit

-- | Assert that `Replace n s d s'` holds.
checkReplace
  :: forall n s d s'. Replace n s d s' => Proxy n -> Proxy s -> Proxy d -> Proxy s' -> Unit
checkReplace _ _ _ _ = unit

-- | Assert that `Broadcast a b c` holds.
checkBroadcast
  :: forall a b c. Broadcast a b c => Proxy a -> Proxy b -> Proxy c -> Unit
checkBroadcast _ _ _ = unit

-- -----------------------------------------------------------------------------
-- The actual assertions (each one is a typecheck-or-fail)
-- -----------------------------------------------------------------------------

-- CountAxes / RankOf
_a_count_nil :: Unit
_a_count_nil = checkCountAxes (Proxy :: Proxy SNil) (Proxy :: Proxy 0)

_a_count_2 :: Unit
_a_count_2 = checkCountAxes
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4))) (Proxy :: Proxy 2)

_a_count_4 :: Unit
_a_count_4 = checkCountAxes
  (Proxy :: Proxy (S4 (Lit 1) (Lit 2) (Lit 3) (Lit 4))) (Proxy :: Proxy 4)

_a_rank_scalar :: Unit
_a_rank_scalar = checkRank (Proxy :: Proxy SNil) (Proxy :: Proxy D0)

_a_rank_1 :: Unit
_a_rank_1 = checkRank
  (Proxy :: Proxy (SCons (Lit 5) SNil)) (Proxy :: Proxy D1)

_a_rank_2 :: Unit
_a_rank_2 = checkRank
  (Proxy :: Proxy (S2 (Lit 768) (Lit 768))) (Proxy :: Proxy D2)

_a_rank_3 :: Unit
_a_rank_3 = checkRank
  (Proxy :: Proxy (S3 (Var "seq") (Lit 12) (Lit 64))) (Proxy :: Proxy D3)

_a_rank_4 :: Unit
_a_rank_4 = checkRank
  (Proxy :: Proxy (S4 (Lit 1) (Lit 2) (Lit 3) (Lit 4))) (Proxy :: Proxy D4)

-- Product
_a_product_nil :: Unit
_a_product_nil = checkProduct (Proxy :: Proxy SNil) (Proxy :: Proxy 1)

_a_product_singleton :: Unit
_a_product_singleton = checkProduct
  (Proxy :: Proxy (SCons (Lit 7) SNil)) (Proxy :: Proxy 7)

_a_product_2x3 :: Unit
_a_product_2x3 = checkProduct
  (Proxy :: Proxy (S2 (Lit 2) (Lit 3))) (Proxy :: Proxy 6)

_a_product_768_768 :: Unit
_a_product_768_768 = checkProduct
  (Proxy :: Proxy (S2 (Lit 768) (Lit 768))) (Proxy :: Proxy 589824)

-- Append
_a_append_empty_left :: Unit
_a_append_empty_left = checkAppend
  (Proxy :: Proxy SNil)
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))

_a_append_empty_right :: Unit
_a_append_empty_right = checkAppend
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy SNil)
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))

_a_append_concat :: Unit
_a_append_concat = checkAppend
  (Proxy :: Proxy (S2 (Lit 1) (Lit 2)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S4 (Lit 1) (Lit 2) (Lit 3) (Lit 4)))

-- Last / Init / Head / Tail
_a_last_singleton :: Unit
_a_last_singleton = checkLast
  (Proxy :: Proxy (SCons (Lit 5) SNil)) (Proxy :: Proxy (Lit 5))

_a_last_S3 :: Unit
_a_last_S3 = checkLast
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3))) (Proxy :: Proxy (Lit 3))

_a_init_S3 :: Unit
_a_init_S3 = checkInit
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3)))
  (Proxy :: Proxy (S2 (Lit 1) (Lit 2)))

_a_head_S3 :: Unit
_a_head_S3 = checkHead
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3))) (Proxy :: Proxy (Lit 1))

_a_tail_S3 :: Unit
_a_tail_S3 = checkTail
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3)))
  (Proxy :: Proxy (S2 (Lit 2) (Lit 3)))

-- Replace
_a_replace_axis_0 :: Unit
_a_replace_axis_0 = checkReplace
  (Proxy :: Proxy 0)
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3)))
  (Proxy :: Proxy (Lit 99))
  (Proxy :: Proxy (S3 (Lit 99) (Lit 2) (Lit 3)))

_a_replace_axis_1 :: Unit
_a_replace_axis_1 = checkReplace
  (Proxy :: Proxy 1)
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3)))
  (Proxy :: Proxy (Lit 99))
  (Proxy :: Proxy (S3 (Lit 1) (Lit 99) (Lit 3)))

_a_replace_axis_2 :: Unit
_a_replace_axis_2 = checkReplace
  (Proxy :: Proxy 2)
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 3)))
  (Proxy :: Proxy (Lit 99))
  (Proxy :: Proxy (S3 (Lit 1) (Lit 2) (Lit 99)))

-- Broadcast
_a_broadcast_eq :: Unit
_a_broadcast_eq = checkBroadcast
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))

_a_broadcast_left_one :: Unit
_a_broadcast_left_one = checkBroadcast
  (Proxy :: Proxy (S2 (Lit 1) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))

_a_broadcast_right_one :: Unit
_a_broadcast_right_one = checkBroadcast
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 1)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))

_a_broadcast_pad_left :: Unit
_a_broadcast_pad_left = checkBroadcast
  (Proxy :: Proxy (SCons (Lit 4) SNil))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 3) (Lit 4)))

_a_broadcast_var_eq :: Unit
_a_broadcast_var_eq = checkBroadcast
  (Proxy :: Proxy (S2 (Var "seq") (Lit 4)))
  (Proxy :: Proxy (S2 (Var "seq") (Lit 4)))
  (Proxy :: Proxy (S2 (Var "seq") (Lit 4)))

_a_broadcast_var_with_one :: Unit
_a_broadcast_var_with_one = checkBroadcast
  (Proxy :: Proxy (S2 (Var "seq") (Lit 4)))
  (Proxy :: Proxy (S2 (Lit 1) (Lit 4)))
  (Proxy :: Proxy (S2 (Var "seq") (Lit 4)))

-- Touch all the bindings so they don't get DCE-warned.
allCompileTimeAssertionsTouched :: Unit
allCompileTimeAssertionsTouched =
  _a_count_nil <> _a_count_2 <> _a_count_4
    <> _a_rank_scalar <> _a_rank_1 <> _a_rank_2 <> _a_rank_3 <> _a_rank_4
    <> _a_product_nil <> _a_product_singleton <> _a_product_2x3
    <> _a_product_768_768
    <> _a_append_empty_left <> _a_append_empty_right <> _a_append_concat
    <> _a_last_singleton <> _a_last_S3
    <> _a_init_S3 <> _a_head_S3 <> _a_tail_S3
    <> _a_replace_axis_0 <> _a_replace_axis_1 <> _a_replace_axis_2
    <> _a_broadcast_eq <> _a_broadcast_left_one <> _a_broadcast_right_one
    <> _a_broadcast_pad_left
    <> _a_broadcast_var_eq <> _a_broadcast_var_with_one

-- =============================================================================
-- Runtime assertions (reflection)
-- =============================================================================

spec :: Spec Unit
spec = describe "Type-level shapes" do
  describe "compile-time assertions" do
    it "all type-level identities hold" $ liftEffect $
      allCompileTimeAssertionsTouched `shouldEqual` unit
  describe "runtime reflection (Lit-only shapes)" do
    it "reflectShape SNil = []" $
      reflectShape (Proxy :: Proxy SNil) `shouldEqual` []
    it "reflectShape [7] = [7]" $
      reflectShape (Proxy :: Proxy (SCons (Lit 7) SNil)) `shouldEqual` [ 7 ]
    it "reflectShape [2, 3] = [2, 3]" $
      reflectShape (Proxy :: Proxy (S2 (Lit 2) (Lit 3))) `shouldEqual` [ 2, 3 ]
    it "reflectShape [768, 768] = [768, 768]" $
      reflectShape (Proxy :: Proxy (S2 (Lit 768) (Lit 768)))
        `shouldEqual` [ 768, 768 ]
    it "reflectShape [12, 64, 32] = [12, 64, 32]" $
      reflectShape (Proxy :: Proxy (S3 (Lit 12) (Lit 64) (Lit 32)))
        `shouldEqual` [ 12, 64, 32 ]
