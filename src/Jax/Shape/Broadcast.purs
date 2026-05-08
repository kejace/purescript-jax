-- | NumPy-style broadcasting at the type level.
-- |
-- |     Broadcast [3, 1, 4] [1, 5, 4] = [3, 5, 4]
-- |     Broadcast [3, 4]    [4]       = [3, 4]
-- |     Broadcast [3, 4]    [5, 4]    -- type error: 3 ≠ 5
-- |
-- | Algorithm (NumPy):
-- |   1. Right-align the two shape lists.
-- |   2. Pad the shorter one with leading 1s.
-- |   3. Pairwise: if dims are equal → that dim; if one is 1 → the
-- |      other; otherwise mismatch.
-- |
-- | Shape is a head-cons list, so we reverse, walk the (now left-
-- | aligned, formerly right-aligned) pair, then reverse the result.
-- |
-- | Symbolic dims (`Var s`) only broadcast in three cases: the same
-- | name on both sides, `Lit 1` against the variable, or vice versa.
-- | Anything else (two different Vars, a Var against a non-1 Lit) is
-- | rejected — we can't decide equality without runtime info, so we
-- | conservatively refuse rather than pretend.
module Jax.Shape.Broadcast
  ( class Broadcast
  , class BroadcastRev
  , class BroadcastDim
  , class Reverse
  , class ReverseAcc
  ) where

import Jax.Shape (Shape, SNil, SCons, Dim, Lit)

-- =============================================================================
-- Reverse a Shape (helper for right-aligned walks)
-- =============================================================================

-- | `Reverse s s'` ≡ "`s'` is `s` reversed".
class Reverse (s :: Shape) (s' :: Shape) | s -> s'

instance reverseShape ::
  ( ReverseAcc s SNil s'
  ) => Reverse s s'

-- | Worker class: accumulating reverse with explicit accumulator.
class ReverseAcc (s :: Shape) (acc :: Shape) (s' :: Shape) | s acc -> s'

instance reverseAccNil :: ReverseAcc SNil acc acc

instance reverseAccCons ::
  ( ReverseAcc rest (SCons d acc) s'
  ) => ReverseAcc (SCons d rest) acc s'

-- =============================================================================
-- Pairwise dim broadcasting
-- =============================================================================

-- | `BroadcastDim a b c` ≡ "broadcasting dim `a` against dim `b`
-- | yields dim `c`". Three cases (in instance-resolution order):
-- |
-- |   1. Identical dims (any same `Lit n` or same `Var s`) → that dim.
-- |   2. `Lit 1` on the left → the right operand.
-- |   3. `Lit 1` on the right → the left operand.
-- |
-- | Anything else is a type error — including two different Vars,
-- | which we conservatively refuse to unify.
class BroadcastDim (a :: Dim) (b :: Dim) (c :: Dim) | a b -> c

instance broadcastDimRefl :: BroadcastDim d d d

else instance broadcastDimLeftOne :: BroadcastDim (Lit 1) d d

else instance broadcastDimRightOne :: BroadcastDim d (Lit 1) d

-- =============================================================================
-- Right-aligned shape broadcasting (operates on reversed shapes)
-- =============================================================================

-- | `BroadcastRev a b c` ≡ "broadcasting two *reversed* shapes
-- | left-to-right (= NumPy's right-aligned walk) yields a *reversed*
-- | result `c`". The top-level `Broadcast` reverses inputs going in
-- | and outputs going out.
class BroadcastRev (a :: Shape) (b :: Shape) (c :: Shape) | a b -> c

instance broadcastRevNilNil :: BroadcastRev SNil SNil SNil

else instance broadcastRevNilCons ::
  ( BroadcastRev SNil rest acc
  ) => BroadcastRev SNil (SCons d rest) (SCons d acc)

else instance broadcastRevConsNil ::
  ( BroadcastRev rest SNil acc
  ) => BroadcastRev (SCons d rest) SNil (SCons d acc)

else instance broadcastRevCons ::
  ( BroadcastDim da db dc
  , BroadcastRev restA restB restC
  ) => BroadcastRev (SCons da restA) (SCons db restB) (SCons dc restC)

-- =============================================================================
-- The user-facing class
-- =============================================================================

-- | `Broadcast a b c` ≡ "broadcasting shapes `a` and `b` (NumPy
-- | semantics) yields shape `c`". Used to type element-wise binary
-- | ops: `(.+.) :: Broadcast a b c => Tensor a -> Tensor b -> Tensor c`.
class Broadcast (a :: Shape) (b :: Shape) (c :: Shape) | a b -> c

instance broadcastShapes ::
  ( Reverse a aRev
  , Reverse b bRev
  , BroadcastRev aRev bRev cRev
  , Reverse cRev c
  ) => Broadcast a b c
