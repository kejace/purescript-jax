-- | Shape-typed deferred-allocation DSL — `Jax.Tensor`'s `T d`,
-- | re-kinded to carry a `Shape`.
-- |
-- | The DSL exists to chain ops without manual `ref`/`dispose`
-- | bookkeeping. A `T s` is an `Effect` that allocates a fresh
-- | `Tensor s` when run. `lit` borrows from a long-lived input;
-- | each use ref-bumps the underlying NDArray so the borrowed
-- | tensor stays alive past the expression. Combinators sequence
-- | through `Effect`, consuming each intermediate as they go.
-- |
-- | Operator vocabulary matches `Jax.Tensor`:
-- |
-- |     +.   elementwise addition          (infixl 6, broadcasts)
-- |     -.   elementwise subtraction       (infixl 6, broadcasts)
-- |     *.   elementwise multiplication    (infixl 7, broadcasts)
-- |     **.  matmul                        (infixl 7, rank-2 only)
-- |
-- | Use `lit` to inject an existing `Tensor s` and `run` to materialize
-- | the final result.
module Jax.Shape.Tensor.Expr
  ( T
  , run
  , lit
    -- * Binary
  , addT
  , subT
  , mulT
  , matmulT
  , (+.)
  , (-.)
  , (*.)
  , (**.)
    -- * Scalar
  , addScalarT
  , mulScalarT
    -- * Unary math
  , transposeT
  , sigmoidT
  , siluT
  , sqrtT
  , squareT
  , rsqrtT
  , sinT
  , tanhT
    -- * Reductions (rank-preserving)
  , meanAxisKeepT
  , sumAxisKeepT
    -- * Activations
  , softmaxT
    -- * Shape
  , reshapeT
  , sliceLastAxisT
  ) where

import Prelude

import Data.Reflectable (class Reflectable)
import Effect (Effect)
import Effect.Class (class MonadEffect, liftEffect)
import Type.Proxy (Proxy)

import Jax.Core as Core
import Jax.Shape (Shape, Lit, S1, S2, class Append, class Init, class Product)
import Jax.Shape.Broadcast (class Broadcast)
import Jax.Shape.Proxy (SProxy, class ReflectShape)
import Jax.Shape.Tensor (Tensor, unsafeAssumeShape, unsafeForgetShape)
import Jax.Shape.Tensor.Op as Op

-- =============================================================================
-- The T type
-- =============================================================================

-- | A deferred allocation of a `Tensor s`. Runtime is a thunk:
-- | running it allocates and returns a fresh shape-typed handle.
newtype T :: Shape -> Type
newtype T s = T (Effect (Tensor s))

-- | Realize a `T s` into a `Tensor s`. Polymorphic in `MonadEffect`
-- | so callers in `Aff` / `ReaderT _ Effect` / etc. don't need to
-- | wrap each `run` in `liftEffect`.
run :: forall s m. MonadEffect m => T s -> m (Tensor s)
run (T eff) = liftEffect eff

-- | Borrow an existing `Tensor s` into the DSL. Each use of the
-- | resulting `T s` ref-bumps the underlying NDArray, so the input
-- | stays alive after the expression has been run.
lit :: forall s. Tensor s -> T s
lit t = T do
  bumped <- Core.ref (unsafeForgetShape t :: Core.NDArray Core.D1)
  pure (unsafeAssumeShape bumped)

-- =============================================================================
-- Binary ops
-- =============================================================================

addT :: forall a b c. Broadcast a b c => T a -> T b -> T c
addT (T ea) (T eb) = T do
  a <- ea
  b <- eb
  Op.add a b

subT :: forall a b c. Broadcast a b c => T a -> T b -> T c
subT (T ea) (T eb) = T do
  a <- ea
  b <- eb
  Op.sub a b

mulT :: forall a b c. Broadcast a b c => T a -> T b -> T c
mulT (T ea) (T eb) = T do
  a <- ea
  b <- eb
  Op.mul a b

matmulT
  :: forall m k n
   . T (S2 m k)
  -> T (S2 k n)
  -> T (S2 m n)
matmulT (T ea) (T eb) = T do
  a <- ea
  b <- eb
  Op.matmul a b

infixl 6 addT as +.
infixl 6 subT as -.
infixl 7 mulT as *.
infixl 7 matmulT as **.

-- =============================================================================
-- Scalar ops
-- =============================================================================

addScalarT :: forall s. T s -> Number -> T s
addScalarT (T ea) n = T do
  a <- ea
  Op.addScalar a n

mulScalarT :: forall s. T s -> Number -> T s
mulScalarT (T ea) n = T do
  a <- ea
  Op.mulScalar a n

-- =============================================================================
-- Unary math
-- =============================================================================

transposeT :: forall m n. T (S2 m n) -> T (S2 n m)
transposeT (T ea) = T do
  a <- ea
  Op.transpose a

sigmoidT :: forall s. T s -> T s
sigmoidT (T ea) = T do
  a <- ea
  Op.sigmoid a

siluT :: forall s. T s -> T s
siluT (T ea) = T do
  a <- ea
  Op.silu a

sqrtT :: forall s. T s -> T s
sqrtT (T ea) = T do
  a <- ea
  Op.sqrt a

squareT :: forall s. T s -> T s
squareT (T ea) = T do
  a <- ea
  Op.square a

rsqrtT :: forall s. T s -> T s
rsqrtT (T ea) = T do
  a <- ea
  Op.rsqrt a

sinT :: forall s. T s -> T s
sinT (T ea) = T do
  a <- ea
  Op.sin a

tanhT :: forall s. T s -> T s
tanhT (T ea) = T do
  a <- ea
  Op.tanh a

-- =============================================================================
-- Reductions
-- =============================================================================

meanAxisKeepT :: forall s s'. T s -> Int -> T s'
meanAxisKeepT (T ea) axis = T do
  a <- ea
  Op.meanAxisKeep a axis

sumAxisKeepT :: forall s s'. T s -> Int -> T s'
sumAxisKeepT (T ea) axis = T do
  a <- ea
  Op.sumAxisKeep a axis

-- =============================================================================
-- Activations
-- =============================================================================

softmaxT :: forall s. T s -> Int -> T s
softmaxT (T ea) axis = T do
  a <- ea
  Op.softmax a axis

-- =============================================================================
-- Shape ops
-- =============================================================================

reshapeT
  :: forall s s' n
   . Product s n
  => Product s' n
  => ReflectShape s'
  => SProxy s'
  -> T s
  -> T s'
reshapeT pNew (T ea) = T do
  a <- ea
  Op.reshape pNew a

sliceLastAxisT
  :: forall s s' k inner
   . Init s inner
  => Append inner (S1 (Lit k)) s'
  => Reflectable k Int
  => Proxy k
  -> Int
  -> T s
  -> T s'
sliceLastAxisT pK start (T ea) = T do
  a <- ea
  Op.sliceLastAxis pK start a
