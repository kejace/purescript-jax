-- | A small expression DSL over `Jax.Core` ops that hides the
-- | refcount / dispose / `ref`-bump dance from the call site.
-- |
-- | The trick: a `T d` is a *deferred* `Effect (NDArray d)` —
-- | re-running the same `T d` allocates a fresh ref-bumped copy of the
-- | borrowed source tensor. So a value like
-- |
-- |     let xT = lit x
-- |     in xT *. xT
-- |
-- | runs `ref x` twice, producing two refcount-bumped handles, both of
-- | which `Core.mul` consumes — leaving `x` itself untouched.
-- |
-- | The user owns the input tensors and the final output (returned by
-- | `run`); intermediate fresh tensors are owned and consumed by the
-- | next combinator in the chain.
-- |
-- | Operator vocabulary (matches standard math precedence):
-- |
-- |   +.   elementwise addition          (infixl 6)
-- |   -.   elementwise subtraction       (infixl 6)
-- |   *.   elementwise multiplication    (infixl 7)
-- |   **.  matmul                        (infixl 7)
-- |
-- | We don't use `Semiring`/`Ring` instances on `T d` because shape-less
-- | `zero`/`one` don't make sense for tensors — the `ad-hoc` operators
-- | give the same readability win without the awkward instance laws.
module Jax.Tensor
  ( T
  , run
  , lit
  -- * Binary ops
  , addT
  , mulT
  , subT
  , matmulT
  , (+.)
  , (-.)
  , (*.)
  , (**.)
  -- * Unary ops
  , rsqrtT
  , squareT
  , sqrtT
  , sigmoidT
  , siluT
  , transposeT
  -- * Reductions (rank-preserving keepdims)
  , meanAxisKeepT
  , sumAxisKeepT
  -- * Scalar ops
  , addScalarT
  , mulScalarT
  -- * Shape ops
  , reshapeT
  , sliceLastAxisT
  , concatAxisT
  -- * Activations
  , softmaxT
  ) where

import Prelude

import Data.Array (snoc, uncons)
import Data.Maybe (Maybe(..))
import Effect (Effect)
import Jax.Core (NDArray)
import Jax.Core as Core

-- | A deferred allocation of an `NDArray d`. `lit` borrows a tensor by
-- | wrapping `ref` so each use is a fresh ref-bump; combinators
-- | sequence through `Effect`. `run` realizes a `T d` into the final
-- | `NDArray d`.
newtype T :: Type -> Type
newtype T d = T (Effect (NDArray d))

-- | Realize a deferred tensor expression into an `NDArray d`. The
-- | resulting tensor has refcount 1; the caller owns it (typically via
-- | `Jax.Managed.allocate`).
run :: forall d. T d -> Effect (NDArray d)
run (T eff) = eff

-- | Borrow an existing tensor. Each use of this `T d` ref-bumps the
-- | underlying handle, so the borrowed tensor stays alive for callers
-- | that hold it after this expression has been `run`.
lit :: forall d. NDArray d -> T d
lit x = T (Core.ref x)

-- =============================================================================
-- Binary
-- =============================================================================

addT :: forall d e f. T d -> T e -> T f
addT (T ax) (T bx) = T do
  a <- ax
  b <- bx
  Core.add a b

mulT :: forall d e f. T d -> T e -> T f
mulT (T ax) (T bx) = T do
  a <- ax
  b <- bx
  Core.mul a b

subT :: forall d e f. T d -> T e -> T f
subT (T ax) (T bx) = T do
  a <- ax
  b <- bx
  Core.sub a b

matmulT :: forall d. T d -> T d -> T d
matmulT (T ax) (T bx) = T do
  a <- ax
  b <- bx
  Core.matmul a b

infixl 6 addT as +.
infixl 6 subT as -.
infixl 7 mulT as *.
infixl 7 matmulT as **.

-- =============================================================================
-- Unary
-- =============================================================================

rsqrtT :: forall d. T d -> T d
rsqrtT (T ax) = T (ax >>= Core.rsqrt)

squareT :: forall d. T d -> T d
squareT (T ax) = T (ax >>= Core.square)

sqrtT :: forall d. T d -> T d
sqrtT (T ax) = T (ax >>= Core.sqrt)

sigmoidT :: forall d. T d -> T d
sigmoidT (T ax) = T (ax >>= Core.sigmoid)

siluT :: forall d. T d -> T d
siluT (T ax) = T (ax >>= Core.silu)

transposeT :: forall d. T d -> T d
transposeT (T ax) = T (ax >>= Core.transpose)

-- =============================================================================
-- Reductions
-- =============================================================================

meanAxisKeepT :: forall d. Int -> T d -> T d
meanAxisKeepT axis (T ax) = T do
  a <- ax
  Core.meanAxisKeep a axis

sumAxisKeepT :: forall d. Int -> T d -> T d
sumAxisKeepT axis (T ax) = T do
  a <- ax
  Core.sumAxisKeep a axis

-- =============================================================================
-- Scalar
-- =============================================================================

addScalarT :: forall d. T d -> Number -> T d
addScalarT (T ax) n = T do
  a <- ax
  Core.addScalar a n

mulScalarT :: forall d. T d -> Number -> T d
mulScalarT (T ax) n = T do
  a <- ax
  Core.mulScalar a n

-- =============================================================================
-- Shape
-- =============================================================================

reshapeT :: forall d e. T d -> Array Int -> T e
reshapeT (T ax) shape = T do
  a <- ax
  Core.reshape a shape

sliceLastAxisT :: forall d. T d -> Int -> Int -> T d
sliceLastAxisT (T ax) start end = T do
  a <- ax
  Core.sliceLastAxis a start end

concatAxisT :: forall d. Array (T d) -> Int -> T d
concatAxisT ts axis = T do
  arrs <- traverseRun ts
  Core.concatAxis arrs axis

-- Local helper: realize an array of `T d` actions in sequence. Avoids
-- pulling Data.Traversable for one use.
traverseRun :: forall d. Array (T d) -> Effect (Array (NDArray d))
traverseRun = go []
  where
  go acc xs = case uncons xs of
    Just { head: T eff, tail } -> do
      a <- eff
      go (snoc acc a) tail
    Nothing -> pure acc

-- =============================================================================
-- Activations
-- =============================================================================

softmaxT :: forall d. T d -> Int -> T d
softmaxT (T ax) axis = T do
  a <- ax
  Core.softmax a axis
