-- | Shape-typed primitive operations on `Tensor s`.
-- |
-- | Each op wraps the corresponding rank-only `Jax.Core` op with a
-- | shape-correct signature. The wrapping costs nothing at runtime —
-- | `withRank` and `unsafeAssumeShape` are phantom-only coercions —
-- | so the type system gets richer information without performance
-- | regression vs. raw `Jax.Core` calls.
-- |
-- | Stage 2 surface: matmul, transpose, reshape (Product-equality),
-- | element-wise binary ops (broadcast-typed), scalar ops, unaries,
-- | sliceLastAxis (Lit-witnessed new size). These cover the
-- | hotspots in `Jax.NN.Block` without committing to fully-typed
-- | reduce / slice / oneHot — those land in Stage 4 when their
-- | constraint design has settled.
module Jax.Shape.Tensor.Op
  ( -- * Allocators (concrete shape via type ascription)
    zeros
  , ones
    -- * Binary
  , matmul
  , add
  , sub
  , mul
    -- * Scalar
  , addScalar
  , mulScalar
    -- * Unary math (shape-preserving)
  , transpose
  , sigmoid
  , silu
  , sqrt
  , square
  , rsqrt
  , sin
  , tanh
    -- * Reductions (rank-preserving, keepdims)
  , meanAxisKeep
  , sumAxisKeep
    -- * Activations
  , softmax
    -- * Shape ops
  , reshape
  , reshapeUnchecked
  , sliceLastAxis
  ) where

import Prelude hiding (add, mul, sub)

import Data.Reflectable (class Reflectable, reflectType)
import Effect (Effect)
import Type.Proxy (Proxy)

import Jax.Core (NDArray)
import Jax.Core as Core
import Jax.Shape (Lit, S1, S2, class Append, class Init, class Product)
import Jax.Shape.Broadcast (class Broadcast)
import Jax.Shape.Proxy (SProxy, class ReflectShape, reflectShape)
import Jax.Shape.Tensor (Tensor, unsafeAssumeShape, unsafeForgetShape, withRank)

-- =============================================================================
-- Allocators
-- =============================================================================
--
-- Allocators don't take a shape value at the term level — they take a
-- `SProxy s` so the type and the runtime size array agree. The `s` must
-- be `Lit`-only (no `Var` dims) since we need to materialize the size
-- array; that's enforced by `ReflectShape`.

-- | Allocate a fresh zero tensor of the given shape. Shape is taken as
-- | a `SProxy s`; the runtime size array comes from `reflectShape`.
zeros :: forall s. ReflectShape s => SProxy s -> Effect (Tensor s)
zeros pShape = do
  result <- Core.zeros (reflectShape pShape)
  pure (unsafeAssumeShape (result :: NDArray Core.D1))
  -- The :: NDArray D1 is a dummy; the runtime is fine because the
  -- phantom is erased. Same trick for ones below.

-- | Allocate a fresh tensor of all ones.
ones :: forall s. ReflectShape s => SProxy s -> Effect (Tensor s)
ones pShape = do
  result <- Core.ones (reflectShape pShape)
  pure (unsafeAssumeShape (result :: NDArray Core.D1))

-- =============================================================================
-- Binary ops
-- =============================================================================

-- | Matrix multiplication on rank-2 tensors. Inner-dim agreement is
-- | type-checked: `[m, k] @ [k, n] = [m, n]`.
matmul
  :: forall m k n
   . Tensor (S2 m k)
  -> Tensor (S2 k n)
  -> Effect (Tensor (S2 m n))
matmul a b = do
  result <- Core.matmul (withRank a :: NDArray Core.D2) (withRank b :: NDArray Core.D2)
  pure (unsafeAssumeShape result)

-- | Elementwise addition with NumPy-style broadcasting. The result
-- | shape is determined by `Broadcast a b c`.
add
  :: forall a b c
   . Broadcast a b c
  => Tensor a
  -> Tensor b
  -> Effect (Tensor c)
add x y = do
  result <- Core.add (unsafeForgetShape x :: NDArray Core.D1)
                     (unsafeForgetShape y :: NDArray Core.D1)
  pure (unsafeAssumeShape result)
  -- The D1 annotation is a placeholder — at runtime Core.add is rank-
  -- agnostic, so the phantom doesn't matter. The shape c carries the
  -- real type info.

-- | Elementwise subtraction with broadcast.
sub
  :: forall a b c
   . Broadcast a b c
  => Tensor a
  -> Tensor b
  -> Effect (Tensor c)
sub x y = do
  result <- Core.sub (unsafeForgetShape x :: NDArray Core.D1)
                     (unsafeForgetShape y :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

-- | Elementwise multiplication with broadcast.
mul
  :: forall a b c
   . Broadcast a b c
  => Tensor a
  -> Tensor b
  -> Effect (Tensor c)
mul x y = do
  result <- Core.mul (unsafeForgetShape x :: NDArray Core.D1)
                     (unsafeForgetShape y :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

-- =============================================================================
-- Scalar ops
-- =============================================================================

-- | Add a scalar to every element. Shape preserved.
addScalar :: forall s. Tensor s -> Number -> Effect (Tensor s)
addScalar t n = do
  result <- Core.addScalar (unsafeForgetShape t :: NDArray Core.D1) n
  pure (unsafeAssumeShape result)

-- | Multiply every element by a scalar. Shape preserved.
mulScalar :: forall s. Tensor s -> Number -> Effect (Tensor s)
mulScalar t n = do
  result <- Core.mulScalar (unsafeForgetShape t :: NDArray Core.D1) n
  pure (unsafeAssumeShape result)

-- =============================================================================
-- Unary math (shape-preserving)
-- =============================================================================

-- | 2-D matrix transpose. `[m, n] -> [n, m]`. Note: this only changes
-- | axes for rank-2; for higher-rank transposes use `Core.transpose`
-- | (which reverses all axes) via a Stage-4 typed wrapper.
transpose :: forall m n. Tensor (S2 m n) -> Effect (Tensor (S2 n m))
transpose t = do
  result <- Core.transpose (withRank t :: NDArray Core.D2)
  pure (unsafeAssumeShape result)

sigmoid :: forall s. Tensor s -> Effect (Tensor s)
sigmoid t = do
  result <- Core.sigmoid (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

silu :: forall s. Tensor s -> Effect (Tensor s)
silu t = do
  result <- Core.silu (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

sqrt :: forall s. Tensor s -> Effect (Tensor s)
sqrt t = do
  result <- Core.sqrt (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

square :: forall s. Tensor s -> Effect (Tensor s)
square t = do
  result <- Core.square (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

rsqrt :: forall s. Tensor s -> Effect (Tensor s)
rsqrt t = do
  result <- Core.rsqrt (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

sin :: forall s. Tensor s -> Effect (Tensor s)
sin t = do
  result <- Core.sin (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

tanh :: forall s. Tensor s -> Effect (Tensor s)
tanh t = do
  result <- Core.tanh (unsafeForgetShape t :: NDArray Core.D1)
  pure (unsafeAssumeShape result)

-- =============================================================================
-- Reductions (rank-preserving, keepdims)
-- =============================================================================
--
-- These use jax-js's keepdims=true so the reduced axis collapses to
-- size 1 rather than disappearing. Shape transform: replacing axis n
-- with `Lit 1`. We require a `Replace` proof to type the result.

-- | Mean over a single axis with keepdims=true. The axis index is a
-- | runtime `Int`; the output shape is the same as the input shape with
-- | the chosen axis collapsed to size 1.
-- |
-- | This signature is Stage-4-honest about shape change but skips the
-- | type-level axis index (we'd need `Replace` here). For Stage 2,
-- | callers pass the axis runtime and the output shape via type
-- | ascription.
meanAxisKeep :: forall s s'. Tensor s -> Int -> Effect (Tensor s')
meanAxisKeep t axis = do
  result <- Core.meanAxisKeep (unsafeForgetShape t :: NDArray Core.D1) axis
  pure (unsafeAssumeShape result)

-- | Sum over a single axis with keepdims=true. See `meanAxisKeep`.
sumAxisKeep :: forall s s'. Tensor s -> Int -> Effect (Tensor s')
sumAxisKeep t axis = do
  result <- Core.sumAxisKeep (unsafeForgetShape t :: NDArray Core.D1) axis
  pure (unsafeAssumeShape result)

-- =============================================================================
-- Activations
-- =============================================================================

-- | Softmax along an axis. Shape preserved. The axis is a runtime Int
-- | for now — typed-axis variants land in Stage 4.
softmax :: forall s. Tensor s -> Int -> Effect (Tensor s)
softmax t axis = do
  result <- Core.softmax (unsafeForgetShape t :: NDArray Core.D1) axis
  pure (unsafeAssumeShape result)

-- =============================================================================
-- Shape ops
-- =============================================================================

-- | Reshape with type-level total-element-count preservation.
-- | `Product s n` and `Product s' n` together require both shapes to
-- | have the same product. Both shapes must be `Lit`-only (`Var`-
-- | containing shapes don't have known products).
-- |
-- | For the common NN reshape `[seq, hidden] → [seq, nHeads, headDim]`
-- | where `seq` is a `Var`, this strict version doesn't apply — use
-- | `reshapeUnchecked` instead, with the runtime size array.
reshape
  :: forall s s' n
   . Product s n
  => Product s' n
  => ReflectShape s'
  => SProxy s'
  -> Tensor s
  -> Effect (Tensor s')
reshape pNew t = do
  let dims = reflectShape pNew
  result <- Core.reshape (unsafeForgetShape t :: NDArray Core.D1) dims
  pure (unsafeAssumeShape result)

-- | Reshape with caller-asserted result shape. The runtime `Array Int`
-- | must match `s'` element-for-element — the type system can't verify
-- | this for shapes containing `Var` dims (most NN code, where seq /
-- | batch / vocab come from runtime config).
-- |
-- | Use this where `reshape` doesn't apply; reach for `reshape` (which
-- | proves `Product s ~ Product s'`) whenever both shapes are
-- | fully-`Lit`. Documented escape, not an unsafeCoerce — the
-- | unsoundness is bounded to the user's claim that `dims` matches
-- | `s'`. If they get the type ascription wrong, downstream typed ops
-- | will likely fail to compile rather than producing garbage.
reshapeUnchecked
  :: forall s s'
   . Array Int
  -> Tensor s
  -> Effect (Tensor s')
reshapeUnchecked dims t = do
  result <- Core.reshape (unsafeForgetShape t :: NDArray Core.D1) dims
  pure (unsafeAssumeShape result)

-- | Slice along the last axis to yield a new known-size last axis.
-- | The new size `k` is given as a `Lit k` proxy; the rest of the
-- | shape is preserved via `Init` + `Append`.
-- |
-- | Useful for RoPE's `[..., dim] → [..., halfDim]` split.
sliceLastAxis
  :: forall s s' k inner
   . Init s inner
  => Append inner (S1 (Lit k)) s'
  => Reflectable k Int
  => Proxy k       -- ^ new last-axis size (statically known)
  -> Int           -- ^ start offset along the last axis
  -> Tensor s
  -> Effect (Tensor s')
sliceLastAxis pK start t = do
  let k = reflectType pK
  result <- Core.sliceLastAxis (unsafeForgetShape t :: NDArray Core.D1) start k
  pure (unsafeAssumeShape result)
