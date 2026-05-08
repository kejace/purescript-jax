-- | Shape-typed tensor wrapper around the rank-only `NDArray`.
-- |
-- | `Tensor (s :: Shape)` is the primary user-facing type for shape-
-- | typed code. `NDArray d` (rank-only) is a Core/FFI concern — most
-- | NN code shouldn't see it directly.
-- |
-- | Stage 1 provides the *type* and the FFI escape; the operations
-- | (matmul, reshape, broadcast, …) get re-typed in Stage 2 alongside
-- | the existing `Jax.Tensor` DSL.
-- |
-- | Implementation detail: `Tensor` is a `newtype` over an NDArray
-- | tagged with a fresh phantom rank `RankSlot`. The shape `s` is the
-- | source of truth; the rank witness is derived via `RankOf s d`
-- | (from `Jax.Shape`) on demand at op boundaries. We use a fresh
-- | phantom rather than re-using D1..D4 directly so the rank-witness
-- | derivation stays type-class-driven (no overlapping branches in
-- | `RankWitness`).
module Jax.Shape.Tensor
  ( Tensor
  , RankSlot
  , unsafeAssumeShape
  , unsafeForgetShape
  , withRank
  ) where

import Unsafe.Coerce (unsafeCoerce)

import Jax.Core (NDArray)
import Jax.Shape (Shape, class RankOf)

-- =============================================================================
-- The wrapper type
-- =============================================================================

-- | Internal phantom for the inner NDArray's rank parameter. The
-- | rank-witness type that matters is derived from the shape `s`
-- | via `RankOf`; this slot is never inspected.
foreign import data RankSlot :: Type

-- | Shape-typed tensor. The shape `s` is the source of truth — `Tensor
-- | (S2 (Lit 768) (Lit 768))` is provably a 768×768 matrix.
-- |
-- | Constructors are not exported: callers obtain a `Tensor` either
-- | from a typed op (most code paths) or via `unsafeAssumeShape` at
-- | an FFI boundary (loaders, allocator wrappers).
newtype Tensor :: Shape -> Type
newtype Tensor s = Tensor (NDArray RankSlot)

-- =============================================================================
-- FFI escape hatches
-- =============================================================================
--
-- These are the *only* ways to materialize a `Tensor s` from an
-- untyped (or differently-typed) `NDArray`. Their presence is
-- intentionally inconvenient: callers must spell out the assumption
-- so a code reviewer (or fp-police) can see it. The `unsafe` prefix
-- makes both the cost and the discipline visible.
--
-- Project rule (enforced in `.claude/fp-police-rules.md`): these may
-- only appear in `src/Jax/Loaders/**` and `src/Jax/Core.purs`.
-- Anywhere else, prefer building `Tensor`s through typed ops.

-- | Assert that an `NDArray` has the given shape. Caller-checked, not
-- | runtime-checked. Use at FFI boundaries (safetensors loader,
-- | tokenizer output) where the shape comes from external metadata.
-- |
-- | Underneath this is a phantom-only coercion: `NDArray d` and
-- | `Tensor s` share a runtime representation (an opaque jax-js
-- | handle). The `unsafeCoerce` is sound at the JS level; what's
-- | unsafe is the *typing claim* that `s` describes the data.
unsafeAssumeShape :: forall s d. NDArray d -> Tensor s
unsafeAssumeShape = unsafeCoerce

-- | Drop the shape, recovering the rank-only `NDArray`. The rank
-- | parameter is whatever the caller demands; usually fixed by
-- | context (e.g. a `forall d. RankOf s d => ⋯` constraint upstream).
-- |
-- | Symmetric to `unsafeAssumeShape`; same caveats.
unsafeForgetShape :: forall s d. Tensor s -> NDArray d
unsafeForgetShape = unsafeCoerce

-- | A safe variant of `unsafeForgetShape` for callers who already
-- | have a `RankOf s d` constraint in scope: at op boundaries the
-- | rank can be derived from the shape, so the cast is constrained
-- | to the exact rank the FFI op expects. The runtime is identical
-- | (still phantom-only); the type system rules out the wrong rank.
withRank :: forall s d. RankOf s d => Tensor s -> NDArray d
withRank = unsafeCoerce
