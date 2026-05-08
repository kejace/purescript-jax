module Jax.Managed
  ( Managed
  , runManaged
  , allocate
  , allocateT
  , managed
  ) where

import Prelude

import Control.Monad.Cont.Trans (ContT(..), runContT)
import Data.Either (Either(..))
import Effect (Effect)
import Effect.Exception (throwException, try)
import Jax.Core (NDArray, dispose)
import Jax.Shape.Tensor (Tensor, disposeT)

-- | Scope-based ownership for jax-js tensors via continuation-passing.
-- |
-- | Allocations made via `allocate` inside a `Managed` block are disposed
-- | after the continuation passed to `runManaged` returns (or throws —
-- | release runs on both paths).
-- |
-- | Contract: the borrowed tensor handle inside `allocate`'s body must
-- | not be passed directly into a jax-js op (which would consume it),
-- | because `Managed` then double-frees on scope exit. Bump the refcount
-- | with `Jax.Core.ref` first and pass the fresh reference to the op.
type Managed = ContT Unit Effect

-- | Run a `Managed` computation. The produced value is fed to the
-- | continuation; allocated resources are released before `runManaged`
-- | returns.
runManaged :: forall a. Managed a -> (a -> Effect Unit) -> Effect Unit
runManaged = runContT

-- | Allocate a tensor and bind its disposal to the enclosing scope.
allocate :: forall d. Effect (NDArray d) -> Managed (NDArray d)
allocate acquire = ContT \k -> do
  a <- acquire
  result <- try (k a)
  dispose a
  case result of
    Left err -> throwException err
    Right v -> pure v

-- | Shape-typed sibling of `allocate`. Same scoping semantics; the
-- | scoped value is a `Tensor s` instead of a rank-only `NDArray d`.
allocateT :: forall s. Effect (Tensor s) -> Managed (Tensor s)
allocateT acquire = ContT \k -> do
  a <- acquire
  result <- try (k a)
  disposeT a
  case result of
    Left err -> throwException err
    Right v -> pure v

-- | Generic acquire/release for non-tensor resources.
managed :: forall a. Effect a -> (a -> Effect Unit) -> Managed a
managed acquire release = ContT \k -> do
  r <- acquire
  result <- try (k r)
  release r
  case result of
    Left err -> throwException err
    Right v -> pure v
