module Jax.NN.RMSNorm (rmsnorm) where

import Prelude hiding (add, mul)

import Effect (Effect)
import Jax.Core (D1, NDArray, addScalar, meanAxisKeep, mul, ref, rsqrt, square)

-- | Root mean square normalization.
-- |
-- | Formula: `y = x * rsqrt(mean(x², axis=-1, keepdims=true) + eps) * weight`
-- |
-- | Inputs `x` and `weight` are *borrowed* — their refcounts are unchanged
-- | on return. The output is a fresh tensor (refcount 1) the caller must
-- | dispose, typically via `Jax.Managed.allocate`.
rmsnorm
  :: forall d
   . Number
  -> NDArray d
  -> NDArray D1
  -> Effect (NDArray d)
rmsnorm eps x weight = do
  -- x²
  xR1 <- ref x
  xSq <- square xR1
  -- mean(x², axis=-1, keepdims=true) → rank-preserving with last axis = 1
  m <- meanAxisKeep xSq (-1)
  -- + eps (scalar broadcast)
  mEps <- addScalar m eps
  -- rsqrt
  invRms <- rsqrt mEps
  -- x * invRms (broadcasts along the last axis)
  xR2 <- ref x
  scaled <- mul xR2 invRms
  -- * weight (broadcasts weight across leading axes of scaled)
  weightR <- ref weight
  mul scaled weightR
