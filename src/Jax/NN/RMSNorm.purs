module Jax.NN.RMSNorm (rmsnorm) where

import Prelude

import Effect (Effect)
import Jax.Core (D1, NDArray)
import Jax.Tensor (addScalarT, lit, meanAxisKeepT, rsqrtT, run, squareT, (*.))

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
rmsnorm eps x weight = run (xT *. invRms *. wT)
  where
  xT = lit x
  wT = lit weight
  invRms = rsqrtT (addScalarT (meanAxisKeepT (-1) (squareT xT)) eps)
