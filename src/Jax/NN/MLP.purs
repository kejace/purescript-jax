module Jax.NN.MLP (mlp) where

import Prelude

import Effect (Effect)
import Jax.Core (D2, NDArray)
import Jax.Tensor (lit, matmulT, mulT, run, siluT)

-- | SwiGLU feed-forward block (the Llama-style MLP).
-- |
-- |   `gate = x · gate_proj`
-- |   `up   = x · up_proj`
-- |   `out  = (silu(gate) · up) · down_proj`
-- |
-- | Shapes (unbatched [seq, hidden]):
-- |   x          [seq, hidden]
-- |   gate_proj  [hidden, intermediate]
-- |   up_proj    [hidden, intermediate]
-- |   down_proj  [intermediate, hidden]
-- |   out        [seq, hidden]
-- |
-- | All inputs are borrowed. Implementation uses the `Jax.Tensor` DSL
-- | so refcount discipline stays inside the wrappers.
mlp
  :: NDArray D2  -- ^ x
  -> NDArray D2  -- ^ gate_proj
  -> NDArray D2  -- ^ up_proj
  -> NDArray D2  -- ^ down_proj
  -> Effect (NDArray D2)
mlp x gateProj upProj downProj = run (matmulT inner (lit downProj))
  where
  xT = lit x
  gate = matmulT xT (lit gateProj)
  up = matmulT xT (lit upProj)
  inner = mulT (siluT gate) up
