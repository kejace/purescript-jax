module Jax.NN.MLP (mlp) where

import Prelude hiding (mul)

import Effect (Effect)
import Jax.Core (D2, NDArray, matmul, mul, ref, silu)

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
-- | All inputs are borrowed.
mlp
  :: NDArray D2  -- ^ x
  -> NDArray D2  -- ^ gate_proj
  -> NDArray D2  -- ^ up_proj
  -> NDArray D2  -- ^ down_proj
  -> Effect (NDArray D2)
mlp x gateProj upProj downProj = do
  -- gate = x · gate_proj
  xR1 <- ref x
  gpR <- ref gateProj
  gate <- matmul xR1 gpR
  -- up = x · up_proj
  xR2 <- ref x
  upR <- ref upProj
  up <- matmul xR2 upR
  -- inner = silu(gate) · up
  gateSilu <- silu gate
  inner <- mul gateSilu up
  -- out = inner · down_proj
  dpR <- ref downProj
  matmul inner dpR
