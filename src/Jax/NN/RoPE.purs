module Jax.NN.RoPE
  ( RoPETables
  , precomputeRoPE
  , applyRoPE
  ) where

import Prelude hiding (add, mul, sub)

import Effect (Effect)
import Effect.Uncurried (EffectFn3, runEffectFn3)
import Jax.Core
  ( D2
  , NDArray
  , add
  , concatAxis
  , mul
  , ref
  , sliceLastAxis
  , sub
  )

-- | Precomputed rotary frequency tables. cos and sin both have shape
-- | `[maxSeqLen, dim/2]`. These are constants for the model's lifetime —
-- | the caller should wrap each in `LongLived`.
type RoPETables = { cos :: NDArray D2, sin :: NDArray D2 }

foreign import precomputeRoPEImpl
  :: EffectFn3 Int Int Number RoPETables

-- | Build sin/cos lookup tables. `dim` is the per-head dimension and
-- | must be even; `theta` is the geometric base (typically 10000.0).
precomputeRoPE
  :: Int        -- ^ dim (per-head)
  -> Int        -- ^ maxSeqLen
  -> Number     -- ^ theta (base)
  -> Effect RoPETables
precomputeRoPE = runEffectFn3 precomputeRoPEImpl

-- | Apply RoPE rotation to a query/key tensor with the provided cos/sin
-- | tables. Rank-polymorphic so it works on both `[seq, dim]` (D2,
-- | single-head) and `[seq, n_heads, dim]` (D3, multi-head). The caller
-- | is responsible for shaping the cos/sin tables to broadcast properly
-- | against `x` — for D3 inputs, reshape from `[seq, halfDim]` to
-- | `[seq, 1, halfDim]` before passing in.
-- |
-- | Convention: half-split (matches Llama / nanoGPT). Splits the last
-- | axis into halves x_first / x_second and applies:
-- |   y_first  = x_first * cos - x_second * sin
-- |   y_second = x_first * sin + x_second * cos
-- |
-- | All inputs are borrowed; the output is a fresh tensor.
applyRoPE
  :: forall d
   . Int           -- ^ halfDim (= last_dim / 2)
  -> NDArray d     -- ^ x with last axis = dim
  -> NDArray d     -- ^ cos table, broadcast-compatible with x
  -> NDArray d     -- ^ sin table, broadcast-compatible with x
  -> Effect (NDArray d)
applyRoPE halfDim x cosT sinT = do
  let dim = halfDim * 2
  -- Split x into halves on the last axis.
  xR1 <- ref x
  xFirst <- sliceLastAxis xR1 0 halfDim
  xR2 <- ref x
  xSecond <- sliceLastAxis xR2 halfDim dim
  -- y_first = x_first * cos - x_second * sin
  xFirstR1 <- ref xFirst
  cosR1 <- ref cosT
  prodFC <- mul xFirstR1 cosR1
  xSecondR1 <- ref xSecond
  sinR1 <- ref sinT
  prodSS <- mul xSecondR1 sinR1
  yFirst <- sub prodFC prodSS
  -- y_second = x_first * sin + x_second * cos
  cosR2 <- ref cosT
  prodSC <- mul xSecond cosR2
  sinR2 <- ref sinT
  prodFS <- mul xFirst sinR2
  ySecond <- add prodFS prodSC
  -- Concatenate along the last axis.
  concatAxis [ yFirst, ySecond ] (-1)
