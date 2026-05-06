module Jax.NN.RoPE
  ( RoPETables
  , precomputeRoPE
  , applyRoPE
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn3, runEffectFn3)
import Jax.Core (D2, NDArray)
import Jax.Tensor (T, addT, concatAxisT, lit, mulT, run, sliceLastAxisT, subT)

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
applyRoPE halfDim x cosT sinT = run $ concatAxisT [ yFirst, ySecond ] (-1)
  where
  dim = halfDim * 2
  xT = lit x
  cT = lit cosT
  sT = lit sinT
  -- Each `lit` is a fresh ref-bumped action — `lit x` used twice ref-bumps
  -- twice. So we reuse the same `xT` / `cT` / `sT` bindings freely.
  xFirst :: T d
  xFirst = sliceLastAxisT xT 0 halfDim
  xSecond :: T d
  xSecond = sliceLastAxisT xT halfDim dim
  yFirst :: T d
  yFirst = subT (mulT xFirst cT) (mulT xSecond sT)
  ySecond :: T d
  ySecond = addT (mulT xFirst sT) (mulT xSecond cT)
