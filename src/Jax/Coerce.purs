-- | Typed shims for `Foreign` values produced by `Jax.Core.toJs` /
-- | `Jax.Core.dataSync`. Each function below documents the rank/dtype
-- | of the producing NDArray; the implementation is a no-op
-- | reinterpretation in `Foreign`. Confining the shape-coercion
-- | pattern to one module lets the fp-police audit lock down direct
-- | `unsafeCoerce` use across the rest of the codebase.
module Jax.Coerce
  ( asInt
  , asNumber
  , asArray1D
  , asArray1DInt
  , asArray2D
  ) where

import Foreign (Foreign)

-- | Read a `Foreign` from `toJs` on a rank-0 int32 NDArray. Holds when
-- | the source is `argmax` / `argmin` over a 1D tensor, or any other
-- | scalar-producing reduction with int32 output.
foreign import asIntImpl :: Foreign -> Int

asInt :: Foreign -> Int
asInt = asIntImpl

-- | Read a `Foreign` from `toJs` on a rank-0 float NDArray.
foreign import asNumberImpl :: Foreign -> Number

asNumber :: Foreign -> Number
asNumber = asNumberImpl

-- | Read a `Foreign` from `toJs` on a rank-1 float NDArray.
foreign import asArray1DImpl :: Foreign -> Array Number

asArray1D :: Foreign -> Array Number
asArray1D = asArray1DImpl

-- | Read a `Foreign` from `toJs` on a rank-1 int32 NDArray (e.g.
-- | `topK.indices` of a 1D logits tensor).
foreign import asArray1DIntImpl :: Foreign -> Array Int

asArray1DInt :: Foreign -> Array Int
asArray1DInt = asArray1DIntImpl

-- | Read a `Foreign` from `toJs` on a rank-2 float NDArray.
foreign import asArray2DImpl :: Foreign -> Array (Array Number)

asArray2D :: Foreign -> Array (Array Number)
asArray2D = asArray2DImpl
