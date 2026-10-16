-- | int32 arrays and integer arithmetic: exact up to 2^31, where float32
-- | is exact only up to 2^24. Arithmetic wraps on overflow like i32 (so an
-- | exact program checks its ranges, as Eigenbird's i32 circuits do).
-- |
-- | The rank tag is tracked as elsewhere; the dtype is not (an op on an
-- | int32 and a float32 array promotes, as in jax-js).
module Jax.Int
  ( arrayInt32
  , fullInt32
  , toInt32
  , toFloat32
  , dtype
  , addScalarInt
  , mulScalarInt
  , floorDivide
  , remainder
  , rightShift
  , leftShift
  , bitwiseAnd
  , bitwiseOr
  , bitwiseXor
  , negative
  , absolute
  , maximum
  , minimum
  , greater
  , greaterEqual
  , less
  , equal
  , where_
  ) where

import Effect (Effect)
import Effect.Uncurried (EffectFn1, EffectFn2, EffectFn3, runEffectFn1, runEffectFn2, runEffectFn3)
import Jax.Core (NDArray, D1)

foreign import arrayInt32Impl :: EffectFn1 (Array Int) (NDArray D1)
foreign import fullInt32Impl :: forall d. EffectFn2 (Array Int) Int (NDArray d)
foreign import toInt32Impl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import toFloat32Impl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import dtypeImpl :: forall d. EffectFn1 (NDArray d) String
foreign import addScalarIntImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import mulScalarIntImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import floorDivideImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import remainderImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import rightShiftImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import leftShiftImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import bitwiseAndImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import bitwiseOrImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import bitwiseXorImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import negativeImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import absoluteImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import maximumImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import minimumImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import greaterImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import greaterEqualImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import lessImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import equalImpl :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)
foreign import whereImpl :: forall c d e f. EffectFn3 (NDArray c) (NDArray d) (NDArray e) (NDArray f)

-- | A rank-1 int32 array.
arrayInt32 :: Array Int -> Effect (NDArray D1)
arrayInt32 = runEffectFn1 arrayInt32Impl

-- | An int32 array of the given shape, every element `v`.
fullInt32 :: forall d. Array Int -> Int -> Effect (NDArray d)
fullInt32 = runEffectFn2 fullInt32Impl

-- | Cast to int32 (float to int truncates toward zero) and back.
toInt32 :: forall d. NDArray d -> Effect (NDArray d)
toInt32 = runEffectFn1 toInt32Impl

toFloat32 :: forall d. NDArray d -> Effect (NDArray d)
toFloat32 = runEffectFn1 toFloat32Impl

-- | The array's dtype ("int32", "float32", "bool", ...). Non-consuming.
dtype :: forall d. NDArray d -> Effect String
dtype = runEffectFn1 dtypeImpl

-- | `a + n` and `a * n` with an int32 scalar (a weak JS number would keep the
-- | dtype too, but this states it).
addScalarInt :: forall d. NDArray d -> Int -> Effect (NDArray d)
addScalarInt = runEffectFn2 addScalarIntImpl

mulScalarInt :: forall d. NDArray d -> Int -> Effect (NDArray d)
mulScalarInt = runEffectFn2 mulScalarIntImpl

-- | Floor division and the matching remainder (sign of the divisor), as numpy.
floorDivide :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
floorDivide = runEffectFn2 floorDivideImpl

remainder :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
remainder = runEffectFn2 remainderImpl

-- | Arithmetic right shift (floor division by 2^k) and left shift.
rightShift :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
rightShift = runEffectFn2 rightShiftImpl

leftShift :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
leftShift = runEffectFn2 leftShiftImpl

bitwiseAnd :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
bitwiseAnd = runEffectFn2 bitwiseAndImpl

bitwiseOr :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
bitwiseOr = runEffectFn2 bitwiseOrImpl

bitwiseXor :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
bitwiseXor = runEffectFn2 bitwiseXorImpl

negative :: forall d. NDArray d -> Effect (NDArray d)
negative = runEffectFn1 negativeImpl

absolute :: forall d. NDArray d -> Effect (NDArray d)
absolute = runEffectFn1 absoluteImpl

maximum :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
maximum = runEffectFn2 maximumImpl

minimum :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
minimum = runEffectFn2 minimumImpl

-- | Comparisons, giving bool arrays (for `where_`).
greater :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
greater = runEffectFn2 greaterImpl

greaterEqual :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
greaterEqual = runEffectFn2 greaterEqualImpl

less :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
less = runEffectFn2 lessImpl

equal :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
equal = runEffectFn2 equalImpl

-- | `where_ c a b`: a where c, b elsewhere.
where_ :: forall c d e f. NDArray c -> NDArray d -> NDArray e -> Effect (NDArray f)
where_ = runEffectFn3 whereImpl
