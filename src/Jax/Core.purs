module Jax.Core
  ( NDArray
  , D1
  , D2
  , D3
  , D4
  , init
  , setDefaultDevice
  , devicePut
  , ref
  , dispose
  , refCount
  , array1D
  , arrayInt1D
  , arrayNested
  , zeros
  , ones
  , arange
  , linspace
  , add
  , mul
  , sub
  , matmul
  , rsqrt
  , mean
  , meanAxisKeep
  , sum
  , sumAxisKeep
  , addScalar
  , mulScalar
  , transpose
  , sigmoid
  , silu
  , square
  , sqrt
  , sin
  , tanh
  , reshape
  , slice
  , sliceAxis
  , sliceLastAxis
  , concat
  , concatAxis
  , repeatAxis
  , take
  , argmax
  , argmin
  , topK
  , softmax
  , logSoftmax
  , oneHot
  , sumAxis
  , cumsum
  , shape
  , dimAt
  , dataSync
  , toJs
  ) where

import Prelude

import Data.Array as Array
import Data.Maybe (fromMaybe)
import Effect (Effect)
import Effect.Uncurried
  ( EffectFn1
  , EffectFn2
  , EffectFn3
  , EffectFn4
  , runEffectFn1
  , runEffectFn2
  , runEffectFn3
  , runEffectFn4
  )
import Foreign (Foreign)

-- | Rank-parameterized opaque tensor handle. The rank tag `d` is a phantom
-- | (catches rank mismatches at compile time without committing to full
-- | type-level dimension arithmetic).
foreign import data NDArray :: Type -> Type

-- | Rank tags. Empty data declarations — never constructed at runtime.
-- | Plan specifies D1/D2/D3 with "..." for higher ranks; D4 covers
-- | typical attention tensors (batch, heads, seq, dim).
data D1
data D2
data D3
data D4

-- Backend ---------------------------------------------------------------------

foreign import initImpl :: Effect Unit
foreign import setDefaultDeviceImpl :: EffectFn1 String Unit

-- | Underlying FFI is rank-agnostic; we re-expose with the input rank
-- | preserved on the way out.
foreign import devicePutImpl
  :: forall d. EffectFn2 (NDArray d) String (NDArray d)

init :: Effect Unit
init = initImpl

setDefaultDevice :: String -> Effect Unit
setDefaultDevice = runEffectFn1 setDefaultDeviceImpl

devicePut :: forall d. NDArray d -> String -> Effect (NDArray d)
devicePut = runEffectFn2 devicePutImpl

-- Reference counting ----------------------------------------------------------
--
-- jax-js consumes arguments on every call. To keep a tensor alive across
-- multiple ops, bump its refcount with `ref` before passing it in.

foreign import refImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import disposeImpl :: forall d. EffectFn1 (NDArray d) Unit
foreign import refCountImpl :: forall d. EffectFn1 (NDArray d) Int

ref :: forall d. NDArray d -> Effect (NDArray d)
ref = runEffectFn1 refImpl

dispose :: forall d. NDArray d -> Effect Unit
dispose = runEffectFn1 disposeImpl

-- | Read jax-js's internal refcount. Diagnostic only; do not branch on it.
refCount :: forall d. NDArray d -> Effect Int
refCount = runEffectFn1 refCountImpl

-- Constructors ----------------------------------------------------------------

foreign import array1DImpl :: EffectFn1 (Array Number) (NDArray D1)
foreign import arrayInt1DImpl :: EffectFn1 (Array Int) (NDArray D1)
foreign import arrayNestedImpl :: forall d. EffectFn1 Foreign (NDArray d)
foreign import zerosImpl :: forall d. EffectFn1 (Array Int) (NDArray d)
foreign import onesImpl :: forall d. EffectFn1 (Array Int) (NDArray d)
foreign import arangeImpl :: EffectFn3 Number Number Number (NDArray D1)
foreign import linspaceImpl :: EffectFn3 Number Number Int (NDArray D1)

array1D :: Array Number -> Effect (NDArray D1)
array1D = runEffectFn1 array1DImpl

-- | Construct an int32 NDArray from a JS Int array. Useful for token IDs
-- | and other index tensors that need integral dtype to feed into `take`.
arrayInt1D :: Array Int -> Effect (NDArray D1)
arrayInt1D = runEffectFn1 arrayInt1DImpl

-- | Construct an NDArray from a nested JS array. Rank is open; the caller
-- | asserts it via the type ascription on the binding site.
arrayNested :: forall d. Foreign -> Effect (NDArray d)
arrayNested = runEffectFn1 arrayNestedImpl

zeros :: forall d. Array Int -> Effect (NDArray d)
zeros = runEffectFn1 zerosImpl

ones :: forall d. Array Int -> Effect (NDArray d)
ones = runEffectFn1 onesImpl

arange :: Number -> Number -> Number -> Effect (NDArray D1)
arange = runEffectFn3 arangeImpl

linspace :: Number -> Number -> Int -> Effect (NDArray D1)
linspace = runEffectFn3 linspaceImpl

-- Binary ops (consume both args) ---------------------------------------------
--
-- Broadcasting policy is currently *unspecified by the plan* (round-2 Tier
-- 2 #8). For now: matmul is rank-preserving (same-rank both sides);
-- elementwise add/mul are open-rank to leave room for broadcasting. If the
-- plan locks down a strict-vs-open broadcast policy, tighten these.

foreign import addImpl
  :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)

foreign import mulImpl
  :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)

foreign import subImpl
  :: forall d e f. EffectFn2 (NDArray d) (NDArray e) (NDArray f)

foreign import matmulImpl
  :: forall d. EffectFn2 (NDArray d) (NDArray d) (NDArray d)

add :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
add = runEffectFn2 addImpl

mul :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
mul = runEffectFn2 mulImpl

sub :: forall d e f. NDArray d -> NDArray e -> Effect (NDArray f)
sub = runEffectFn2 subImpl

matmul :: forall d. NDArray d -> NDArray d -> Effect (NDArray d)
matmul = runEffectFn2 matmulImpl

-- Unary math -----------------------------------------------------------------

foreign import rsqrtImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import meanImpl :: forall d e. EffectFn1 (NDArray d) (NDArray e)
foreign import meanAxisKeepImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import sumImpl :: forall d e. EffectFn1 (NDArray d) (NDArray e)
foreign import sumAxisKeepImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import addScalarImpl :: forall d. EffectFn2 (NDArray d) Number (NDArray d)
foreign import mulScalarImpl :: forall d. EffectFn2 (NDArray d) Number (NDArray d)
foreign import transposeImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import sigmoidImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import siluImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import squareImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import sqrtImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import sinImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)
foreign import tanhImpl :: forall d. EffectFn1 (NDArray d) (NDArray d)

rsqrt :: forall d. NDArray d -> Effect (NDArray d)
rsqrt = runEffectFn1 rsqrtImpl

-- | Reduces rank when called with an axis (jax-js: `a.mean(axis)`); a
-- | full reduction collapses all axes to a scalar. Output rank left open.
mean :: forall d e. NDArray d -> Effect (NDArray e)
mean = runEffectFn1 meanImpl

-- | Mean over a single axis with `keepdims=true`, so the output shape
-- | matches the input shape with the reduced axis collapsed to size 1.
-- | Rank is preserved.
meanAxisKeep :: forall d. NDArray d -> Int -> Effect (NDArray d)
meanAxisKeep = runEffectFn2 meanAxisKeepImpl

sum :: forall d e. NDArray d -> Effect (NDArray e)
sum = runEffectFn1 sumImpl

-- | Sum over a single axis with `keepdims=true`. See `meanAxisKeep`.
sumAxisKeep :: forall d. NDArray d -> Int -> Effect (NDArray d)
sumAxisKeep = runEffectFn2 sumAxisKeepImpl

-- | Add a scalar Number to every element of a tensor. Rank-preserving.
addScalar :: forall d. NDArray d -> Number -> Effect (NDArray d)
addScalar = runEffectFn2 addScalarImpl

-- | Multiply every element of a tensor by a scalar Number. Rank-preserving.
mulScalar :: forall d. NDArray d -> Number -> Effect (NDArray d)
mulScalar = runEffectFn2 mulScalarImpl

transpose :: forall d. NDArray d -> Effect (NDArray d)
transpose = runEffectFn1 transposeImpl

sigmoid :: forall d. NDArray d -> Effect (NDArray d)
sigmoid = runEffectFn1 sigmoidImpl

-- | SiLU / Swish activation: x * sigmoid(x). Bound to `nn.silu` for the
-- | fused implementation rather than computed from primitives.
silu :: forall d. NDArray d -> Effect (NDArray d)
silu = runEffectFn1 siluImpl

square :: forall d. NDArray d -> Effect (NDArray d)
square = runEffectFn1 squareImpl

sqrt :: forall d. NDArray d -> Effect (NDArray d)
sqrt = runEffectFn1 sqrtImpl

sin :: forall d. NDArray d -> Effect (NDArray d)
sin = runEffectFn1 sinImpl

tanh :: forall d. NDArray d -> Effect (NDArray d)
tanh = runEffectFn1 tanhImpl

-- Shape ops -------------------------------------------------------------------

foreign import reshapeImpl
  :: forall d e. EffectFn2 (NDArray d) (Array Int) (NDArray e)

foreign import sliceImpl
  :: forall d. EffectFn3 (NDArray d) Int Int (NDArray d)

foreign import sliceAxisImpl
  :: forall d. EffectFn4 (NDArray d) Int Int Int (NDArray d)

foreign import sliceLastAxisImpl
  :: forall d. EffectFn3 (NDArray d) Int Int (NDArray d)

foreign import concatImpl
  :: forall d. EffectFn1 (Array (NDArray d)) (NDArray d)

foreign import concatAxisImpl
  :: forall d. EffectFn2 (Array (NDArray d)) Int (NDArray d)

foreign import repeatAxisImpl
  :: forall d. EffectFn3 (NDArray d) Int Int (NDArray d)

foreign import takeImpl
  :: forall d e. EffectFn3 (NDArray d) (NDArray D1) Int (NDArray e)

foreign import argmaxImpl
  :: forall d e. EffectFn2 (NDArray d) Int (NDArray e)

foreign import argminImpl
  :: forall d e. EffectFn2 (NDArray d) Int (NDArray e)

foreign import topKImpl
  :: forall d
   . EffectFn3
       (NDArray d)
       Int
       Int
       { values :: NDArray d, indices :: NDArray d }

foreign import softmaxImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import logSoftmaxImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import oneHotImpl :: forall d e. EffectFn2 (NDArray d) Int (NDArray e)
foreign import cumsumImpl :: forall d. EffectFn2 (NDArray d) Int (NDArray d)
foreign import sumAxisImpl :: forall d e. EffectFn2 (NDArray d) Int (NDArray e)

reshape :: forall d e. NDArray d -> Array Int -> Effect (NDArray e)
reshape = runEffectFn2 reshapeImpl

-- | Slice a 1D array's only axis from `start` to `end`. For multi-dim
-- | tensors use `sliceAxis` or `sliceLastAxis`.
slice :: NDArray D1 -> Int -> Int -> Effect (NDArray D1)
slice = runEffectFn3 sliceImpl

-- | Slice along an arbitrary axis (negative indices count from the end).
sliceAxis :: forall d. NDArray d -> Int -> Int -> Int -> Effect (NDArray d)
sliceAxis = runEffectFn4 sliceAxisImpl

-- | Slice along the last axis. Convenience for the common case
-- | (e.g. RoPE's first-half / second-half split of head_dim).
sliceLastAxis :: forall d. NDArray d -> Int -> Int -> Effect (NDArray d)
sliceLastAxis = runEffectFn3 sliceLastAxisImpl

concat :: forall d. Array (NDArray d) -> Effect (NDArray d)
concat = runEffectFn1 concatImpl

-- | Concatenate along an explicit axis.
concatAxis :: forall d. Array (NDArray d) -> Int -> Effect (NDArray d)
concatAxis = runEffectFn2 concatAxisImpl

-- | Repeat each element of `a` `n` times along `axis`. Equivalent to
-- | `torch.repeat_interleave` and `np.repeat`. Use for GQA kv-head
-- | expansion: `k :: [seq, n_kv, head_dim]` → `[seq, n_q, head_dim]` via
-- | `repeatAxis k (n_q `div` n_kv) 1`.
repeatAxis :: forall d. NDArray d -> Int -> Int -> Effect (NDArray d)
repeatAxis = runEffectFn3 repeatAxisImpl

-- | Gather rows / slices from `a` along `axis` according to `indices`.
-- | For embedding lookup: `take table tokenIds 0 :: NDArray D2`.
take :: forall d e. NDArray d -> NDArray D1 -> Int -> Effect (NDArray e)
take = runEffectFn3 takeImpl

-- | Index of maximum element along an axis. Output dtype is int32.
argmax :: forall d e. NDArray d -> Int -> Effect (NDArray e)
argmax = runEffectFn2 argmaxImpl

-- | Index of minimum element along an axis. Output dtype is int32.
argmin :: forall d e. NDArray d -> Int -> Effect (NDArray e)
argmin = runEffectFn2 argminImpl

-- | Top-k along `axis`: returns the `k` largest values (descending) and
-- | their indices. Both outputs have the same rank as the input with
-- | `axis` reduced to size `k`.
topK
  :: forall d
   . NDArray d
  -> Int
  -> Int
  -> Effect { values :: NDArray d, indices :: NDArray d }
topK = runEffectFn3 topKImpl

-- | Softmax along an axis: `nn.softmax`.
softmax :: forall d. NDArray d -> Int -> Effect (NDArray d)
softmax = runEffectFn2 softmaxImpl

-- | Log-softmax along an axis: `nn.logSoftmax`.
logSoftmax :: forall d. NDArray d -> Int -> Effect (NDArray d)
logSoftmax = runEffectFn2 logSoftmaxImpl

-- | One-hot encode integer indices into a `[..., numClasses]` tensor of
-- | floats. Output rank = input rank + 1.
oneHot :: forall d e. NDArray d -> Int -> Effect (NDArray e)
oneHot = runEffectFn2 oneHotImpl

-- | Cumulative sum along an axis.
cumsum :: forall d. NDArray d -> Int -> Effect (NDArray d)
cumsum = runEffectFn2 cumsumImpl

-- | Sum along a single axis (axis is dropped).
sumAxis :: forall d e. NDArray d -> Int -> Effect (NDArray e)
sumAxis = runEffectFn2 sumAxisImpl

-- Inspection (non-consuming) -------------------------------------------------

foreign import shapeImpl :: forall d. EffectFn1 (NDArray d) (Array Int)
foreign import dataSyncImpl :: forall d. EffectFn1 (NDArray d) Foreign
foreign import jsImpl :: forall d. EffectFn1 (NDArray d) Foreign

shape :: forall d. NDArray d -> Effect (Array Int)
shape = runEffectFn1 shapeImpl

-- | Read a single dimension by index. Returns 0 if `axis` is out of
-- | bounds (matches the behaviour of `fromMaybe 0 <<< (sh !! axis)`
-- | that recurs ~20 times across the codebase).
dimAt :: forall d. NDArray d -> Int -> Effect Int
dimAt a axis = do
  sh <- shape a
  pure (fromMaybe 0 (Array.index sh axis))

dataSync :: forall d. NDArray d -> Effect Foreign
dataSync = runEffectFn1 dataSyncImpl

toJs :: forall d. NDArray d -> Effect Foreign
toJs = runEffectFn1 jsImpl
