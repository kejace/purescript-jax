-- | Reading arrays back without blocking. jax-js reads WebGPU arrays
-- | synchronously only through `OffscreenCanvas`, which Deno (and some
-- | workers) lack, so `Jax.Core.dataSync` / `toJs` throw there; these do not.
-- | Like `Jax.Core`'s readers, they leave the caller's reference intact.
module Jax.Async
  ( dataAff
  , toJsAff
  , blockUntilReady
  ) where

import Prelude

import Control.Promise (Promise, toAffE)
import Effect.Aff (Aff)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Foreign (Foreign)
import Jax.Core (NDArray)

foreign import dataAffImpl :: forall d. EffectFn1 (NDArray d) (Promise (Array Number))
foreign import toJsAffImpl :: forall d. EffectFn1 (NDArray d) (Promise Foreign)
foreign import blockUntilReadyImpl :: forall d. EffectFn1 (NDArray d) (Promise Unit)

-- | The array's elements, flattened (row-major), as numbers (int32 and
-- | float32 alike).
dataAff :: forall d. NDArray d -> Aff (Array Number)
dataAff a = toAffE (runEffectFn1 dataAffImpl a)

-- | The array as a nested JS array (`Array.jsAsync`).
toJsAff :: forall d. NDArray d -> Aff Foreign
toJsAff a = toAffE (runEffectFn1 toJsAffImpl a)

-- | Wait until the array's pending kernels have run.
blockUntilReady :: forall d. NDArray d -> Aff Unit
blockUntilReady a = toAffE (runEffectFn1 blockUntilReadyImpl a)
