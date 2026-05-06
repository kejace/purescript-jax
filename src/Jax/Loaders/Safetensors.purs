module Jax.Loaders.Safetensors
  ( SafetensorsMap
  , parseSafetensors
  , tensorNames
  , getTensor
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn1, EffectFn2, runEffectFn1, runEffectFn2)
import Foreign (Foreign)
import Jax.Core (NDArray)

-- | Opaque map of tensor-name → NDArray, produced by parsing a
-- | safetensors blob. Each NDArray inside lives until the map itself is
-- | discarded (or the user `dispose`s individual tensors, which is fine
-- | since they're just NDArray handles).
foreign import data SafetensorsMap :: Type

foreign import parseSafetensorsImpl
  :: EffectFn1 Foreign SafetensorsMap

foreign import tensorNamesImpl
  :: EffectFn1 SafetensorsMap (Array String)

foreign import getTensorImpl
  :: forall d. EffectFn2 SafetensorsMap String (NDArray d)

-- | Parse a safetensors blob (Uint8Array or ArrayBuffer) into a map of
-- | tensors. dtypes declared in the header are mapped to jax-js dtypes
-- | (F32 → float32, I32 → int32, etc.); unsupported dtypes throw.
parseSafetensors :: Foreign -> Effect SafetensorsMap
parseSafetensors = runEffectFn1 parseSafetensorsImpl

-- | List the tensor names in a parsed map.
tensorNames :: SafetensorsMap -> Effect (Array String)
tensorNames = runEffectFn1 tensorNamesImpl

-- | Look up a tensor by name. The returned NDArray is a *borrowed*
-- | handle from the map; if you need to retain it past the map's
-- | lifetime, ref-bump it.
getTensor :: forall d. SafetensorsMap -> String -> Effect (NDArray d)
getTensor = runEffectFn2 getTensorImpl
