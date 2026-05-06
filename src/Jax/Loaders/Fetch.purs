module Jax.Loaders.Fetch
  ( fetchBytes
  , fetchText
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn3, runEffectFn3)
import Foreign (Foreign)

foreign import fetchBytesImpl
  :: EffectFn3 String (Foreign -> Effect Unit) (String -> Effect Unit) Unit

foreign import fetchTextImpl
  :: EffectFn3 String (String -> Effect Unit) (String -> Effect Unit) Unit

fetchBytes
  :: String
  -> (Foreign -> Effect Unit)
  -> (String -> Effect Unit)
  -> Effect Unit
fetchBytes = runEffectFn3 fetchBytesImpl

fetchText
  :: String
  -> (String -> Effect Unit)
  -> (String -> Effect Unit)
  -> Effect Unit
fetchText = runEffectFn3 fetchTextImpl
