module Jax.Random
  ( Key
  , mkKey
  , splitKey2
  , sampleCategorical
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn1, EffectFn2, runEffectFn1, runEffectFn2)
import Jax.Core (D1, NDArray, dispose, toJs)
import Unsafe.Coerce (unsafeCoerce)

-- | Opaque jax-js PRNG key. At runtime it's an int32 NDArray of small
-- | length (typically 2). Refcount-managed like any other tensor.
foreign import data Key :: Type

foreign import mkKeyImpl :: EffectFn1 Int Key
foreign import splitKey2Impl :: EffectFn1 Key { a :: Key, b :: Key }
foreign import sampleCategoricalImpl
  :: forall d. EffectFn2 Key (NDArray d) (NDArray D1)

-- | Construct a PRNG key from an Int seed. Same seed → same key.
mkKey :: Int -> Effect Key
mkKey = runEffectFn1 mkKeyImpl

-- | Split a key into two independent keys. Both are returned; the
-- | original is consumed.
splitKey2 :: Key -> Effect { a :: Key, b :: Key }
splitKey2 = runEffectFn1 splitKey2Impl

-- | Sample a categorical from a 1D logits tensor. Consumes both `key`
-- | and `logits`. Returns the sampled index as an Int.
sampleCategorical :: Key -> NDArray D1 -> Effect Int
sampleCategorical key logits = do
  -- categorical's output is a rank-0 int32 NDArray; toJs returns the
  -- scalar number.
  idxArr <- runEffectFn2 sampleCategoricalImpl key logits
  raw <- toJs idxArr
  dispose idxArr
  pure (unsafeCoerce raw :: Int)

