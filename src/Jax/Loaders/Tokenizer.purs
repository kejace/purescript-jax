module Jax.Loaders.Tokenizer
  ( Tokenizer
  , defaultTokenizer
  , encode
  , decode
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn2, runEffectFn2)

-- | Opaque BPE tokenizer (jax-js's `BpeEncoding`).
foreign import data Tokenizer :: Type

-- | Module-level pre-loaded `cl100k_base` (GPT-4 / GPT-3.5 family). The
-- | underlying load happens once via top-level await at module init —
-- | the first run fetches the vocab file from CDN.
foreign import defaultTokenizerImpl :: Tokenizer

foreign import encodeImpl :: EffectFn2 Tokenizer String (Array Int)
foreign import decodeImpl :: EffectFn2 Tokenizer (Array Int) String

-- | The default `cl100k_base` tokenizer.
defaultTokenizer :: Tokenizer
defaultTokenizer = defaultTokenizerImpl

-- | Encode a UTF-8 string to a list of token IDs.
encode :: Tokenizer -> String -> Effect (Array Int)
encode = runEffectFn2 encodeImpl

-- | Decode token IDs back to a string.
decode :: Tokenizer -> Array Int -> Effect String
decode = runEffectFn2 decodeImpl
