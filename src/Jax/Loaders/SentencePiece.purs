module Jax.Loaders.SentencePiece
  ( SentencePiece
  , fromBinary
  , encode
  , decode
  , bosToken
  , eosToken
  , vocabSize
  ) where

import Effect (Effect)
import Effect.Uncurried (EffectFn1, EffectFn2, runEffectFn1, runEffectFn2)
import Foreign (Foreign)

-- | SentencePiece Unigram tokenizer (the family used by Llama / Gemma /
-- | T5). Construct from a `.model` file's bytes via `fromBinary`.
foreign import data SentencePiece :: Type

foreign import fromBinaryImpl :: EffectFn1 Foreign SentencePiece
foreign import encodeImpl :: EffectFn2 SentencePiece String (Array Int)
foreign import decodeImpl :: EffectFn2 SentencePiece (Array Int) String
foreign import bosTokenImpl :: SentencePiece -> Int
foreign import eosTokenImpl :: SentencePiece -> Int
foreign import vocabSizeImpl :: SentencePiece -> Int

-- | Load from raw `.model` bytes (Uint8Array or ArrayBuffer-likes).
fromBinary :: Foreign -> Effect SentencePiece
fromBinary = runEffectFn1 fromBinaryImpl

-- | Encode a string to token IDs.
encode :: SentencePiece -> String -> Effect (Array Int)
encode = runEffectFn2 encodeImpl

-- | Decode token IDs back to a string.
decode :: SentencePiece -> Array Int -> Effect String
decode = runEffectFn2 decodeImpl

-- | Beginning-of-sequence token ID.
bosToken :: SentencePiece -> Int
bosToken = bosTokenImpl

-- | End-of-sequence token ID.
eosToken :: SentencePiece -> Int
eosToken = eosTokenImpl

-- | Vocabulary size.
vocabSize :: SentencePiece -> Int
vocabSize = vocabSizeImpl
