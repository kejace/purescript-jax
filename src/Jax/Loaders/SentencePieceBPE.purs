module Jax.Loaders.SentencePieceBPE
  ( SentencePieceBPE
  , fromBinary
  , encode
  , decode
  , bosToken
  , eosToken
  , unkToken
  , vocabSize
  , setAddDummyPrefix
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn1, EffectFn2, runEffectFn1, runEffectFn2)
import Effect.Uncurried as Effect.Uncurried
import Foreign (Foreign)

-- | SentencePiece BPE tokenizer (the family used by Llama / Mistral /
-- | TinyLlama). Construct from a `tokenizer.model` file's bytes via
-- | `fromBinary`. Distinct from `Jax.Loaders.SentencePiece` which only
-- | handles Unigram-trained models (T5-style); Llama uses BPE.
foreign import data SentencePieceBPE :: Type

foreign import fromBinaryImpl :: EffectFn1 Foreign SentencePieceBPE
foreign import encodeImpl :: EffectFn2 SentencePieceBPE String (Array Int)
foreign import decodeImpl :: EffectFn2 SentencePieceBPE (Array Int) String
foreign import bosTokenImpl :: SentencePieceBPE -> Int
foreign import eosTokenImpl :: SentencePieceBPE -> Int
foreign import unkTokenImpl :: SentencePieceBPE -> Int
foreign import vocabSizeImpl :: SentencePieceBPE -> Int

-- | Load from raw `.model` bytes (Uint8Array or ArrayBuffer-likes).
fromBinary :: Foreign -> Effect SentencePieceBPE
fromBinary = runEffectFn1 fromBinaryImpl

-- | Encode a string to token IDs via greedy SentencePiece BPE.
encode :: SentencePieceBPE -> String -> Effect (Array Int)
encode = runEffectFn2 encodeImpl

-- | Decode token IDs back to a string. Byte-fallback tokens are
-- | accumulated and decoded as UTF-8.
decode :: SentencePieceBPE -> Array Int -> Effect String
decode = runEffectFn2 decodeImpl

bosToken :: SentencePieceBPE -> Int
bosToken = bosTokenImpl

eosToken :: SentencePieceBPE -> Int
eosToken = eosTokenImpl

unkToken :: SentencePieceBPE -> Int
unkToken = unkTokenImpl

vocabSize :: SentencePieceBPE -> Int
vocabSize = vocabSizeImpl

foreign import setAddDummyPrefixImpl
  :: Effect.Uncurried.EffectFn2 SentencePieceBPE Boolean Unit

-- | Toggle the SentencePiece `add_dummy_prefix` normalization. Set to
-- | `false` to match HuggingFace's non-legacy Llama tokenizer behavior
-- | (no leading ▁ prepended to the first word). The default (matching
-- | the proto) is `true`.
setAddDummyPrefix :: SentencePieceBPE -> Boolean -> Effect Unit
setAddDummyPrefix = Effect.Uncurried.runEffectFn2 setAddDummyPrefixImpl
