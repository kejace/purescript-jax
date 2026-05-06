-- | Parse a HuggingFace `config.json` (Llama-arch) into our internal
-- | `ModelConfig`. Implemented as an Argonaut codec so missing /
-- | wrong-typed fields produce a structured `JsonDecodeError` rather
-- | than silently propagating `undefined` (the previous JS-side
-- | `JSON.parse` + field-pluck did the latter).
-- |
-- | Optional fields with defaults: `head_dim` (computed from
-- | `hidden_size / num_attention_heads`), `rope_theta` (10000),
-- | `rms_norm_eps` (1e-6).
module Jax.Loaders.Config
  ( parseLlamaConfig
  , configCodec
  ) where

import Prelude

import Data.Argonaut.Parser (jsonParser)
import Data.Codec.Argonaut (JsonCodec, decode, printJsonDecodeError)
import Data.Codec.Argonaut as CA
import Data.Codec.Argonaut.Common as CACommon
import Data.Codec.Argonaut.Record as CAR
import Data.Either (Either(..))
import Data.Maybe (Maybe(..), fromMaybe)
import Effect (Effect)
import Effect.Exception (throw)
import Jax.NN.Block (ModelConfig)

-- | Decode a `Llama`-style `config.json` into our `ModelConfig`.
-- | Throws an Effect-level exception with a printed decoder error if
-- | the JSON is malformed or missing required fields.
parseLlamaConfig :: String -> Effect ModelConfig
parseLlamaConfig s = case jsonParser s of
  Left err -> throw $ "config.json: invalid JSON: " <> err
  Right json -> case decode configCodec json of
    Left err -> throw $ "config.json: " <> printJsonDecodeError err
    Right cfg -> pure cfg

-- | The codec uses the raw HF field names (snake_case), and applies
-- | defaults during the conversion to our PS-side `ModelConfig`
-- | (camelCase, with `headDim` derived if absent and `ropeTheta` /
-- | `normEps` defaulted).
configCodec :: JsonCodec ModelConfig
configCodec = CA.prismaticCodec "ModelConfig" toCfg fromCfg rawCodec
  where
  toCfg :: RawConfig -> Maybe ModelConfig
  toCfg r = Just
    { hidden: r.hidden_size
    , nHeads: r.num_attention_heads
    , nKvHeads: fromMaybe r.num_attention_heads r.num_key_value_heads
    , headDim: fromMaybe (r.hidden_size / r.num_attention_heads) r.head_dim
    , intermediate: r.intermediate_size
    , nLayers: r.num_hidden_layers
    , maxSeqLen: fromMaybe 2048 r.max_position_embeddings
    , vocabSize: r.vocab_size
    , ropeTheta: fromMaybe 10000.0 r.rope_theta
    , normEps: fromMaybe 1.0e-6 r.rms_norm_eps
    }

  fromCfg :: ModelConfig -> RawConfig
  fromCfg c =
    { hidden_size: c.hidden
    , num_attention_heads: c.nHeads
    , num_key_value_heads: Just c.nKvHeads
    , head_dim: Just c.headDim
    , intermediate_size: c.intermediate
    , num_hidden_layers: c.nLayers
    , max_position_embeddings: Just c.maxSeqLen
    , vocab_size: c.vocabSize
    , rope_theta: Just c.ropeTheta
    , rms_norm_eps: Just c.normEps
    }

type RawConfig =
  { hidden_size :: Int
  , num_attention_heads :: Int
  , num_key_value_heads :: Maybe Int
  , head_dim :: Maybe Int
  , intermediate_size :: Int
  , num_hidden_layers :: Int
  , max_position_embeddings :: Maybe Int
  , vocab_size :: Int
  , rope_theta :: Maybe Number
  , rms_norm_eps :: Maybe Number
  }

rawCodec :: JsonCodec RawConfig
rawCodec = CAR.object "LlamaConfigRaw"
  { hidden_size: CA.int
  , num_attention_heads: CA.int
  , num_key_value_heads: CACommon.maybe CA.int
  , head_dim: CACommon.maybe CA.int
  , intermediate_size: CA.int
  , num_hidden_layers: CA.int
  , max_position_embeddings: CACommon.maybe CA.int
  , vocab_size: CA.int
  , rope_theta: CACommon.maybe CA.number
  , rms_norm_eps: CACommon.maybe CA.number
  }
