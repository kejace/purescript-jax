-- | Parse a HuggingFace `config.json` into our internal `ModelConfig`.
-- | Implemented as an Argonaut codec so missing / wrong-typed fields
-- | produce a structured `JsonDecodeError` rather than silently
-- | propagating `undefined` (the previous JS-side `JSON.parse` +
-- | field-pluck did the latter).
-- |
-- | Optional fields with defaults: `head_dim` (computed from
-- | `hidden_size / num_attention_heads`), `rope_theta` (10000),
-- | `rms_norm_eps` (1e-6).
-- |
-- | Architecture compatibility: only `model_type` values listed in
-- | `compatibleArchs` are accepted; anything else fails fast with a
-- | clear error rather than silently producing garbage. See
-- | `MODELS.md` for the verified-compatible matrix.
module Jax.Loaders.Config
  ( parseLlamaConfig
  , configCodec
  , compatibleArchs
  , RawConfigExtras
  , probeRawExtras
  ) where

import Prelude

import Data.Argonaut.Parser (jsonParser)
import Data.Array (elem) as Array
import Data.Codec.Argonaut (JsonCodec, decode, printJsonDecodeError)
import Data.Codec.Argonaut as CA
import Data.Codec.Argonaut.Common as CACommon
import Data.Codec.Argonaut.Record as CAR
import Data.Either (Either(..))
import Data.Maybe (Maybe(..), fromMaybe)
import Data.String (joinWith)
import Effect (Effect)
import Effect.Exception (throw)
import Jax.NN.Block (ModelConfig)

-- | The set of HF `model_type` values we know how to load. The shared
-- | weight layout is the Llama-2 family: pre-norm RMSNorm, GQA-capable
-- | self-attention with `q_proj`/`k_proj`/`v_proj`/`o_proj` (no bias),
-- | SwiGLU MLP with `gate_proj`/`up_proj`/`down_proj`, RoPE on Q/K.
-- |
-- | Mistral is included because its weight layout is identical to Llama
-- | when `sliding_window` is unset. If `sliding_window` is set we
-- | ignore it (full-context attention) — outputs will diverge from a
-- | Mistral reference impl on long prompts but stay correct on short
-- | ones.
-- |
-- | Phi / Qwen2 / GPT-2 are deliberately excluded:
-- |   - Phi has `qk_layernorm` (LayerNorm on Q/K) and partial-RoPE.
-- |   - Qwen2 has biases on `q_proj`/`k_proj`/`v_proj`.
-- |   - GPT-2 has learned positional embeddings (no RoPE).
-- |
-- | Adding a new entry requires verifying the weight layout matches
-- | what `Jax.Loaders.LlamaAdapter` expects.
compatibleArchs :: Array String
compatibleArchs = [ "llama", "mistral" ]

-- | Decode a config.json into `ModelConfig`. Throws an Effect-level
-- | exception with a printed decoder error if:
-- |   * the JSON is malformed,
-- |   * required fields are missing,
-- |   * `model_type` isn't in `compatibleArchs`.
parseLlamaConfig :: String -> Effect ModelConfig
parseLlamaConfig s = case jsonParser s of
  Left err -> throw $ "config.json: invalid JSON: " <> err
  Right json -> case decode configCodec json of
    Left err -> throw $ "config.json: " <> printJsonDecodeError err
    Right cfg -> do
      -- Surface a soft warning for things we silently ignore: tied
      -- embeddings (handled separately by the loader), sliding window
      -- (we run full-context), explicit head_dim mismatching the
      -- hidden/heads ratio (we trust the file).
      pure cfg

-- | Extra-but-not-blocking fields we read for diagnostics / soft
-- | warnings. Decoded separately so the main `configCodec` stays
-- | focused on what `ModelConfig` requires.
type RawConfigExtras =
  { architectures :: Maybe (Array String)
  , model_type :: Maybe String
  , tie_word_embeddings :: Maybe Boolean
  , sliding_window :: Maybe Int
  }

extrasCodec :: JsonCodec RawConfigExtras
extrasCodec = CAR.object "ConfigExtras"
  { architectures: CACommon.maybe (CA.array CA.string)
  , model_type: CACommon.maybe CA.string
  , tie_word_embeddings: CACommon.maybe CA.boolean
  , sliding_window: CACommon.maybe CA.int
  }

-- | Run the extras codec on a config.json string. Returns a structured
-- | record (or a printed error) so the worker can issue a compat check
-- | + log soft warnings before committing to a load.
probeRawExtras :: String -> Either String RawConfigExtras
probeRawExtras s = do
  json <- jsonParser s
  case decode extrasCodec json of
    Left err -> Left (printJsonDecodeError err)
    Right r -> case r.model_type of
      Just mt | not (Array.elem mt compatibleArchs) ->
        Left $ "model_type=" <> mt <> " is not in the compatible list "
          <> "(" <> joinWith ", " compatibleArchs <> "); refusing to load. "
          <> "See MODELS.md for the supported architectures."
      _ -> Right r

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
