module Jax.Loaders.Config
  ( parseLlamaConfig
  ) where

import Prelude

import Effect (Effect)
import Jax.NN.Block (ModelConfig)

foreign import parseLlamaConfigImpl :: String -> ModelConfig

-- | Parse a HuggingFace `config.json` (text) into our `ModelConfig`.
-- | Supplies sensible defaults for missing fields (e.g. rope_theta=10000).
parseLlamaConfig :: String -> Effect ModelConfig
parseLlamaConfig s = pure (parseLlamaConfigImpl s)
