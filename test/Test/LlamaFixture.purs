module Test.LlamaFixture
  ( makeLlamaFixture
  , llamaFixtureCfg
  ) where

import Effect (Effect)
import Foreign (Foreign)

foreign import makeLlamaFixtureImpl :: Effect Foreign
foreign import llamaFixtureCfgImpl ::
  { vocab :: Int
  , hidden :: Int
  , nHeads :: Int
  , nKvHeads :: Int
  , headDim :: Int
  , intermediate :: Int
  , nLayers :: Int
  }

makeLlamaFixture :: Effect Foreign
makeLlamaFixture = makeLlamaFixtureImpl

llamaFixtureCfg ::
  { vocab :: Int
  , hidden :: Int
  , nHeads :: Int
  , nKvHeads :: Int
  , headDim :: Int
  , intermediate :: Int
  , nLayers :: Int
  }
llamaFixtureCfg = llamaFixtureCfgImpl
