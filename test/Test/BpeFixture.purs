module Test.BpeFixture
  ( loadTokenizerBytes
  , fixtureMeta
  , fixtureCases
  , Case
  , Meta
  ) where

import Effect (Effect)
import Foreign (Foreign)

type Meta =
  { vocabSize :: Int
  , bos :: Int
  , eos :: Int
  , unk :: Int
  }

type Case =
  { text :: String
  , ids :: Array Int
  , decoded :: String
  }

foreign import loadTokenizerBytesImpl :: Effect Foreign
foreign import fixtureMetaImpl :: Effect Meta
foreign import fixtureCasesImpl :: Effect (Array Case)

loadTokenizerBytes :: Effect Foreign
loadTokenizerBytes = loadTokenizerBytesImpl

fixtureMeta :: Effect Meta
fixtureMeta = fixtureMetaImpl

fixtureCases :: Effect (Array Case)
fixtureCases = fixtureCasesImpl
