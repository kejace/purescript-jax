module Test.SafetensorsFixture (makeFixture, makeBF16Fixture) where

import Effect (Effect)
import Foreign (Foreign)

foreign import makeFixtureImpl :: Effect Foreign
foreign import makeBF16FixtureImpl :: Effect Foreign

makeFixture :: Effect Foreign
makeFixture = makeFixtureImpl

makeBF16Fixture :: Effect Foreign
makeBF16Fixture = makeBF16FixtureImpl
