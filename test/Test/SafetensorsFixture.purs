module Test.SafetensorsFixture (makeFixture) where

import Effect (Effect)
import Foreign (Foreign)

foreign import makeFixtureImpl :: Effect Foreign

makeFixture :: Effect Foreign
makeFixture = makeFixtureImpl
