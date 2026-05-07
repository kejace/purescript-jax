-- | Reimagined microGPT-spirit CLI demo — top-to-bottom, real
-- | transformer, real autograd. The interesting code lives in
-- | `Jax.Demo.Microgpt`; this file is the CLI driver.
-- |
-- | The same pipeline runs in the browser tab — see the tab UI at
-- | `index.html` and `src/Browser.purs`. Both call into the same
-- | `Jax.Demo.Microgpt.runMicrogpt`.
module Microgpt.Jax.Main where

import Prelude

import Effect (Effect)
import Effect.Console (log)
import Jax.Core (init, setDefaultDevice)
import Jax.Demo.Microgpt (defaultParams, runMicrogpt)

main :: Effect Unit
main = do
  init
  _ <- setDefaultDevice "wasm"
  log "[microgpt-jax] start"
  runMicrogpt defaultParams
    { onStart: \r -> log $ "[microgpt-jax] paramCount=" <> show r.paramCount
                       <> " · vocab=" <> show r.vocabSize
    , onProgress: \i loss ->
        when (mod i 10 == 0 || i <= 1) do
          log $ "  step " <> show i <> " · loss " <> show loss
    , onSampled: \i text -> log $ "  sample " <> show (i + 1) <> ": " <> text
    , onDone: \finalLoss -> do
        log $ "[microgpt-jax] training done · final loss " <> show finalLoss
        log "[microgpt-jax] done"
    }
