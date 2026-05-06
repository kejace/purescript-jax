module Browser where

import Prelude

import Data.Array (intercalate)
import Data.Int (fromString)
import Data.Maybe (Maybe(..), fromMaybe)
import Data.String (split, trim)
import Data.String.Pattern (Pattern(..))
import Data.Traversable (traverse)
import Effect (Effect)
import Effect.Console (log)
import Effect.Uncurried
  ( EffectFn1
  , EffectFn2
  , runEffectFn1
  , runEffectFn2
  )
import Foreign (Foreign)
import Unsafe.Coerce (unsafeCoerce)

-- Default benchmark prompt: matches the previous synthetic vocab.
defaultBenchPrompt :: Array Int
defaultBenchPrompt = [ 0, 1, 2 ]

-- DOM / Worker FFI -----------------------------------------------------------

foreign import data Element :: Type
foreign import data Worker :: Type

foreign import getElByIdImpl :: EffectFn1 String Element
foreign import setTextImpl :: EffectFn2 Element String Unit
foreign import appendTextImpl :: EffectFn2 Element String Unit
foreign import onClickImpl :: EffectFn2 Element (Effect Unit) Unit
foreign import getValueImpl :: EffectFn1 Element String
foreign import setHtmlImpl :: EffectFn2 Element String Unit
foreign import setStyleDisplayImpl :: EffectFn2 Element String Unit

foreign import mkWorkerImpl :: EffectFn1 String Worker
foreign import postMessageImpl :: forall a. EffectFn2 Worker a Unit
foreign import onMessageImpl :: EffectFn2 Worker (Foreign -> Effect Unit) Unit

getElById :: String -> Effect Element
getElById = runEffectFn1 getElByIdImpl

setText :: Element -> String -> Effect Unit
setText = runEffectFn2 setTextImpl

appendText :: Element -> String -> Effect Unit
appendText = runEffectFn2 appendTextImpl

onClick :: Element -> Effect Unit -> Effect Unit
onClick = runEffectFn2 onClickImpl

getValue :: Element -> Effect String
getValue = runEffectFn1 getValueImpl

setHtml :: Element -> String -> Effect Unit
setHtml = runEffectFn2 setHtmlImpl

setStyleDisplay :: Element -> String -> Effect Unit
setStyleDisplay = runEffectFn2 setStyleDisplayImpl

mkWorker :: String -> Effect Worker
mkWorker = runEffectFn1 mkWorkerImpl

postMessage :: forall a. Worker -> a -> Effect Unit
postMessage = runEffectFn2 postMessageImpl

onMessage :: Worker -> (Foreign -> Effect Unit) -> Effect Unit
onMessage = runEffectFn2 onMessageImpl

-- Main entry -----------------------------------------------------------------

main :: Effect Unit
main = do
  log "[browser] booting; spawning worker"
  worker <- mkWorker "/dist/worker.js"
  -- UI elements
  backendEl <- getElById "backend"
  generateBtn <- getElById "generate"
  benchmarkBtn <- getElById "benchmark"
  loadBtn <- getElById "loadWeights"
  clearCacheBtn <- getElById "clearCache"
  weightsUrlEl <- getElById "weightsUrl"
  tokenizerUrlEl <- getElById "tokenizerUrl"
  loadStatusEl <- getElById "loadStatus"
  promptEl <- getElById "prompt"
  maxNewEl <- getElById "maxNew"
  outputEl <- getElById "output"
  statsEl <- getElById "stats"
  benchTable <- getElById "benchTable"
  benchBody <- getElById "benchBody"
  setText backendEl "spawning worker…"
  -- Wire incoming messages from the worker.
  onMessage worker \msg -> do
    let
      m = unsafeCoerce msg ::
        { kind :: String
        , backend :: String
        , value :: Int
        , text :: String
        , isEos :: Boolean
        , count :: Int
        , ok :: Boolean
        , prefillMs :: Number
        , decodeMs :: Number
        , totalMs :: Number
        , tensorCount :: Int
        , fetchMs :: Number
        , parseMs :: Number
        , vocabSize :: Int
        , promptTokens :: Int
        , stoppedAtEos :: Boolean
        , url :: String
        , err :: String
        }
      showMs :: Number -> String
      showMs = toFixed1
    case m.kind of
      "ready" -> do
        setText backendEl m.backend
        log $ "[browser] worker ready on " <> m.backend
      "tokenText" ->
        if m.isEos then pure unit  -- hide the EOS marker from rendered output
        else appendText outputEl m.text
      "generateStart" ->
        setText statsEl ("encoded " <> show m.promptTokens <> " prompt tokens · running…")
      "generateError" -> do
        setText statsEl ("error: " <> m.err)
        setText outputEl ""
      "done" -> do
        let
          stop = if m.stoppedAtEos then " · stopped at EOS" else ""
          msg' = "generated " <> show m.count <> " tokens · "
            <> "prefill " <> toFixed1 m.prefillMs <> " ms · "
            <> "decode " <> toFixed1 m.decodeMs <> " ms · "
            <> "total " <> toFixed1 m.totalMs <> " ms · "
            <> showTps m.count m.totalMs <> " tok/s" <> stop
        setText statsEl msg'
      "benchResult" -> appendBenchRow benchBody m
      "benchmarkDone" -> setText statsEl "benchmark complete"
      "loadStart" -> setText loadStatusEl ("fetching " <> m.url <> "…")
      "loadTokenizer" -> setText loadStatusEl
        ("tokenizer ready (vocab " <> show m.vocabSize <> ") · fetching weights…")
      "loadFetched" -> setText loadStatusEl
        ("fetched in " <> showMs m.fetchMs <> " ms · parsing…")
      "loadDone" -> setText loadStatusEl
        ( "loaded " <> show m.tensorCount <> " tensors · "
            <> "fetch+parse total " <> showMs m.totalMs <> " ms"
        )
      "loadError" -> setText loadStatusEl ("load failed: " <> m.err)
      _ -> log $ "[browser] unknown message: " <> m.kind
  -- Wire the Generate button.
  onClick generateBtn do
    setStyleDisplay benchTable "none"
    promptStr <- getValue promptEl
    maxNewStr <- getValue maxNewEl
    let maxNew = fromMaybe 40 (fromString maxNewStr)
    debug <- hasUrlParamImpl "debug"
    -- Render the prompt as the visible head of the output, then stream
    -- generated text into the same element.
    setText outputEl promptStr
    setText statsEl "encoding…"
    postMessage worker
      { kind: "generate"
      , prompt: promptStr
      , maxNew
      , debug
      }
  -- Wire the Load Model button (weights + tokenizer together).
  onClick loadBtn do
    setText loadStatusEl "starting…"
    url <- getValue weightsUrlEl
    tokenizerUrl <- getValue tokenizerUrlEl
    postMessage worker
      { kind: "loadModel", url, tokenizerUrl }
  -- Wire the Clear Cache button.
  onClick clearCacheBtn do
    setText loadStatusEl "clearing OPFS cache…"
    clearOpfs
      (\n -> setText loadStatusEl ("cache cleared (" <> show n <> " entries)"))
      (\err -> setText loadStatusEl ("cache clear failed: " <> err))
  -- Wire the Benchmark button. Always uses the synthetic model on a
  -- fixed token prompt — no tokenizer/weights required.
  onClick benchmarkBtn do
    setText outputEl ""
    setText statsEl "benchmarking…"
    setHtml benchBody ""
    setStyleDisplay benchTable "table"
    maxNewStr <- getValue maxNewEl
    let maxNew = fromMaybe 10 (fromString maxNewStr)
    postMessage worker
      { kind: "benchmark"
      , benchPrompt: defaultBenchPrompt
      , maxNew
      }

-- Helpers --------------------------------------------------------------------

parseTokenList :: String -> Array Int
parseTokenList s =
  case traverse fromString (map trim (split (Pattern ",") s)) of
    Just xs -> xs
    Nothing -> [ 0 ]

formatTokens :: Array Int -> String
formatTokens xs = "[" <> intercalate ", " (map show xs) <> "]"

showTps :: Int -> Number -> String
showTps count totalMs =
  if totalMs > 0.0 then
    toFixed1 ((toN count) * 1000.0 / totalMs)
  else "∞"

-- Append a row to the benchmark table.
appendBenchRow
  :: forall r
   . Element
  -> { backend :: String, ok :: Boolean, count :: Int
     , prefillMs :: Number, decodeMs :: Number, totalMs :: Number
     | r }
  -> Effect Unit
appendBenchRow body r = do
  current <- getInnerHtmlImpl body
  let
    row =
      if r.ok then
        "<tr><td><b>" <> r.backend <> "</b></td>"
          <> "<td>" <> show r.count <> "</td>"
          <> "<td>" <> toFixed1 r.prefillMs <> "</td>"
          <> "<td>" <> toFixed1 r.decodeMs <> "</td>"
          <> "<td>" <> toFixed1 r.totalMs <> "</td>"
          <> "<td>" <> showTps r.count r.totalMs <> "</td></tr>"
      else
        "<tr><td><b>" <> r.backend <> "</b></td><td colspan=\"5\">unavailable</td></tr>"
  setHtml body (current <> row)

foreign import getInnerHtmlImpl :: Element -> Effect String

-- | Read a URL query parameter (returns `true` when present, regardless
-- | of value — supports both `?debug` and `?debug=1`).
foreign import hasUrlParamImpl :: String -> Effect Boolean

foreign import clearOpfsImpl
  :: EffectFn2 (Int -> Effect Unit) (String -> Effect Unit) Unit

clearOpfs
  :: (Int -> Effect Unit)
  -> (String -> Effect Unit)
  -> Effect Unit
clearOpfs = runEffectFn2 clearOpfsImpl

-- | Format a Number to one decimal place (e.g. for ms / tok-s in the UI).
foreign import toFixed1Impl :: Number -> String

toFixed1 :: Number -> String
toFixed1 = toFixed1Impl

toN :: Int -> Number
toN = toNumberImpl

foreign import toNumberImpl :: Int -> Number
