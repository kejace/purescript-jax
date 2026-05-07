module Browser where

import Prelude

import Data.Array (null) as Array
import Data.Array (intercalate)
import Data.Either (Either(..))
import Data.Int (fromString)
import Data.Maybe (Maybe(..), fromMaybe)
import Data.Number (fromString) as Number
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
import Jax.Worker.Protocol
  ( WorkerIn(..)
  , WorkerOut(..)
  , decodeStr
  , encodeStr
  , workerInCodec
  , workerOutCodec
  )

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
foreign import postMessageImpl :: EffectFn2 Worker String Unit
foreign import onMessageImpl :: EffectFn2 Worker (String -> Effect Unit) Unit

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

postMessageRaw :: Worker -> String -> Effect Unit
postMessageRaw = runEffectFn2 postMessageImpl

-- | Encode a `WorkerIn` and post it.
postIn :: Worker -> WorkerIn -> Effect Unit
postIn worker msg = postMessageRaw worker (encodeStr workerInCodec msg)

onMessageRaw :: Worker -> (String -> Effect Unit) -> Effect Unit
onMessageRaw = runEffectFn2 onMessageImpl

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
  modelPresetEl <- getElById "modelPreset"
  loadStatusEl <- getElById "loadStatus"
  promptEl <- getElById "prompt"
  maxNewEl <- getElById "maxNew"
  outputEl <- getElById "output"
  statsEl <- getElById "stats"
  benchTable <- getElById "benchTable"
  benchBody <- getElById "benchBody"
  trainBtn <- getElById "trainSynthetic"
  trainStepsEl <- getElById "trainSteps"
  trainLREl <- getElById "trainLR"
  samplingModeEl <- getElById "samplingMode"
  temperatureEl <- getElById "temperature"
  topKEl <- getElById "topK"
  topPEl <- getElById "topP"
  -- microGPT tab elements
  microgptCorpusEl <- getElById "microgptCorpus"
  microgptStepsEl <- getElById "microgptSteps"
  microgptLREl <- getElById "microgptLR"
  microgptTempEl <- getElById "microgptTemp"
  microgptNumSamplesEl <- getElById "microgptNumSamples"
  microgptMaxLenEl <- getElById "microgptMaxLen"
  microgptTrainBtn <- getElById "microgptTrain"
  microgptStatusEl <- getElById "microgptStatus"
  microgptStatsEl <- getElById "microgptStats"
  microgptSamplesEl <- getElById "microgptSamples"
  setText backendEl "spawning worker…"
  -- Wire incoming messages from the worker.
  onMessageRaw worker \raw -> case decodeStr workerOutCodec raw of
    Left err -> log $ "[browser] decode error: " <> err <> " (raw: " <> raw <> ")"
    Right msg -> case msg of
      Ready r -> do
        setText backendEl r.backend
        log $ "[browser] worker ready on " <> r.backend
      TokenText r ->
        if r.isEos then pure unit
        else appendText outputEl r.text
      GenerateStart r ->
        setText statsEl ("encoded " <> show r.promptTokens <> " prompt tokens · running…")
      GenerateError r -> do
        setText statsEl ("error: " <> r.err)
        setText outputEl ""
      Done r -> do
        let
          stop = if r.stoppedAtEos then " · stopped at EOS" else ""
          msg' = "generated " <> show r.count <> " tokens · "
            <> "prefill " <> toFixed1 r.prefillMs <> " ms · "
            <> "decode " <> toFixed1 r.decodeMs <> " ms · "
            <> "total " <> toFixed1 r.totalMs <> " ms · "
            <> showTps r.count r.totalMs <> " tok/s" <> stop
        setText statsEl msg'
      BenchResult r -> appendBenchRow benchBody r
      BenchmarkDone -> setText statsEl "benchmark complete"
      TrainStart r -> setText statsEl
        ( "training " <> show r.paramCount <> " params · "
            <> "initial loss " <> toFixed4 r.initialLoss
            <> " · L2 " <> toFixed4 r.initialL2
            <> formatPerLayer r.initialPerLayerL2
            <> " · " <> show r.steps <> " steps planned"
        )
      TrainStep r -> setText statsEl
        ( "step " <> show r.step <> " · loss " <> toFixed4 r.loss )
      TrainDone r -> setText statsEl
        ( "training complete · final loss " <> toFixed4 r.finalLoss
            <> " · L2 " <> toFixed4 r.finalL2
            <> formatPerLayer r.finalPerLayerL2
            <> " · before/after gen: "
            <> formatTokens r.initialGen <> " → " <> formatTokens r.finalGen
            <> " · " <> toFixed1 r.totalMs <> " ms"
        )
      TrainError r -> setText statsEl ("train error: " <> r.err)
      MicrogptStart r -> do
        setText microgptStatusEl ""
        setText microgptStatsEl
          ( "training " <> show r.paramCount <> " params · vocab "
              <> show r.vocabSize <> " · " <> show r.numSteps <> " steps planned"
          )
        setText microgptSamplesEl ""
      MicrogptStep r -> setText microgptStatsEl
        ( "step " <> show r.step <> " · loss " <> toFixed4 r.loss )
      MicrogptSample r -> appendText microgptSamplesEl
        ("→ " <> r.text <> "\n")
      MicrogptDone r -> setText microgptStatusEl
        ( "done · final loss " <> toFixed4 r.finalLoss
            <> " · " <> toFixed1 r.totalMs <> " ms"
        )
      MicrogptError r -> setText microgptStatusEl ("error: " <> r.err)
      LoadStart r -> setText loadStatusEl ("fetching " <> r.url <> "…")
      LoadConfigMsg _ -> pure unit
      LoadTokenizer r -> setText loadStatusEl
        ("tokenizer ready (vocab " <> show r.vocabSize <> ") · fetching weights…")
      LoadFetched r -> setText loadStatusEl
        ("fetched in " <> toFixed1 r.fetchMs <> " ms · parsing…")
      LoadDone r -> setText loadStatusEl
        ( "loaded " <> show r.tensorCount <> " tensors · "
            <> "fetch+parse total " <> toFixed1 r.totalMs <> " ms"
        )
      LoadError r -> setText loadStatusEl ("load failed: " <> r.err)
  -- Wire the Generate button.
  onClick generateBtn do
    setStyleDisplay benchTable "none"
    promptStr <- getValue promptEl
    maxNewStr <- getValue maxNewEl
    let maxNew = fromMaybe 40 (fromString maxNewStr)
    debug <- hasUrlParamImpl "debug"
    -- Read sampling controls. Defaults are conservative greedy-equivalent:
    --   mode = greedy  → ignores the rest entirely
    --   temperature 0.8, topK 40, topP 0.9 if user picked a sampling mode
    samplingMode <- getValue samplingModeEl
    tempStr <- getValue temperatureEl
    topKStr <- getValue topKEl
    topPStr <- getValue topPEl
    let
      sampling =
        { mode: samplingMode
        , temperature: fromMaybe 0.8 (Number.fromString tempStr)
        , topK: fromMaybe 40 (fromString topKStr)
        , topP: fromMaybe 0.9 (Number.fromString topPStr)
        , seed: 42  -- TODO: expose if reproducibility is needed
        }
    setText outputEl promptStr
    setText statsEl "encoding…"
    postIn worker $ Generate
      { prompt: promptStr, maxNew, debug, sampling }
  -- Preset dropdown: when changed, fill the URL fields with the
  -- preset's stable local paths. Picking "custom" leaves them alone.
  onChange modelPresetEl do
    preset <- getValue modelPresetEl
    case preset of
      "smol" -> do
        setText weightsUrlEl ""  -- clear before re-set so input change events fire
        setText tokenizerUrlEl ""
        setValue weightsUrlEl "/local/smol-llama-101m/model.safetensors"
        setValue tokenizerUrlEl "/local/smol-llama-101m/tokenizer.model"
      "tinyllama" -> do
        setValue weightsUrlEl "/local/tinyllama-1.1b-chat/model.safetensors"
        setValue tokenizerUrlEl "/local/tinyllama-1.1b-chat/tokenizer.model"
      _ -> pure unit
  -- Wire the Load Model button (weights + tokenizer together).
  onClick loadBtn do
    setText loadStatusEl "starting…"
    url <- getValue weightsUrlEl
    tokenizerUrl <- getValue tokenizerUrlEl
    postIn worker $ LoadModel { url, tokenizerUrl }
  -- Wire the Clear Cache button.
  onClick clearCacheBtn do
    setText loadStatusEl "clearing OPFS cache…"
    clearOpfs
      (\n -> setText loadStatusEl ("cache cleared (" <> show n <> " entries)"))
      (\err -> setText loadStatusEl ("cache clear failed: " <> err))
  -- Wire the Train Synthetic button — runs entirely in-process, no
  -- model load required. Posts TrainStart / TrainStep / TrainDone.
  onClick trainBtn do
    setText outputEl ""
    setStyleDisplay benchTable "none"
    stepsStr <- getValue trainStepsEl
    lrStr <- getValue trainLREl
    let
      steps = fromMaybe 60 (fromString stepsStr)
      learningRate = fromMaybe 0.05 (Number.fromString lrStr)
    setText statsEl ("training " <> show steps <> " steps @ lr=" <> lrStr <> "…")
    postIn worker $ TrainSynthetic { steps, learningRate }
  -- Wire the microGPT Train button. Independent of the inference model;
  -- runs the full Karpathy-style pipeline (tokenize → init → train →
  -- sample) inside the worker. Posts MicrogptStart / Step / Sample /
  -- Done / Error.
  onClick microgptTrainBtn do
    setText microgptStatusEl "training…"
    setText microgptStatsEl ""
    setText microgptSamplesEl ""
    corpus <- getValue microgptCorpusEl
    stepsStr <- getValue microgptStepsEl
    lrStr <- getValue microgptLREl
    tempStr <- getValue microgptTempEl
    numSamplesStr <- getValue microgptNumSamplesEl
    maxLenStr <- getValue microgptMaxLenEl
    let
      params =
        { corpus
        , numSteps: fromMaybe 100 (fromString stepsStr)
        , lr: fromMaybe 0.005 (Number.fromString lrStr)
        , temperature: fromMaybe 0.8 (Number.fromString tempStr)
        , numSamples: fromMaybe 5 (fromString numSamplesStr)
        , maxSampleLen: fromMaybe 16 (fromString maxLenStr)
        , seed: 1337
        }
    postIn worker $ MicrogptTrain params
  -- Wire the Benchmark button. Always uses the synthetic model.
  onClick benchmarkBtn do
    setText outputEl ""
    setText statsEl "benchmarking…"
    setHtml benchBody ""
    setStyleDisplay benchTable "table"
    maxNewStr <- getValue maxNewEl
    let maxNew = fromMaybe 10 (fromString maxNewStr)
    postIn worker $ Benchmark
      { benchPrompt: defaultBenchPrompt, maxNew }

-- Helpers --------------------------------------------------------------------

parseTokenList :: String -> Array Int
parseTokenList s =
  case traverse fromString (map trim (split (Pattern ",") s)) of
    Just xs -> xs
    Nothing -> [ 0 ]

formatTokens :: Array Int -> String
formatTokens xs = "[" <> intercalate ", " (map show xs) <> "]"

-- Render per-layer L2 norms as " · per-layer [0.123, 0.456]". Empty
-- when no layers were reported (e.g. a stretch goal where someone
-- runs training on a non-stacked toy model).
formatPerLayer :: Array Number -> String
formatPerLayer xs =
  if Array.null xs then ""
  else " · per-layer [" <> intercalate ", " (map toFixed4 xs) <> "]"

showTps :: Int -> Number -> String
showTps count totalMs =
  if totalMs > 0.0 then
    toFixed1 ((toN count) * 1000.0 / totalMs)
  else "∞"

-- Append a row to the benchmark table.
appendBenchRow
  :: Element
  -> { backend :: String, ok :: Boolean, count :: Int
     , prefillMs :: Number, decodeMs :: Number, totalMs :: Number
     }
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

-- | Set an `<input>` / `<select>` element's `value` property
-- | (preserves event listeners; for `setText` the equivalent is
-- | textContent which is wrong for form fields).
foreign import setValueImpl :: EffectFn2 Element String Unit

setValue :: Element -> String -> Effect Unit
setValue = runEffectFn2 setValueImpl

-- | Wire an "input changed" handler. For `<select>`, `change` fires
-- | when the selected option changes; for text inputs, `input` is
-- | usually preferable. This binding uses `change` because the only
-- | current caller is the model-preset dropdown.
foreign import onChangeImpl :: EffectFn2 Element (Effect Unit) Unit

onChange :: Element -> Effect Unit -> Effect Unit
onChange = runEffectFn2 onChangeImpl

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

-- | Format a Number to four decimal places (for losses / L2 norms).
foreign import toFixed4Impl :: Number -> String

toFixed4 :: Number -> String
toFixed4 = toFixed4Impl

toN :: Int -> Number
toN = toNumberImpl

foreign import toNumberImpl :: Int -> Number
