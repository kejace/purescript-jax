-- | Type-safe message protocol between the browser-thread Browser and
-- | the dedicated Worker. Both sides import this module; the wire
-- | format is a JSON string. The codec values handle ser/de and
-- | give clear failure messages when a field is missing or wrong-typed.
-- |
-- | Wire format: `{ "tag": "Generate", "value": { ... } }` (taggedSum
-- | convention). Compared to the old `{ "kind": "...", ...rest }`
-- | god-object record, this:
-- |   * makes invalid messages fail at the codec boundary, not deep
-- |     inside a handler that did `unsafeCoerce`,
-- |   * enforces exhaustiveness on both sides — adding a constructor
-- |     here is a compile error in any module that pattern-matches it,
-- |   * survives field renames (you change the codec; PS catches the
-- |     mismatch).
module Jax.Worker.Protocol
  ( -- * Inbound (browser → worker)
    WorkerIn(..)
  , SamplingParams
  , workerInCodec
  -- * Outbound (worker → browser)
  , WorkerOut(..)
  , workerOutCodec
  -- * Helpers
  , encodeStr
  , decodeStr
  ) where

import Prelude

import Data.Argonaut.Core (Json, stringify)
import Data.Argonaut.Parser (jsonParser)
import Data.Bifunctor (lmap)
import Data.Codec.Argonaut (JsonCodec, JsonDecodeError(..), printJsonDecodeError)
import Data.Codec.Argonaut as CA
import Data.Codec.Argonaut.Record as CAR
import Data.Codec.Argonaut.Sum as CAS
import Data.Either (Either(..))
import Data.Maybe (Maybe(..))
import Data.Tuple (Tuple(..))

-- =============================================================================
-- Inbound — browser to worker
-- =============================================================================

-- | Sampling-strategy parameters. Carries all four parameter values
-- | regardless of which mode is selected; the worker reads only the
-- | ones relevant to `mode` and ignores the rest. Putting them in a
-- | flat record keeps the codec trivial.
-- |
-- | `mode` ∈ {"greedy", "temperature", "topK", "topP"}.
type SamplingParams =
  { mode :: String
  , temperature :: Number
  , topK :: Int
  , topP :: Number
  , seed :: Int
  }

data WorkerIn
  = LoadModel
      { url :: String
      , tokenizerUrl :: String
      }
  | Generate
      { prompt :: String
      , maxNew :: Int
      , debug :: Boolean
      , sampling :: SamplingParams
      }
  | Benchmark
      { benchPrompt :: Array Int
      , maxNew :: Int
      }
  | TrainSynthetic
      { steps :: Int
      , learningRate :: Number
      }
  | MicrogptTrain
      { corpus :: String
      , numSteps :: Int
      , lr :: Number
      , seed :: Int
      }
  | MicrogptSample
      { temperature :: Number
      , numSamples :: Int
      , maxSampleLen :: Int
      , seed :: Int
      }
  | SetBackend { backend :: String }

derive instance Eq WorkerIn

data WorkerInTag = TagLoadModel | TagGenerate | TagBenchmark | TagTrainSynthetic
                 | TagMicrogptTrain | TagMicrogptSample | TagSetBackend

printIn :: WorkerInTag -> String
printIn = case _ of
  TagLoadModel -> "loadModel"
  TagGenerate -> "generate"
  TagBenchmark -> "benchmark"
  TagTrainSynthetic -> "trainSynthetic"
  TagMicrogptTrain -> "microgptTrain"
  TagMicrogptSample -> "microgptSample"
  TagSetBackend -> "setBackend"

parseIn :: String -> Maybe WorkerInTag
parseIn = case _ of
  "loadModel" -> Just TagLoadModel
  "generate" -> Just TagGenerate
  "benchmark" -> Just TagBenchmark
  "trainSynthetic" -> Just TagTrainSynthetic
  "microgptTrain" -> Just TagMicrogptTrain
  "microgptSample" -> Just TagMicrogptSample
  "setBackend" -> Just TagSetBackend
  _ -> Nothing

loadModelPayload :: JsonCodec { url :: String, tokenizerUrl :: String }
loadModelPayload = CAR.object "LoadModel"
  { url: CA.string
  , tokenizerUrl: CA.string
  }

samplingParamsCodec :: JsonCodec SamplingParams
samplingParamsCodec = CAR.object "SamplingParams"
  { mode: CA.string
  , temperature: CA.number
  , topK: CA.int
  , topP: CA.number
  , seed: CA.int
  }

generatePayload
  :: JsonCodec
       { prompt :: String, maxNew :: Int, debug :: Boolean
       , sampling :: SamplingParams
       }
generatePayload = CAR.object "Generate"
  { prompt: CA.string
  , maxNew: CA.int
  , debug: CA.boolean
  , sampling: samplingParamsCodec
  }

benchmarkPayload :: JsonCodec { benchPrompt :: Array Int, maxNew :: Int }
benchmarkPayload = CAR.object "Benchmark"
  { benchPrompt: CA.array CA.int
  , maxNew: CA.int
  }

trainSyntheticPayload :: JsonCodec { steps :: Int, learningRate :: Number }
trainSyntheticPayload = CAR.object "TrainSynthetic"
  { steps: CA.int
  , learningRate: CA.number
  }

microgptTrainPayload
  :: JsonCodec
       { corpus :: String, numSteps :: Int, lr :: Number, seed :: Int }
microgptTrainPayload = CAR.object "MicrogptTrain"
  { corpus: CA.string, numSteps: CA.int, lr: CA.number, seed: CA.int }

microgptSampleInPayload
  :: JsonCodec
       { temperature :: Number, numSamples :: Int, maxSampleLen :: Int, seed :: Int }
microgptSampleInPayload = CAR.object "MicrogptSample"
  { temperature: CA.number, numSamples: CA.int, maxSampleLen: CA.int, seed: CA.int }

setBackendPayload :: JsonCodec { backend :: String }
setBackendPayload = CAR.object "SetBackend"
  { backend: CA.string
  }

workerInCodec :: JsonCodec WorkerIn
workerInCodec = CAS.taggedSum "WorkerIn" printIn parseIn fromIn toIn
  where
  fromIn = case _ of
    TagLoadModel -> Right (map LoadModel <<< CA.decode loadModelPayload)
    TagGenerate -> Right (map Generate <<< CA.decode generatePayload)
    TagBenchmark -> Right (map Benchmark <<< CA.decode benchmarkPayload)
    TagTrainSynthetic -> Right (map TrainSynthetic <<< CA.decode trainSyntheticPayload)
    TagMicrogptTrain -> Right (map MicrogptTrain <<< CA.decode microgptTrainPayload)
    TagMicrogptSample -> Right (map MicrogptSample <<< CA.decode microgptSampleInPayload)
    TagSetBackend -> Right (map SetBackend <<< CA.decode setBackendPayload)

  toIn = case _ of
    LoadModel r -> Tuple TagLoadModel (Just (CA.encode loadModelPayload r))
    Generate r -> Tuple TagGenerate (Just (CA.encode generatePayload r))
    Benchmark r -> Tuple TagBenchmark (Just (CA.encode benchmarkPayload r))
    TrainSynthetic r -> Tuple TagTrainSynthetic (Just (CA.encode trainSyntheticPayload r))
    MicrogptTrain r -> Tuple TagMicrogptTrain (Just (CA.encode microgptTrainPayload r))
    MicrogptSample r -> Tuple TagMicrogptSample (Just (CA.encode microgptSampleInPayload r))
    SetBackend r -> Tuple TagSetBackend (Just (CA.encode setBackendPayload r))

-- =============================================================================
-- Outbound — worker to browser
-- =============================================================================

data WorkerOut
  = Ready { backend :: String }
  | LoadStart { url :: String }
  | LoadConfigMsg { hidden :: Int, nHeads :: Int, nKvHeads :: Int, nLayers :: Int, vocab :: Int }
  | LoadTokenizer { vocabSize :: Int }
  | LoadFetched { url :: String, fetchMs :: Number }
  | LoadDone
      { url :: String
      , tensorCount :: Int
      , parseMs :: Number
      , adaptMs :: Number
      , totalMs :: Number
      }
  | LoadError { url :: String, err :: String }
  | GenerateStart { promptIds :: Array Int, promptTokens :: Int }
  | GenerateError { err :: String }
  | TokenText { value :: Int, text :: String, isEos :: Boolean }
  | Done
      { count :: Int
      , prefillMs :: Number
      , decodeMs :: Number
      , totalMs :: Number
      , text :: String
      , stoppedAtEos :: Boolean
      }
  | BenchResult
      { backend :: String
      , ok :: Boolean
      , count :: Int
      , prefillMs :: Number
      , decodeMs :: Number
      , totalMs :: Number
      }
  | BenchmarkDone
  | TrainStart
      { paramCount :: Int
      , initialLoss :: Number
      , initialL2 :: Number
      , initialPerLayerL2 :: Array Number
      , steps :: Int
      }
  | TrainStep { step :: Int, loss :: Number }
  | TrainDone
      { finalLoss :: Number
      , finalL2 :: Number
      , finalPerLayerL2 :: Array Number
      , initialGen :: Array Int
      , finalGen :: Array Int
      , totalMs :: Number
      }
  | TrainError { err :: String }
  | MicrogptStart { paramCount :: Int, vocabSize :: Int, numSteps :: Int }
  | MicrogptStep { step :: Int, loss :: Number }
  | MicrogptSampled { index :: Int, text :: String }
  | MicrogptTrainDone { finalLoss :: Number, totalMs :: Number }
  | MicrogptSampleDone { totalMs :: Number }
  | MicrogptError { err :: String }
  | BackendError { tried :: String, err :: String }

derive instance Eq WorkerOut

data WorkerOutTag
  = TagReady | TagLoadStart | TagLoadConfig | TagLoadTokenizer | TagLoadFetched
  | TagLoadDone | TagLoadError | TagGenerateStart | TagGenerateError | TagTokenText
  | TagDone | TagBenchResult | TagBenchmarkDone
  | TagTrainStart | TagTrainStep | TagTrainDone | TagTrainError
  | TagMicrogptStart | TagMicrogptStep | TagMicrogptSampled | TagMicrogptTrainDone
  | TagMicrogptSampleDone | TagMicrogptError
  | TagBackendError

printOut :: WorkerOutTag -> String
printOut = case _ of
  TagReady -> "ready"
  TagLoadStart -> "loadStart"
  TagLoadConfig -> "loadConfig"
  TagLoadTokenizer -> "loadTokenizer"
  TagLoadFetched -> "loadFetched"
  TagLoadDone -> "loadDone"
  TagLoadError -> "loadError"
  TagGenerateStart -> "generateStart"
  TagGenerateError -> "generateError"
  TagTokenText -> "tokenText"
  TagDone -> "done"
  TagBenchResult -> "benchResult"
  TagBenchmarkDone -> "benchmarkDone"
  TagTrainStart -> "trainStart"
  TagTrainStep -> "trainStep"
  TagTrainDone -> "trainDone"
  TagTrainError -> "trainError"
  TagMicrogptStart -> "microgptStart"
  TagMicrogptStep -> "microgptStep"
  TagMicrogptSampled -> "microgptSampled"
  TagMicrogptTrainDone -> "microgptTrainDone"
  TagMicrogptSampleDone -> "microgptSampleDone"
  TagMicrogptError -> "microgptError"
  TagBackendError -> "backendError"

parseOut :: String -> Maybe WorkerOutTag
parseOut = case _ of
  "ready" -> Just TagReady
  "loadStart" -> Just TagLoadStart
  "loadConfig" -> Just TagLoadConfig
  "loadTokenizer" -> Just TagLoadTokenizer
  "loadFetched" -> Just TagLoadFetched
  "loadDone" -> Just TagLoadDone
  "loadError" -> Just TagLoadError
  "generateStart" -> Just TagGenerateStart
  "generateError" -> Just TagGenerateError
  "tokenText" -> Just TagTokenText
  "done" -> Just TagDone
  "benchResult" -> Just TagBenchResult
  "benchmarkDone" -> Just TagBenchmarkDone
  "trainStart" -> Just TagTrainStart
  "trainStep" -> Just TagTrainStep
  "trainDone" -> Just TagTrainDone
  "trainError" -> Just TagTrainError
  "microgptStart" -> Just TagMicrogptStart
  "microgptStep" -> Just TagMicrogptStep
  "microgptSampled" -> Just TagMicrogptSampled
  "microgptTrainDone" -> Just TagMicrogptTrainDone
  "microgptSampleDone" -> Just TagMicrogptSampleDone
  "microgptError" -> Just TagMicrogptError
  "backendError" -> Just TagBackendError
  _ -> Nothing

readyP :: JsonCodec { backend :: String }
readyP = CAR.object "Ready" { backend: CA.string }

loadStartP :: JsonCodec { url :: String }
loadStartP = CAR.object "LoadStart" { url: CA.string }

loadConfigP :: JsonCodec { hidden :: Int, nHeads :: Int, nKvHeads :: Int, nLayers :: Int, vocab :: Int }
loadConfigP = CAR.object "LoadConfig"
  { hidden: CA.int, nHeads: CA.int, nKvHeads: CA.int, nLayers: CA.int, vocab: CA.int }

loadTokenizerP :: JsonCodec { vocabSize :: Int }
loadTokenizerP = CAR.object "LoadTokenizer" { vocabSize: CA.int }

loadFetchedP :: JsonCodec { url :: String, fetchMs :: Number }
loadFetchedP = CAR.object "LoadFetched" { url: CA.string, fetchMs: CA.number }

loadDoneP :: JsonCodec
  { url :: String, tensorCount :: Int, parseMs :: Number, adaptMs :: Number, totalMs :: Number }
loadDoneP = CAR.object "LoadDone"
  { url: CA.string, tensorCount: CA.int, parseMs: CA.number
  , adaptMs: CA.number, totalMs: CA.number }

loadErrorP :: JsonCodec { url :: String, err :: String }
loadErrorP = CAR.object "LoadError" { url: CA.string, err: CA.string }

generateStartP :: JsonCodec { promptIds :: Array Int, promptTokens :: Int }
generateStartP = CAR.object "GenerateStart"
  { promptIds: CA.array CA.int, promptTokens: CA.int }

generateErrorP :: JsonCodec { err :: String }
generateErrorP = CAR.object "GenerateError" { err: CA.string }

tokenTextP :: JsonCodec { value :: Int, text :: String, isEos :: Boolean }
tokenTextP = CAR.object "TokenText"
  { value: CA.int, text: CA.string, isEos: CA.boolean }

doneP :: JsonCodec
  { count :: Int, prefillMs :: Number, decodeMs :: Number, totalMs :: Number
  , text :: String, stoppedAtEos :: Boolean }
doneP = CAR.object "Done"
  { count: CA.int, prefillMs: CA.number, decodeMs: CA.number, totalMs: CA.number
  , text: CA.string, stoppedAtEos: CA.boolean }

benchResultP :: JsonCodec
  { backend :: String, ok :: Boolean, count :: Int
  , prefillMs :: Number, decodeMs :: Number, totalMs :: Number }
benchResultP = CAR.object "BenchResult"
  { backend: CA.string, ok: CA.boolean, count: CA.int
  , prefillMs: CA.number, decodeMs: CA.number, totalMs: CA.number }

trainStartP :: JsonCodec
  { paramCount :: Int, initialLoss :: Number, initialL2 :: Number
  , initialPerLayerL2 :: Array Number, steps :: Int }
trainStartP = CAR.object "TrainStart"
  { paramCount: CA.int, initialLoss: CA.number
  , initialL2: CA.number, initialPerLayerL2: CA.array CA.number
  , steps: CA.int }

trainStepP :: JsonCodec { step :: Int, loss :: Number }
trainStepP = CAR.object "TrainStep" { step: CA.int, loss: CA.number }

trainDoneP :: JsonCodec
  { finalLoss :: Number, finalL2 :: Number, finalPerLayerL2 :: Array Number
  , initialGen :: Array Int, finalGen :: Array Int, totalMs :: Number }
trainDoneP = CAR.object "TrainDone"
  { finalLoss: CA.number, finalL2: CA.number
  , finalPerLayerL2: CA.array CA.number
  , initialGen: CA.array CA.int, finalGen: CA.array CA.int
  , totalMs: CA.number }

trainErrorP :: JsonCodec { err :: String }
trainErrorP = CAR.object "TrainError" { err: CA.string }

microgptStartP :: JsonCodec { paramCount :: Int, vocabSize :: Int, numSteps :: Int }
microgptStartP = CAR.object "MicrogptStart"
  { paramCount: CA.int, vocabSize: CA.int, numSteps: CA.int }

microgptStepP :: JsonCodec { step :: Int, loss :: Number }
microgptStepP = CAR.object "MicrogptStep" { step: CA.int, loss: CA.number }

microgptSampledP :: JsonCodec { index :: Int, text :: String }
microgptSampledP = CAR.object "MicrogptSampled" { index: CA.int, text: CA.string }

microgptTrainDoneP :: JsonCodec { finalLoss :: Number, totalMs :: Number }
microgptTrainDoneP = CAR.object "MicrogptTrainDone"
  { finalLoss: CA.number, totalMs: CA.number }

microgptSampleDoneP :: JsonCodec { totalMs :: Number }
microgptSampleDoneP = CAR.object "MicrogptSampleDone" { totalMs: CA.number }

microgptErrorP :: JsonCodec { err :: String }
microgptErrorP = CAR.object "MicrogptError" { err: CA.string }

backendErrorP :: JsonCodec { tried :: String, err :: String }
backendErrorP = CAR.object "BackendError"
  { tried: CA.string, err: CA.string }

workerOutCodec :: JsonCodec WorkerOut
workerOutCodec = CAS.taggedSum "WorkerOut" printOut parseOut fromOut toOut
  where
  fromOut = case _ of
    TagReady -> Right (map Ready <<< CA.decode readyP)
    TagLoadStart -> Right (map LoadStart <<< CA.decode loadStartP)
    TagLoadConfig -> Right (map LoadConfigMsg <<< CA.decode loadConfigP)
    TagLoadTokenizer -> Right (map LoadTokenizer <<< CA.decode loadTokenizerP)
    TagLoadFetched -> Right (map LoadFetched <<< CA.decode loadFetchedP)
    TagLoadDone -> Right (map LoadDone <<< CA.decode loadDoneP)
    TagLoadError -> Right (map LoadError <<< CA.decode loadErrorP)
    TagGenerateStart -> Right (map GenerateStart <<< CA.decode generateStartP)
    TagGenerateError -> Right (map GenerateError <<< CA.decode generateErrorP)
    TagTokenText -> Right (map TokenText <<< CA.decode tokenTextP)
    TagDone -> Right (map Done <<< CA.decode doneP)
    TagBenchResult -> Right (map BenchResult <<< CA.decode benchResultP)
    TagBenchmarkDone -> Left BenchmarkDone
    TagTrainStart -> Right (map TrainStart <<< CA.decode trainStartP)
    TagTrainStep -> Right (map TrainStep <<< CA.decode trainStepP)
    TagTrainDone -> Right (map TrainDone <<< CA.decode trainDoneP)
    TagTrainError -> Right (map TrainError <<< CA.decode trainErrorP)
    TagMicrogptStart -> Right (map MicrogptStart <<< CA.decode microgptStartP)
    TagMicrogptStep -> Right (map MicrogptStep <<< CA.decode microgptStepP)
    TagMicrogptSampled -> Right (map MicrogptSampled <<< CA.decode microgptSampledP)
    TagMicrogptTrainDone -> Right (map MicrogptTrainDone <<< CA.decode microgptTrainDoneP)
    TagMicrogptSampleDone -> Right (map MicrogptSampleDone <<< CA.decode microgptSampleDoneP)
    TagMicrogptError -> Right (map MicrogptError <<< CA.decode microgptErrorP)
    TagBackendError -> Right (map BackendError <<< CA.decode backendErrorP)

  toOut = case _ of
    Ready r -> Tuple TagReady (Just (CA.encode readyP r))
    LoadStart r -> Tuple TagLoadStart (Just (CA.encode loadStartP r))
    LoadConfigMsg r -> Tuple TagLoadConfig (Just (CA.encode loadConfigP r))
    LoadTokenizer r -> Tuple TagLoadTokenizer (Just (CA.encode loadTokenizerP r))
    LoadFetched r -> Tuple TagLoadFetched (Just (CA.encode loadFetchedP r))
    LoadDone r -> Tuple TagLoadDone (Just (CA.encode loadDoneP r))
    LoadError r -> Tuple TagLoadError (Just (CA.encode loadErrorP r))
    GenerateStart r -> Tuple TagGenerateStart (Just (CA.encode generateStartP r))
    GenerateError r -> Tuple TagGenerateError (Just (CA.encode generateErrorP r))
    TokenText r -> Tuple TagTokenText (Just (CA.encode tokenTextP r))
    Done r -> Tuple TagDone (Just (CA.encode doneP r))
    BenchResult r -> Tuple TagBenchResult (Just (CA.encode benchResultP r))
    BenchmarkDone -> Tuple TagBenchmarkDone Nothing
    TrainStart r -> Tuple TagTrainStart (Just (CA.encode trainStartP r))
    TrainStep r -> Tuple TagTrainStep (Just (CA.encode trainStepP r))
    TrainDone r -> Tuple TagTrainDone (Just (CA.encode trainDoneP r))
    TrainError r -> Tuple TagTrainError (Just (CA.encode trainErrorP r))
    MicrogptStart r -> Tuple TagMicrogptStart (Just (CA.encode microgptStartP r))
    MicrogptStep r -> Tuple TagMicrogptStep (Just (CA.encode microgptStepP r))
    MicrogptSampled r -> Tuple TagMicrogptSampled (Just (CA.encode microgptSampledP r))
    MicrogptTrainDone r -> Tuple TagMicrogptTrainDone (Just (CA.encode microgptTrainDoneP r))
    MicrogptSampleDone r -> Tuple TagMicrogptSampleDone (Just (CA.encode microgptSampleDoneP r))
    MicrogptError r -> Tuple TagMicrogptError (Just (CA.encode microgptErrorP r))
    BackendError r -> Tuple TagBackendError (Just (CA.encode backendErrorP r))

-- =============================================================================
-- String helpers
-- =============================================================================

-- | Encode any codec'd value to a JSON string.
encodeStr :: forall a. JsonCodec a -> a -> String
encodeStr codec a = stringify (CA.encode codec a)

-- | Parse and decode a JSON string. Combines `jsonParser` + the
-- | supplied codec. Errors are reported as human-readable strings.
decodeStr :: forall a. JsonCodec a -> String -> Either String a
decodeStr codec s = do
  json <- jsonParser s
  lmap printJsonDecodeError (CA.decode codec json)
