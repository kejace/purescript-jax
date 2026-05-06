module Worker where

import Prelude

import Control.Monad.Trans.Class (lift)
import Data.Array as Array
import Data.Foldable (foldl, traverse_)
import Data.Either (Either(..))
import Data.Maybe (Maybe(..))
import Data.String (length, take) as String
import Effect (Effect)
import Effect.Console (log)
import Effect.Exception (message, try)
import Effect.Ref (Ref)
import Effect.Ref as Ref
import Data.Function.Uncurried (Fn3, runFn3)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Foreign (Foreign)
import Jax.Coerce (asArray1D, asArray1DInt)
import Jax.Core (D1, D2, NDArray, arrayInt1D, dispose, linspace, ref, reshape, shape, sliceAxis, toJs, topK)
import Jax.NN.Block (emptyKVCacheStack, forwardCachedWithHead)
import Jax.Loaders.Config (parseLlamaConfig)
import Jax.Loaders.Fetch (fetchBytes, fetchText)
import Jax.Loaders.LlamaAdapter (loadLlamaWeights)
import Jax.Loaders.Safetensors (parseSafetensors, tensorNames)
import Jax.Loaders.SentencePieceBPE (SentencePieceBPE)
import Jax.Loaders.SentencePieceBPE as SBPE
import Jax.Managed (Managed, allocate, runManaged)
import Jax.NN.Block (LayerWeights, ModelConfig, ModelWeights)
import Jax.NN.Generate
  ( generateGreedyCachedStream
  , generateGreedyCachedStreamUntilWithHead
  )
import Jax.NN.RoPE (RoPETables, precomputeRoPE)
import Unsafe.Coerce (unsafeCoerce)

-- FFI ------------------------------------------------------------------------

foreign import selfOnMessageImpl :: EffectFn1 (Foreign -> Effect Unit) Unit
foreign import selfPostMessageImpl :: forall a. EffectFn1 a Unit
foreign import performanceNowImpl :: Effect Number
foreign import trySetDeviceImpl :: EffectFn1 String Boolean
foreign import hasWebGpuImpl :: Effect Boolean
foreign import arrayLengthImpl :: forall a. Array a -> Int

selfOnMessage :: (Foreign -> Effect Unit) -> Effect Unit
selfOnMessage = runEffectFn1 selfOnMessageImpl

selfPostMessage :: forall a. a -> Effect Unit
selfPostMessage = runEffectFn1 selfPostMessageImpl

performanceNow :: Effect Number
performanceNow = performanceNowImpl

trySetDevice :: String -> Effect Boolean
trySetDevice = runEffectFn1 trySetDeviceImpl

hasWebGpu :: Effect Boolean
hasWebGpu = hasWebGpuImpl

arrayLength :: forall a. Array a -> Int
arrayLength = arrayLengthImpl

-- Backend ladder -------------------------------------------------------------

selectBackend :: Effect String
selectBackend = do
  hasGpu <- hasWebGpu
  if hasGpu then do
    ok <- trySetDevice "webgpu"
    if ok then pure "webgpu" else fallbackWasm
  else fallbackWasm
  where
  fallbackWasm = do
    ok <- trySetDevice "wasm"
    if ok then pure "wasm"
    else do
      _ <- trySetDevice "cpu"
      pure "cpu"

-- Loaded model state --------------------------------------------------------

type LoadedModel =
  { weights :: ModelWeights
  , lmHead :: NDArray D2
  , rope :: RoPETables
  , cfg :: ModelConfig
  , tokenizer :: SentencePieceBPE
  }

-- Worker entry ---------------------------------------------------------------

main :: Effect Unit
main = do
  log "[worker] booting [build: gqa-fix-v5-debugflag]"
  backend <- selectBackend
  log $ "[worker] backend = " <> backend
  selfPostMessage { kind: "ready", backend }
  modelRef <- Ref.new (Nothing :: Maybe LoadedModel)
  selfOnMessage \msg -> do
    let
      m = unsafeCoerce msg ::
        { kind :: String
        , prompt :: String
        , maxNew :: Int
        , debug :: Boolean
        , url :: String
        , tokenizerUrl :: String
        , benchPrompt :: Array Int
        }
    case m.kind of
      "generate" -> handleGenerate modelRef m.prompt m.maxNew m.debug
      "benchmark" -> handleBenchmark m.benchPrompt m.maxNew
      "loadModel" -> handleLoadModel modelRef m.url m.tokenizerUrl
      _ -> log $ "[worker] unknown message kind: " <> m.kind

-- Model loading -------------------------------------------------------------

handleLoadModel :: Ref (Maybe LoadedModel) -> String -> String -> Effect Unit
handleLoadModel modelRef weightsUrl tokenizerUrl = do
  log $ "[worker] loadModel weights=" <> weightsUrl
    <> " tokenizer=" <> tokenizerUrl
  selfPostMessage { kind: "loadStart", url: weightsUrl }
  start <- performanceNow
  let configUrl = configUrlFromWeights weightsUrl
  -- Step 1: fetch config.json
  fetchText configUrl
    ( \configJson -> do
        cfg <- parseLlamaConfig configJson
        log $ "[worker] config: hidden=" <> show cfg.hidden
          <> " nHeads=" <> show cfg.nHeads
          <> " nKvHeads=" <> show cfg.nKvHeads
          <> " nLayers=" <> show cfg.nLayers
          <> " vocab=" <> show cfg.vocabSize
        selfPostMessage { kind: "loadConfig", cfg }
        -- Step 2: fetch tokenizer.model
        fetchBytes tokenizerUrl
          ( \tokBytes -> do
              tokenizer <- SBPE.fromBinary tokBytes
              -- HF Llama (non-legacy) tokenizer: skip the dummy prefix.
              -- The trained embedding rows expect e.g. "Once" (id 26222)
              -- not "▁Once" (id 9038) for the first word — feeding the
              -- ▁-prefixed variant gives the model unseen IDs and
              -- garbage logits.
              SBPE.setAddDummyPrefix tokenizer false
              log $ "[worker] tokenizer loaded · vocabSize="
                <> show (SBPE.vocabSize tokenizer)
              selfPostMessage
                { kind: "loadTokenizer"
                , vocabSize: SBPE.vocabSize tokenizer
                }
              -- Step 3: fetch model.safetensors
              fetchBytes weightsUrl
                ( \bytes -> do
                    fetched <- performanceNow
                    log $ "[worker] fetched safetensors in "
                      <> show (fetched - start) <> " ms; parsing"
                    selfPostMessage
                      { kind: "loadFetched"
                      , url: weightsUrl
                      , fetchMs: fetched - start
                      }
                    -- Wrap parse + adapt in try/catch so synchronous
                    -- errors (e.g. missing tensor key, header parse fail)
                    -- surface as loadError instead of bubbling up to the
                    -- worker's onerror.
                    result <- try do
                      parsed <- parseSafetensors bytes
                      names <- tensorNames parsed
                      parsedAt <- performanceNow
                      ckpt <- loadLlamaWeights cfg parsed
                      rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
                      adaptedAt <- performanceNow
                      pure
                        { ckpt, rope, names, parsedAt, adaptedAt }
                    case result of
                      Left e -> do
                        let err = message e
                        log $ "[worker] parse/adapt failed: " <> err
                        selfPostMessage
                          { kind: "loadError"
                          , url: weightsUrl
                          , err: "parse/adapt: " <> err
                          }
                      Right r -> do
                        Ref.write
                          ( Just
                              { weights: r.ckpt.weights
                              , lmHead: r.ckpt.lmHead
                              , rope: r.rope
                              , cfg
                              , tokenizer
                              }
                          )
                          modelRef
                        log $ "[worker] adapted "
                          <> show (arrayLength r.names) <> " tensors → ModelWeights in "
                          <> show (r.adaptedAt - r.parsedAt) <> " ms"
                        selfPostMessage
                          { kind: "loadDone"
                          , url: weightsUrl
                          , tensorCount: arrayLength r.names
                          , parseMs: r.parsedAt - fetched
                          , adaptMs: r.adaptedAt - r.parsedAt
                          , totalMs: r.adaptedAt - start
                          }
                )
                ( \err -> do
                    log $ "[worker] safetensors fetch failed: " <> err
                    selfPostMessage
                      { kind: "loadError", url: weightsUrl, err }
                )
          )
          ( \err -> do
              log $ "[worker] tokenizer.model fetch failed: " <> err
              selfPostMessage
                { kind: "loadError"
                , url: tokenizerUrl
                , err: "tokenizer.model: " <> err
                }
          )
    )
    ( \err -> do
        log $ "[worker] config.json fetch failed: " <> err
        selfPostMessage
          { kind: "loadError"
          , url: configUrl
          , err: "config.json: " <> err
          }
    )

-- | Derive the config.json URL from a model.safetensors URL by string
-- | replacement. Works for HuggingFace's `resolve/main/...` paths.
configUrlFromWeights :: String -> String
configUrlFromWeights url = replace "model.safetensors" "config.json" url

foreign import replaceImpl :: Fn3 String String String String

replace :: String -> String -> String -> String
replace search replacement input = runFn3 replaceImpl search replacement input

-- Generate -------------------------------------------------------------------

handleGenerate
  :: Ref (Maybe LoadedModel) -> String -> Int -> Boolean -> Effect Unit
handleGenerate modelRef prompt maxNew debug = do
  loaded <- Ref.read modelRef
  case loaded of
    Nothing ->
      selfPostMessage
        { kind: "generateError"
        , err: "no model loaded — click \"load model\" first"
        }
    Just lm -> generateText lm prompt maxNew debug

generateText :: LoadedModel -> String -> Int -> Boolean -> Effect Unit
generateText lm prompt maxNew debug = do
  promptIds <- SBPE.encode lm.tokenizer prompt
  -- Smol-Llama-101M-Chat-v1's tokenizer config has add_bos_token=False,
  -- and HF generation respects this — no BOS prepended. Our previous
  -- behaviour (`Array.cons bos promptIds`) made the model see an
  -- extra leading token it didn't expect.
  let
    eos = SBPE.eosToken lm.tokenizer
    promptIdsBos = promptIds
  log $ "[worker] generate: prompt=" <> show (String.length prompt)
    <> " chars → " <> show (Array.length promptIdsBos) <> " tokens; eos=" <> show eos
  -- Dev-only: gated by `?debug` URL param on the page. Logs prompt IDs
  -- and a top-5 next-token diagnostic prefill — useful for comparing
  -- against a reference (e.g., HF transformers) when bringing up a
  -- new model architecture.
  when debug do
    log $ "[worker] prompt token IDs: " <> show promptIdsBos
    diagPrefillTop5 lm promptIdsBos
  selfPostMessage
    { kind: "generateStart"
    , promptIds: promptIdsBos
    , promptTokens: Array.length promptIdsBos
    }
  start <- performanceNow
  prefillEndRef <- Ref.new 0.0
  countRef <- Ref.new 0
  -- Streaming-decode strategy: keep the full id list (prompt + generated)
  -- and re-decode it each step, posting the *delta* text vs the previous
  -- step. This gives us two important things:
  --   1. Byte-fallback safety — half a multi-byte char emits no visible
  --      delta until its tail bytes arrive.
  --   2. The first generated token's leading ▁ becomes a real space in
  --      the delta (instead of being eaten by the dummy-prefix strip in
  --      `decode`, which only fires for the very first character).
  -- Seed the buffer with the BOS-prepended prompt so the baseline is the
  -- prompt's decoded text.
  promptText <- SBPE.decode lm.tokenizer promptIdsBos
  newIdsRef <- Ref.new promptIdsBos
  prevTextRef <- Ref.new promptText
  let
    onTok t = do
      n <- Ref.read countRef
      when (n == 0) do
        now <- performanceNow
        Ref.write now prefillEndRef
      Ref.modify_ (_ + 1) countRef
      Ref.modify_ (\xs -> Array.snoc xs t) newIdsRef
      ids <- Ref.read newIdsRef
      fullText <- SBPE.decode lm.tokenizer ids
      prev <- Ref.read prevTextRef
      -- Streaming-safe delta: prev is always a prefix of fullText for
      -- well-formed UTF-8 streams (because we re-decode the cumulative
      -- id list each step). The byte-fallback case is exactly why we
      -- need this: half a multi-byte char emits no visible delta until
      -- its tail bytes arrive.
      let
        suffix =
          if String.length fullText >= String.length prev
             && String.take (String.length prev) fullText == prev
            then dropChars (String.length prev) fullText
            else fullText
      Ref.write fullText prevTextRef
      selfPostMessage
        { kind: "tokenText"
        , value: t
        , text: suffix
        , isEos: t == eos
        }
  generateGreedyCachedStreamUntilWithHead
    lm.cfg lm.weights lm.lmHead lm.rope (Just eos) promptIdsBos maxNew onTok
  end <- performanceNow
  prefillEnd <- Ref.read prefillEndRef
  count <- Ref.read countRef
  newIds <- Ref.read newIdsRef
  finalText <- SBPE.decode lm.tokenizer newIds
  selfPostMessage
    { kind: "done"
    , count
    , prefillMs: prefillEnd - start
    , decodeMs: end - prefillEnd
    , totalMs: end - start
    , text: finalText
    , stoppedAtEos: case Array.last newIds of
        Just t -> t == eos
        Nothing -> false
    }

-- | Diagnostic: run a fresh prefill on `promptIdsBos`, take the last
-- | row of the logits, top-5 it, and log values + decoded pieces.
-- | Helps localize divergence from a reference implementation.
diagPrefillTop5 :: LoadedModel -> Array Int -> Effect Unit
diagPrefillTop5 lm promptIds = do
  cache0 <- emptyKVCacheStack lm.cfg
  ids <- arrayInt1D promptIds
  { logits } <- forwardCachedWithHead lm.cfg lm.weights lm.lmHead lm.rope cache0 0 ids
  dispose ids
  sh <- shape logits
  let
    seqLen = case Array.head sh of
      Just n -> n
      Nothing -> 0
    vocab = case sh Array.!! 1 of
      Just n -> n
      Nothing -> 0
  logitsR <- ref logits
  lastRow <- sliceAxis logitsR 0 (seqLen - 1) seqLen
  dispose logits
  lastRowR <- ref lastRow
  flat <- reshape lastRowR [ vocab ] :: Effect (NDArray D1)
  dispose lastRow
  flatR <- ref flat
  { values: topVals, indices: topIds } <- topK flatR 5 0
  vF <- toJs topVals
  iF <- toJs topIds
  dispose flat
  dispose topVals
  dispose topIds
  let
    vs = asArray1D vF
    is = asArray1DInt iF
  log "[worker] PREFILL top-5 next-token logits:"
  traverseN_ 5 \k ->
    case Array.index is k, Array.index vs k of
      Just i, Just v -> log $ "  id=" <> show i <> "  logit=" <> show v
      _, _ -> pure unit

traverseN_ :: Int -> (Int -> Effect Unit) -> Effect Unit
traverseN_ n f = go 0
  where
  go i
    | i >= n = pure unit
    | otherwise = f i *> go (i + 1)

-- | Drop the first `n` chars of a string. PureScript's `Data.String`
-- | provides `drop` but counts code units; that's fine for our use
-- | because both `prev` and `fullText` decoder output share the same
-- | UTF-16 prefix.
dropChars :: Int -> String -> String
dropChars n s = dropImpl n s

foreign import dropImpl :: Int -> String -> String

-- Benchmark -----------------------------------------------------------------

handleBenchmark :: Array Int -> Int -> Effect Unit
handleBenchmark prompt maxNew = do
  let backends = [ "webgpu", "wasm", "cpu" ]
  traverse_ (benchOne prompt maxNew) backends
  selfPostMessage { kind: "benchmarkDone" }

benchOne :: Array Int -> Int -> String -> Effect Unit
benchOne prompt maxNew backend = do
  ok <- trySetDevice backend
  if not ok then
    selfPostMessage
      { kind: "benchResult", backend, ok: false, count: 0
      , prefillMs: 0.0, decodeMs: 0.0, totalMs: 0.0
      }
  else do
    let
      cfg :: ModelConfig
      cfg =
        { hidden: 8, nHeads: 2, nKvHeads: 2, headDim: 4
        , intermediate: 16, nLayers: 2, maxSeqLen: 64
        , vocabSize: 5, ropeTheta: 10000.0, normEps: 1.0e-6
        }
    start <- performanceNow
    prefillEndRef <- Ref.new 0.0
    countRef <- Ref.new 0
    runManaged (buildSyntheticModel cfg) \{ weights, rope } -> do
      let
        onTok _ = do
          n <- Ref.read countRef
          when (n == 0) do
            now <- performanceNow
            Ref.write now prefillEndRef
          Ref.modify_ (_ + 1) countRef
      generateGreedyCachedStream cfg weights rope prompt maxNew onTok
    end <- performanceNow
    prefillEnd <- Ref.read prefillEndRef
    count <- Ref.read countRef
    selfPostMessage
      { kind: "benchResult"
      , backend
      , ok: true
      , count
      , prefillMs: prefillEnd - start
      , decodeMs: end - prefillEnd
      , totalMs: end - start
      }

-- Synthetic model (only used by the benchmark sweep) ------------------------

varyingWeight :: forall d. Array Int -> Effect (NDArray d)
varyingWeight sh = do
  let n = foldl (*) 1 sh
  base <- linspace (-0.1) 0.1 n
  baseR <- ref base
  reshape baseR sh

buildSyntheticModel
  :: ModelConfig
  -> Managed { weights :: ModelWeights, rope :: RoPETables }
buildSyntheticModel cfg = do
  emb <- allocate (varyingWeight [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2))
  layer0 <- buildSyntheticLayer cfg
  layer1 <- buildSyntheticLayer cfg
  fn <- allocate (varyingWeight [ cfg.hidden ] :: Effect (NDArray D1))
  rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
  cosT <- allocate (pure rope.cos)
  sinT <- allocate (pure rope.sin)
  let
    weights = { embedding: emb, layers: [ layer0, layer1 ], finalNorm: fn }
    ropeTables = { cos: cosT, sin: sinT }
  pure { weights, rope: ropeTables }

buildSyntheticLayer :: ModelConfig -> Managed LayerWeights
buildSyntheticLayer cfg = do
  attnNorm <- allocate (varyingWeight [ cfg.hidden ] :: Effect (NDArray D1))
  wq <- allocate (varyingWeight [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
  wk <- allocate (varyingWeight [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
  wv <- allocate (varyingWeight [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
  wo <- allocate (varyingWeight [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
  mlpNorm <- allocate (varyingWeight [ cfg.hidden ] :: Effect (NDArray D1))
  gp <- allocate (varyingWeight [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
  up <- allocate (varyingWeight [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
  dp <- allocate (varyingWeight [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
  pure
    { attnNorm
    , attn: { wq, wk, wv, wo }
    , mlpNorm
    , mlp: { gateProj: gp, upProj: up, downProj: dp }
    }
