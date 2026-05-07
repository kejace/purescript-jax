module Worker where

import Prelude

import Control.Monad.Trans.Class (lift)
import Data.Array as Array
import Data.Either (Either(..))
import Data.Foldable (foldl, traverse_)
import Data.Function.Uncurried (Fn3, runFn3)
import Data.Maybe (Maybe(..))
import Data.String (length, take) as String
import Effect (Effect)
import Effect.Aff (attempt, launchAff_)
import Effect.Class (liftEffect)
import Effect.Console (log)
import Effect.Exception (message, throw, try)
import Effect.Ref (Ref)
import Effect.Ref as Ref
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Jax.Coerce (asArray1D, asArray1DInt, asNumber)
import Jax.Core (D1, D2, NDArray, arrayInt1D, dimAt, dispose, linspace, ref, reshape, sliceAxis, toJs, topK)
import Jax.Loaders.Config (parseLlamaConfig, probeRawExtras)
import Jax.Loaders.Fetch (fetchBytes, fetchText)
import Jax.Loaders.LlamaAdapter (loadLlamaWeights)
import Jax.Loaders.Safetensors (parseSafetensors, tensorNames)
import Jax.Loaders.SentencePieceBPE (SentencePieceBPE)
import Jax.Loaders.SentencePieceBPE as SBPE
import Jax.Managed (Managed, allocate, runManaged)
import Jax.NN.Block (LayerWeights, ModelConfig, ModelWeights, emptyKVCacheStack, forwardCachedWithHead)
import Jax.NN.Generate
  ( generateGreedyCached
  , generateGreedyCachedStream
  )
import Jax.NN.Generate as Generate
import Jax.NN.RoPE (RoPETables, precomputeRoPE)
import Jax.NN.Train (makeCrossEntropyLoss)
import Jax.Random as Random
import Jax.Optax as Optax
import Jax.Autodiff (valueAndGradT)
import Jax.Pytree (countParams, countTensors, perLayerL2sq, sumSquaredL2)
import Jax.Demo.Microgpt as Microgpt
import Data.Traversable (traverse)
import Data.Number as Math
import Jax.Worker.Protocol
  ( SamplingParams
  , WorkerIn(..)
  , WorkerOut(..)
  , decodeStr
  , encodeStr
  , workerInCodec
  , workerOutCodec
  )

-- FFI ------------------------------------------------------------------------
--
-- Wire shape: messages are JSON strings on both sides. JS shims do
-- `self.postMessage(str)` / `e.data` (a String). PS-side codecs handle
-- ser/de via `encodeStr` / `decodeStr` from `Jax.Worker.Protocol`.

foreign import selfOnMessageImpl :: EffectFn1 (String -> Effect Unit) Unit
foreign import selfPostMessageImpl :: EffectFn1 String Unit
foreign import performanceNowImpl :: Effect Number
foreign import trySetDeviceImpl :: EffectFn1 String Boolean
foreign import hasWebGpuImpl :: Effect Boolean
foreign import arrayLengthImpl :: forall a. Array a -> Int

selfOnMessage :: (String -> Effect Unit) -> Effect Unit
selfOnMessage = runEffectFn1 selfOnMessageImpl

selfPostMessageRaw :: String -> Effect Unit
selfPostMessageRaw = runEffectFn1 selfPostMessageImpl

-- | Encode a `WorkerOut` and post it. The single point at which the
-- | wire format is decided.
post :: WorkerOut -> Effect Unit
post msg = selfPostMessageRaw (encodeStr workerOutCodec msg)

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
  log "[worker] booting [build: codec-protocol-v6]"
  backend <- selectBackend
  log $ "[worker] backend = " <> backend
  post (Ready { backend })
  modelRef <- Ref.new (Nothing :: Maybe LoadedModel)
  selfOnMessage \raw -> case decodeStr workerInCodec raw of
    Left err -> log $ "[worker] decode error: " <> err
    Right msg -> case msg of
      LoadModel r -> handleLoadModel modelRef r.url r.tokenizerUrl
      Generate r -> handleGenerate modelRef r.prompt r.maxNew r.debug r.sampling
      Benchmark r -> handleBenchmark r.benchPrompt r.maxNew
      TrainSynthetic r -> handleTrainSynthetic r.steps r.learningRate
      MicrogptTrain r -> handleMicrogptTrain r

-- Model loading -------------------------------------------------------------

-- | Load weights + tokenizer + config in a single linear `Aff`
-- | computation. `attempt` catches Aff errors (fetch failures, decoder
-- | errors) and surfaces them through `LoadError`. The previous
-- | callback-style implementation needed 5 levels of nesting and three
-- | separate `onErr` handlers; this is one flat `do`.
handleLoadModel :: Ref (Maybe LoadedModel) -> String -> String -> Effect Unit
handleLoadModel modelRef weightsUrl tokenizerUrl = do
  log $ "[worker] loadModel weights=" <> weightsUrl
    <> " tokenizer=" <> tokenizerUrl
  post (LoadStart { url: weightsUrl })
  let configUrl = configUrlFromWeights weightsUrl
  launchAff_ do
    start <- liftEffect performanceNow
    result <- attempt do
      configJson <- fetchText configUrl
      -- Compatibility check: refuse Phi/Qwen2/etc. before we do
      -- anything irreversible. probeRawExtras is a lightweight
      -- decode that returns Left with an explanation if model_type
      -- isn't in `compatibleArchs`.
      case probeRawExtras configJson of
        Left err -> liftEffect $ throw $ "config compat: " <> err
        Right extras -> liftEffect do
          case extras.sliding_window of
            Just w | w > 0 -> log $ "[worker] note: sliding_window="
              <> show w
              <> " set in config but not enforced; outputs on prompts "
              <> "longer than this may diverge from reference."
            _ -> pure unit
      cfg <- liftEffect $ parseLlamaConfig configJson
      liftEffect $ log $ "[worker] config: hidden=" <> show cfg.hidden
        <> " nHeads=" <> show cfg.nHeads
        <> " nKvHeads=" <> show cfg.nKvHeads
        <> " nLayers=" <> show cfg.nLayers
        <> " vocab=" <> show cfg.vocabSize
      liftEffect $ post $ LoadConfigMsg
        { hidden: cfg.hidden
        , nHeads: cfg.nHeads
        , nKvHeads: cfg.nKvHeads
        , nLayers: cfg.nLayers
        , vocab: cfg.vocabSize
        }
      -- Tokenizer.
      tokBytes <- fetchBytes tokenizerUrl
      tokenizer <- liftEffect $ SBPE.fromBinary tokBytes
      liftEffect $ SBPE.setAddDummyPrefix tokenizer false
      liftEffect $ log $ "[worker] tokenizer loaded · vocabSize="
        <> show (SBPE.vocabSize tokenizer)
      liftEffect $ post (LoadTokenizer { vocabSize: SBPE.vocabSize tokenizer })
      -- Weights.
      bytes <- fetchBytes weightsUrl
      fetched <- liftEffect performanceNow
      liftEffect $ log $ "[worker] fetched safetensors in "
        <> show (fetched - start) <> " ms; parsing"
      liftEffect $ post $ LoadFetched
        { url: weightsUrl, fetchMs: fetched - start }
      -- Parse + adapt are pure Effect; wrap with try so adapter errors
      -- (missing tensor key, dtype mismatch) round-trip out as Aff
      -- error messages too.
      r <- liftEffect $ try do
        parsed <- parseSafetensors bytes
        names <- tensorNames parsed
        parsedAt <- performanceNow
        ckpt <- loadLlamaWeights cfg parsed
        rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
        adaptedAt <- performanceNow
        pure { ckpt, rope, names, parsedAt, adaptedAt }
      case r of
        Left e -> liftEffect $ post $ LoadError
          { url: weightsUrl, err: "parse/adapt: " <> message e }
        Right ok -> liftEffect do
          Ref.write
            ( Just
                { weights: ok.ckpt.weights
                , lmHead: ok.ckpt.lmHead
                , rope: ok.rope
                , cfg
                , tokenizer
                }
            )
            modelRef
          -- Heterogeneous-pytree introspection (Jax.Pytree). Replaces
          -- what would have been an unsafeCoerce-shaped traversal of
          -- ModelWeights with a typeclass-dispatched fold; both
          -- counts are compile-time-checked against the record shape.
          let nTensorsPT = countTensors ok.ckpt.weights
          nParams <- countParams ok.ckpt.weights
          log $ "[worker] adapted "
            <> show (arrayLength ok.names) <> " tensors → ModelWeights"
            <> " (" <> show nTensorsPT <> " in pytree, "
            <> show nParams <> " params) in "
            <> show (ok.adaptedAt - ok.parsedAt) <> " ms"
          post $ LoadDone
            { url: weightsUrl
            , tensorCount: arrayLength ok.names
            , parseMs: ok.parsedAt - fetched
            , adaptMs: ok.adaptedAt - ok.parsedAt
            , totalMs: ok.adaptedAt - start
            }
    case result of
      Left e -> liftEffect $ post $ LoadError
        { url: weightsUrl, err: message e }
      Right _ -> pure unit

-- | Derive the config.json URL from a model.safetensors URL by string
-- | replacement. Works for HuggingFace's `resolve/main/...` paths.
configUrlFromWeights :: String -> String
configUrlFromWeights url = replace "model.safetensors" "config.json" url

foreign import replaceImpl :: Fn3 String String String String

replace :: String -> String -> String -> String
replace search replacement input = runFn3 replaceImpl search replacement input

-- Generate -------------------------------------------------------------------

handleGenerate
  :: Ref (Maybe LoadedModel)
  -> String
  -> Int
  -> Boolean
  -> SamplingParams
  -> Effect Unit
handleGenerate modelRef prompt maxNew debug sampling = do
  loaded <- Ref.read modelRef
  case loaded of
    Nothing ->
      post $ GenerateError
        { err: "no model loaded — click \"load model\" first" }
    Just lm -> generateText lm prompt maxNew debug sampling

generateText
  :: LoadedModel
  -> String
  -> Int
  -> Boolean
  -> SamplingParams
  -> Effect Unit
generateText lm prompt maxNew debug sampling = do
  promptIds <- SBPE.encode lm.tokenizer prompt
  let
    eos = SBPE.eosToken lm.tokenizer
    promptIdsBos = promptIds
  log $ "[worker] generate: prompt=" <> show (String.length prompt)
    <> " chars → " <> show (Array.length promptIdsBos) <> " tokens; eos=" <> show eos
  when debug do
    log $ "[worker] prompt token IDs: " <> show promptIdsBos
    diagPrefillTop5 lm promptIdsBos
  post $ GenerateStart
    { promptIds: promptIdsBos, promptTokens: Array.length promptIdsBos }
  start <- performanceNow
  prefillEndRef <- Ref.new 0.0
  countRef <- Ref.new 0
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
      let
        suffix =
          if String.length fullText >= String.length prev
             && String.take (String.length prev) fullText == prev
            then dropChars (String.length prev) fullText
            else fullText
      Ref.write fullText prevTextRef
      post $ TokenText { value: t, text: suffix, isEos: t == eos }
  -- Build the sampler from the requested mode. Each Sampler-keyed
  -- variant splits an internal PRNG key on every step (see
  -- Generate.purs). Greedy ignores keys.
  sampler <- case sampling.mode of
    "temperature" -> do
      key <- Random.mkKey sampling.seed
      pure (Generate.temperatureSamplerExternal key sampling.temperature)
    "topK" -> do
      key <- Random.mkKey sampling.seed
      pure (Generate.topKSamplerExternal key sampling.topK sampling.temperature)
    "topP" -> do
      key <- Random.mkKey sampling.seed
      pure (Generate.topPSamplerExternal key sampling.topP sampling.temperature)
    _ -> pure Generate.greedySamplerExternal
  let
    stopFn t = t == eos
  _ <- Generate.decodeLoopWithHead
    lm.cfg lm.weights lm.lmHead lm.rope sampler onTok stopFn promptIdsBos maxNew
  pure unit
  end <- performanceNow
  prefillEnd <- Ref.read prefillEndRef
  count <- Ref.read countRef
  newIds <- Ref.read newIdsRef
  finalText <- SBPE.decode lm.tokenizer newIds
  post $ Done
    { count
    , prefillMs: prefillEnd - start
    , decodeMs: end - prefillEnd
    , totalMs: end - start
    , text: finalText
    , stoppedAtEos: case Array.last newIds of
        Just t -> t == eos
        Nothing -> false
    }

-- | Diagnostic: run a fresh prefill on `promptIds`, take the last row
-- | of the logits, top-5 it, and log values + decoded pieces. Helps
-- | localize divergence from a reference implementation.
diagPrefillTop5 :: LoadedModel -> Array Int -> Effect Unit
diagPrefillTop5 lm promptIds = do
  cache0 <- emptyKVCacheStack lm.cfg
  ids <- arrayInt1D promptIds
  { logits } <- forwardCachedWithHead lm.cfg lm.weights lm.lmHead lm.rope cache0 0 ids
  dispose ids
  seqLen <- dimAt logits 0
  vocab <- dimAt logits 1
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

-- | Drop the first `n` chars of a string. Counts UTF-16 code units.
dropChars :: Int -> String -> String
dropChars n s = dropImpl n s

foreign import dropImpl :: Int -> String -> String

-- Benchmark -----------------------------------------------------------------

handleBenchmark :: Array Int -> Int -> Effect Unit
handleBenchmark prompt maxNew = do
  let backends = [ "webgpu", "wasm", "cpu" ]
  traverse_ (benchOne prompt maxNew) backends
  post BenchmarkDone

benchOne :: Array Int -> Int -> String -> Effect Unit
benchOne prompt maxNew backend = do
  ok <- trySetDevice backend
  if not ok then
    post $ BenchResult
      { backend, ok: false, count: 0
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
    post $ BenchResult
      { backend
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

-- Training: synthetic small transformer ------------------------------------

-- | Build a `ModelWeights` directly in Effect (no Managed scope), so
-- | tensors survive across training steps. Uses linspace init like the
-- | tests' `buildVaryingWeights`.
buildSyntheticWeights :: ModelConfig -> Effect ModelWeights
buildSyntheticWeights cfg = do
  emb <- varyingWeight [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2)
  fn <- varyingWeight [ cfg.hidden ] :: Effect (NDArray D1)
  layers <- traverseN' cfg.nLayers \_ -> do
    attnNorm <- varyingWeight [ cfg.hidden ] :: Effect (NDArray D1)
    wq <- varyingWeight [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2)
    wk <- varyingWeight [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2)
    wv <- varyingWeight [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2)
    wo <- varyingWeight [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2)
    mlpNorm <- varyingWeight [ cfg.hidden ] :: Effect (NDArray D1)
    gp <- varyingWeight [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2)
    up <- varyingWeight [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2)
    dp <- varyingWeight [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2)
    pure
      { attnNorm
      , attn: { wq, wk, wv, wo }
      , mlpNorm
      , mlp: { gateProj: gp, upProj: up, downProj: dp }
      }
  pure { embedding: emb, layers, finalNorm: fn }

traverseN' :: forall a. Int -> (Int -> Effect a) -> Effect (Array a)
traverseN' n f = go 0 []
  where
  go i acc
    | i >= n = pure acc
    | otherwise = do
        x <- f i
        go (i + 1) (Array.snoc acc x)

-- | Refcount-bump every leaf in a `ModelWeights`. Used to give the
-- | optimizer its own owned references when initializing — `Optax.initT`
-- | consumes its input.
refModelWeights :: ModelWeights -> Effect ModelWeights
refModelWeights w = do
  emb <- ref w.embedding
  fn <- ref w.finalNorm
  layers <- traverse refLayer w.layers
  pure { embedding: emb, layers, finalNorm: fn }
  where
  refLayer lw = do
    an <- ref lw.attnNorm
    wq <- ref lw.attn.wq
    wk <- ref lw.attn.wk
    wv <- ref lw.attn.wv
    wo <- ref lw.attn.wo
    mn <- ref lw.mlpNorm
    gp <- ref lw.mlp.gateProj
    up <- ref lw.mlp.upProj
    dp <- ref lw.mlp.downProj
    pure
      { attnNorm: an
      , attn: { wq, wk, wv, wo }
      , mlpNorm: mn
      , mlp: { gateProj: gp, upProj: up, downProj: dp }
      }

-- | Run a synthetic-transformer training demo: small ModelWeights,
-- | next-token cross-entropy on a fixed [1,2,3]→[2,3,4] sequence,
-- | Adam over `steps` iterations, posting per-step loss.
-- | Reports initial/final L2 norm via the Jax.Pytree heterogeneous
-- | traversal; reports before/after greedy generation from prompt [1].
handleTrainSynthetic :: Int -> Number -> Effect Unit
handleTrainSynthetic steps lr = do
  let
    cfg :: ModelConfig
    cfg =
      { hidden: 8, nHeads: 2, nKvHeads: 2, headDim: 4
      , intermediate: 16, nLayers: 2, maxSeqLen: 8
      , vocabSize: 8, ropeTheta: 10000.0, normEps: 1.0e-6
      }
  result <- try do
    start <- performanceNow
    weights0 <- buildSyntheticWeights cfg
    rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
    -- Same toy task as the test: predict prompt[i+1] from prompt[:i+1].
    promptIds <- arrayInt1D [ 1, 2, 3 ]
    targetIds <- arrayInt1D [ 2, 3, 4 ]
    let lossFn = makeCrossEntropyLoss cfg rope promptIds targetIds
    vagFn <- valueAndGradT lossFn
    -- Initial diagnostics.
    initVag <- runEffectFn1 vagFn weights0
    initLossF <- toJs initVag.value
    let initialLoss = asNumber initLossF
    dispose initVag.value
    initialL2sq <- sumSquaredL2 weights0
    initialPerLayerSq <- perLayerL2sq cfg.nLayers weights0
    let initialPerLayer = map Math.sqrt initialPerLayerSq
    nParams <- countParams weights0
    -- Initial generation (greedy from [1]).
    initialGen <- generateGreedyCached cfg weights0 rope [ 1 ] 3
    post $ TrainStart
      { paramCount: nParams
      , initialLoss
      , initialL2: Math.sqrt initialL2sq
      , initialPerLayerL2: initialPerLayer
      , steps
      }
    -- Optimizer init (consumes weights → use a ref-bumped copy).
    opt <- Optax.adam lr
    weightsForInit <- refModelWeights weights0
    state0 <- Optax.initT opt weightsForInit
    finalState <- trainLoop cfg rope vagFn opt steps 0
      { weights: weights0, state: state0, grad: initVag.grad }
    -- Final diagnostics.
    finalVag <- runEffectFn1 vagFn finalState.weights
    finalLossF <- toJs finalVag.value
    let finalLoss = asNumber finalLossF
    dispose finalVag.value
    finalL2sq <- sumSquaredL2 finalState.weights
    finalPerLayerSq <- perLayerL2sq cfg.nLayers finalState.weights
    let finalPerLayer = map Math.sqrt finalPerLayerSq
    finalGen <- generateGreedyCached cfg finalState.weights rope [ 1 ] 3
    end <- performanceNow
    pure
      { finalLoss
      , finalL2: Math.sqrt finalL2sq
      , finalPerLayerL2: finalPerLayer
      , initialGen
      , finalGen
      , totalMs: end - start
      }
  case result of
    Left e -> post $ TrainError { err: message e }
    Right r -> post $ TrainDone
      { finalLoss: r.finalLoss
      , finalL2: r.finalL2
      , finalPerLayerL2: r.finalPerLayerL2
      , initialGen: r.initialGen
      , finalGen: r.finalGen
      , totalMs: r.totalMs
      }

-- | One iteration of the Adam loop. Posts a TrainStep after each.
trainLoop
  :: forall e
   . ModelConfig
  -> RoPETables
  -> EffectFn1 ModelWeights { value :: NDArray e, grad :: ModelWeights }
  -> Optax.Transformation
  -> Int
  -> Int
  -> { weights :: ModelWeights, state :: Optax.OptState, grad :: ModelWeights }
  -> Effect { weights :: ModelWeights, state :: Optax.OptState }
trainLoop _ _ _ _ 0 _ acc = pure { weights: acc.weights, state: acc.state }
trainLoop cfg rope vagFn opt nLeft stepIdx acc = do
  { updates, state: newState } <- Optax.updateT opt acc.grad acc.state
  newWeights <- Optax.applyUpdatesT acc.weights updates
  vag <- runEffectFn1 vagFn newWeights
  lossF <- toJs vag.value
  let stepLoss = asNumber lossF
  dispose vag.value
  post $ TrainStep { step: stepIdx + 1, loss: stepLoss }
  trainLoop cfg rope vagFn opt (nLeft - 1) (stepIdx + 1)
    { weights: newWeights, state: newState, grad: vag.grad }

-- | In-browser microGPT training. Same pipeline as the CLI demo —
-- | same `Jax.Demo.Microgpt.runMicrogpt` — but the callbacks post
-- | typed protocol messages instead of writing to console. The UI
-- | wires those into a live loss feed + per-sample text.
handleMicrogptTrain
  :: { corpus :: String, numSteps :: Int, lr :: Number
     , temperature :: Number, numSamples :: Int, maxSampleLen :: Int
     , seed :: Int }
  -> Effect Unit
handleMicrogptTrain r = do
  start <- performanceNow
  result <- try do
    let
      defs = Microgpt.defaultParams
      params = defs
        { corpus = r.corpus
        , numSteps = r.numSteps
        , lr = r.lr
        , temperature = r.temperature
        , numSamples = r.numSamples
        , maxSampleLen = r.maxSampleLen
        , seed = r.seed
        }
    Microgpt.runMicrogpt params
      { onStart: \s -> post $ MicrogptStart
          { paramCount: s.paramCount, vocabSize: s.vocabSize, numSteps: r.numSteps }
      , onProgress: \i loss -> post $ MicrogptStep { step: i, loss }
      , onSampled: \i text -> post $ MicrogptSample { index: i, text }
      , onDone: \finalLoss -> do
          end <- performanceNow
          post $ MicrogptDone { finalLoss, totalMs: end - start }
      }
  case result of
    Left e -> post $ MicrogptError { err: message e }
    Right _ -> pure unit
