module Jax.NN.Generate
  ( generateGreedy
  , generateGreedyCached
  , generateTemperature
  , generateTopK
  , generateTopP
  , generateGreedyCachedStream
  , generateGreedyCachedStreamUntil
  , generateGreedyCachedStreamUntilWithHead
  ) where

import Prelude hiding (add)

import Data.Array (last, length, snoc, head, (!!)) as Array
import Data.Foldable (traverse_)
import Data.Maybe (Maybe(..), fromMaybe)
import Effect (Effect)
import Jax.Core
  ( D1
  , D2
  , NDArray
  , arrayInt1D
  , dispose
  , ref
  , reshape
  , shape
  , sliceAxis
  )
import Jax.NN.Attention (KVCache, KVCacheStack)
import Jax.NN.Block
  ( ModelConfig
  , ModelWeights
  , emptyKVCacheStack
  , forwardCached
  , forwardCachedWithHead
  , forwardLogits
  )
import Jax.NN.RoPE (RoPETables)
import Jax.NN.Sampling (sampleGreedy, sampleTemperature, sampleTopK, sampleTopP)
import Jax.Random (Key, splitKey2)

-- | Greedy autoregressive generation. Naive — recomputes the full forward
-- | pass every step (O(n²) total).
generateGreedy
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateGreedy cfg weights rope prompt maxNew = go prompt maxNew
  where
  go ctx 0 = pure ctx
  go ctx n = do
    ids <- arrayInt1D ctx
    logits <- forwardLogits cfg weights rope ids
    dispose ids
    nextId <- sampleLastRow logits
    go (Array.snoc ctx nextId) (n - 1)

-- | KVCache-accelerated greedy generation. Prefill once, then each decode
-- | step processes only the single new token against the cached K/V —
-- | O(n) decode rather than the O(n²) of `generateGreedy`.
generateGreedyCached
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateGreedyCached cfg weights rope prompt maxNew = do
  cache0 <- emptyKVCacheStack cfg
  -- Prefill on the prompt.
  promptIds <- arrayInt1D prompt
  { newCache: cache1, logits: l1 } <-
    forwardCached cfg weights rope cache0 0 promptIds
  dispose promptIds
  firstNew <- sampleLastRow l1
  let
    startPos = Array.length prompt
    ctx0 = Array.snoc prompt firstNew
  -- Decode loop: one new token per step.
  final <- iterDecode (maxNew - 1) (startPos + 1) ctx0 cache1
  -- Dispose final cache (the cache lives across all decode steps).
  traverse_ disposeKVCache final.cache
  pure final.ctx
  where
  iterDecode
    :: Int
    -> Int
    -> Array Int
    -> KVCacheStack
    -> Effect { ctx :: Array Int, cache :: KVCacheStack }
  iterDecode 0 _ ctx cache = pure { ctx, cache }
  iterDecode n pos ctx cache = do
    let lastTok = case Array.last ctx of
          Just x -> x
          Nothing -> 0
    tokIds <- arrayInt1D [ lastTok ]
    { newCache, logits } <- forwardCached cfg weights rope cache pos tokIds
    dispose tokIds
    nextTok <- sampleLastRow logits
    iterDecode (n - 1) (pos + 1) (Array.snoc ctx nextTok) newCache

-- | Slice the last row of [seq, vocab] logits, sample greedy, dispose.
-- | Consumes the input logits tensor.
sampleLastRow :: NDArray D2 -> Effect Int
sampleLastRow logits = do
  sh <- shape logits
  let
    seqLen = fromMaybe 0 (Array.head sh)
    vocab = fromMaybe 0 (sh Array.!! 1)
  logitsR <- ref logits
  lastRow <- sliceAxis logitsR 0 (seqLen - 1) seqLen
  dispose logits
  lastRowR <- ref lastRow
  flat <- reshape lastRowR [ vocab ] :: Effect (NDArray D1)
  dispose lastRow
  idx <- sampleGreedy flat
  dispose flat
  pure idx

disposeKVCache :: KVCache -> Effect Unit
disposeKVCache kv = do
  dispose kv.k
  dispose kv.v

-- | KVCache-accelerated temperature-sampled generation. Threads a PRNG
-- | key through the decode loop, splitting it at each step.
generateTemperature
  :: Key
  -> Number          -- ^ temperature (>0; smaller = sharper)
  -> ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateTemperature initialKey temperature cfg weights rope prompt maxNew = do
  cache0 <- emptyKVCacheStack cfg
  promptIds <- arrayInt1D prompt
  { newCache: cache1, logits: l1 } <-
    forwardCached cfg weights rope cache0 0 promptIds
  dispose promptIds
  -- Split the initial key: one for sampling, one to thread forward.
  ks <- splitKey2 initialKey
  firstNew <- sampleLastRowTemp ks.a temperature l1
  let
    startPos = Array.length prompt
    ctx0 = Array.snoc prompt firstNew
  final <- iterDecode ks.b temperature (maxNew - 1) (startPos + 1) ctx0 cache1
  traverse_ disposeKVCache final.cache
  pure final.ctx
  where
  iterDecode
    :: Key
    -> Number
    -> Int
    -> Int
    -> Array Int
    -> KVCacheStack
    -> Effect { ctx :: Array Int, cache :: KVCacheStack }
  iterDecode _ _ 0 _ ctx cache = pure { ctx, cache }
  iterDecode key temp n pos ctx cache = do
    let lastTok = case Array.last ctx of
          Just x -> x
          Nothing -> 0
    tokIds <- arrayInt1D [ lastTok ]
    { newCache, logits } <- forwardCached cfg weights rope cache pos tokIds
    dispose tokIds
    ks <- splitKey2 key
    nextTok <- sampleLastRowTemp ks.a temp logits
    iterDecode ks.b temp (n - 1) (pos + 1) (Array.snoc ctx nextTok) newCache

-- | Slice last row → temperature-sample. Consumes `key` and `logits`.
sampleLastRowTemp :: Key -> Number -> NDArray D2 -> Effect Int
sampleLastRowTemp key temp logits = do
  sh <- shape logits
  let
    seqLen = fromMaybe 0 (Array.head sh)
    vocab = fromMaybe 0 (sh Array.!! 1)
  logitsR <- ref logits
  lastRow <- sliceAxis logitsR 0 (seqLen - 1) seqLen
  dispose logits
  lastRowR <- ref lastRow
  flat <- reshape lastRowR [ vocab ] :: Effect (NDArray D1)
  dispose lastRow
  idx <- sampleTemperature key temp flat
  dispose flat
  pure idx

-- | Slice last row → top-k-temperature-sample. Consumes `key` and `logits`.
sampleLastRowTopK :: Key -> Int -> Number -> NDArray D2 -> Effect Int
sampleLastRowTopK key k temp logits = do
  sh <- shape logits
  let
    seqLen = fromMaybe 0 (Array.head sh)
    vocab = fromMaybe 0 (sh Array.!! 1)
  logitsR <- ref logits
  lastRow <- sliceAxis logitsR 0 (seqLen - 1) seqLen
  dispose logits
  lastRowR <- ref lastRow
  flat <- reshape lastRowR [ vocab ] :: Effect (NDArray D1)
  dispose lastRow
  idx <- sampleTopK key k temp flat
  dispose flat
  pure idx

-- | KVCache-accelerated top-k temperature sampling.
generateTopK
  :: Key
  -> Int           -- ^ k
  -> Number        -- ^ temperature
  -> ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateTopK initialKey k temperature cfg weights rope prompt maxNew = do
  cache0 <- emptyKVCacheStack cfg
  promptIds <- arrayInt1D prompt
  { newCache: cache1, logits: l1 } <-
    forwardCached cfg weights rope cache0 0 promptIds
  dispose promptIds
  ks <- splitKey2 initialKey
  firstNew <- sampleLastRowTopK ks.a k temperature l1
  let
    startPos = Array.length prompt
    ctx0 = Array.snoc prompt firstNew
  final <- iterDecode ks.b k temperature (maxNew - 1) (startPos + 1) ctx0 cache1
  traverse_ disposeKVCache final.cache
  pure final.ctx
  where
  iterDecode
    :: Key
    -> Int
    -> Number
    -> Int
    -> Int
    -> Array Int
    -> KVCacheStack
    -> Effect { ctx :: Array Int, cache :: KVCacheStack }
  iterDecode _ _ _ 0 _ ctx cache = pure { ctx, cache }
  iterDecode key kk temp n pos ctx cache = do
    let lastTok = case Array.last ctx of
          Just x -> x
          Nothing -> 0
    tokIds <- arrayInt1D [ lastTok ]
    { newCache, logits } <- forwardCached cfg weights rope cache pos tokIds
    dispose tokIds
    ks <- splitKey2 key
    nextTok <- sampleLastRowTopK ks.a kk temp logits
    iterDecode ks.b kk temp (n - 1) (pos + 1) (Array.snoc ctx nextTok) newCache

-- | KVCache-accelerated top-p (nucleus) temperature sampling.
generateTopP
  :: Key
  -> Number        -- ^ p (e.g. 0.9)
  -> Number        -- ^ temperature
  -> ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateTopP initialKey p temperature cfg weights rope prompt maxNew = do
  cache0 <- emptyKVCacheStack cfg
  promptIds <- arrayInt1D prompt
  { newCache: cache1, logits: l1 } <-
    forwardCached cfg weights rope cache0 0 promptIds
  dispose promptIds
  ks <- splitKey2 initialKey
  firstNew <- sampleLastRowTopP ks.a p temperature l1
  let
    startPos = Array.length prompt
    ctx0 = Array.snoc prompt firstNew
  final <- iterDecode ks.b p temperature (maxNew - 1) (startPos + 1) ctx0 cache1
  traverse_ disposeKVCache final.cache
  pure final.ctx
  where
  iterDecode
    :: Key
    -> Number
    -> Number
    -> Int
    -> Int
    -> Array Int
    -> KVCacheStack
    -> Effect { ctx :: Array Int, cache :: KVCacheStack }
  iterDecode _ _ _ 0 _ ctx cache = pure { ctx, cache }
  iterDecode key pp temp n pos ctx cache = do
    let lastTok = case Array.last ctx of
          Just x -> x
          Nothing -> 0
    tokIds <- arrayInt1D [ lastTok ]
    { newCache, logits } <- forwardCached cfg weights rope cache pos tokIds
    dispose tokIds
    ks <- splitKey2 key
    nextTok <- sampleLastRowTopP ks.a pp temp logits
    iterDecode ks.b pp temp (n - 1) (pos + 1) (Array.snoc ctx nextTok) newCache

-- | Slice last row → top-p sample. Consumes `key` and `logits`.
sampleLastRowTopP :: Key -> Number -> Number -> NDArray D2 -> Effect Int
sampleLastRowTopP key p temp logits = do
  sh <- shape logits
  let
    seqLen = fromMaybe 0 (Array.head sh)
    vocab = fromMaybe 0 (sh Array.!! 1)
  logitsR <- ref logits
  lastRow <- sliceAxis logitsR 0 (seqLen - 1) seqLen
  dispose logits
  lastRowR <- ref lastRow
  flat <- reshape lastRowR [ vocab ] :: Effect (NDArray D1)
  dispose lastRow
  idx <- sampleTopP key p temp flat
  dispose flat
  pure idx

-- | KVCache-accelerated greedy generation that streams each new token to
-- | a user-supplied callback. The callback is called once per new token,
-- | in order — useful for incremental rendering of generation.
generateGreedyCachedStream
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int       -- ^ prompt
  -> Int             -- ^ maxNew
  -> (Int -> Effect Unit)  -- ^ called once per generated token id
  -> Effect Unit
generateGreedyCachedStream cfg weights rope prompt maxNew onToken =
  generateGreedyCachedStreamUntil cfg weights rope Nothing prompt maxNew onToken

-- | KVCache-accelerated greedy generation that streams each new token to
-- | a callback and stops early when the EOS id is sampled. Pass
-- | `Nothing` for `eosId` to disable early stopping (matches
-- | `generateGreedyCachedStream`).
-- |
-- | The EOS token *is* delivered to the callback before the loop
-- | terminates — callers can choose to render or hide it.
generateGreedyCachedStreamUntil
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Maybe Int       -- ^ eos id (Nothing = no early stop)
  -> Array Int       -- ^ prompt
  -> Int             -- ^ maxNew
  -> (Int -> Effect Unit)
  -> Effect Unit
generateGreedyCachedStreamUntil cfg weights rope eosId prompt maxNew onToken =
  generateGreedyCachedStreamUntilWithHead
    cfg weights weights.embedding rope eosId prompt maxNew onToken

-- | `generateGreedyCachedStreamUntil` with an explicit LM-head projection
-- | matrix. Use when the loaded checkpoint has `tie_word_embeddings: false`
-- | and ships a separate `lm_head.weight`. For tied checkpoints, the
-- | callers above pass `weights.embedding` and get the original behaviour.
generateGreedyCachedStreamUntilWithHead
  :: ModelConfig
  -> ModelWeights
  -> NDArray D2          -- ^ LM-head projection [vocab, hidden]
  -> RoPETables
  -> Maybe Int           -- ^ eos id (Nothing = no early stop)
  -> Array Int
  -> Int
  -> (Int -> Effect Unit)
  -> Effect Unit
generateGreedyCachedStreamUntilWithHead
  cfg weights head rope eosId prompt maxNew onToken = do
  cache0 <- emptyKVCacheStack cfg
  promptIds <- arrayInt1D prompt
  { newCache: cache1, logits: l1 } <-
    forwardCachedWithHead cfg weights head rope cache0 0 promptIds
  dispose promptIds
  firstNew <- sampleLastRow l1
  onToken firstNew
  let
    startPos = Array.length prompt
    isEos = case eosId of
      Just e -> firstNew == e
      Nothing -> false
  finalCache <-
    if isEos then pure cache1
    else streamLoop firstNew (startPos + 1) (maxNew - 1) cache1
  traverse_ disposeKVCache finalCache
  where
  streamLoop
    :: Int
    -> Int
    -> Int
    -> KVCacheStack
    -> Effect KVCacheStack
  streamLoop _ _ 0 cache = pure cache
  streamLoop lastTok pos n cache = do
    tokIds <- arrayInt1D [ lastTok ]
    { newCache, logits } <-
      forwardCachedWithHead cfg weights head rope cache pos tokIds
    dispose tokIds
    nextTok <- sampleLastRow logits
    onToken nextTok
    case eosId of
      Just e | nextTok == e -> pure newCache
      _ -> streamLoop nextTok (pos + 1) (n - 1) newCache
