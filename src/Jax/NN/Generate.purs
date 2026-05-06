-- | Autoregressive generation built on a single primitive `decodeLoop`.
-- |
-- | All public `generate*` functions are thin wrappers that pick a
-- | `Sampler` (greedy / temperature / top-k / top-p) and a `Reporter`
-- | (silent collector or per-token callback for streaming UIs), then
-- | call `decodeLoop`. This is a one-stop shop for autoregressive
-- | inference; the original five mutually-recursive decode loops have
-- | collapsed into one.
-- |
-- | Memory discipline (jax-js refcount):
-- |   * `forwardCached*` returns a fresh logits tensor; the loop owns
-- |     and disposes it via `withLastRow`.
-- |   * The KVCacheStack is consumed by each `forwardCached*` call and
-- |     replaced with the returned `newCache` — no leaks unless the
-- |     loop body throws (in which case the unfinished cache leaks; we
-- |     don't currently bracket-protect this because there's no
-- |     recoverable scenario in our callers).
module Jax.NN.Generate
  ( -- * The primitive
    Sampler(..)
  , runSampler
  , Reporter
  , StopCond
  , decodeLoop
  , decodeLoopWithHead
  -- * Helpers
  , withLastRow
  -- * Pre-baked recipes
  , generateGreedy
  , generateGreedyCached
  , generateTemperature
  , generateTopK
  , generateTopP
  , generateGreedyCachedStream
  , generateGreedyCachedStreamUntil
  , generateGreedyCachedStreamUntilWithHead
  ) where

import Prelude

import Control.Monad.Rec.Class (Step(..), tailRecM)
import Data.Array (last, length, snoc, head, (!!)) as Array
import Data.Foldable (traverse_)
import Data.Maybe (Maybe(..), fromMaybe)
import Effect (Effect)
import Effect.Ref as Ref
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

-- | A sampler turns a 1D logits tensor into a token id and the next
-- | sampler to use. The next-sampler return lets temperature/top-k/
-- | top-p re-key on each step without inventing a separate API per
-- | sampler. Greedy ignores keys and returns itself; keyed samplers
-- | split a private key on each invocation.
-- |
-- | Newtype rather than `type` because the recursive shape
-- | (`Sampler` → `Effect { ... Sampler }`) would be a cycle in a type
-- | synonym. The `NDArray D1` argument is consumed.
newtype Sampler = Sampler (NDArray D1 -> Effect { token :: Int, next :: Sampler })

runSampler :: Sampler -> NDArray D1 -> Effect { token :: Int, next :: Sampler }
runSampler (Sampler f) = f

-- | Per-token reporter. `mempty`-equivalent for batch generation;
-- | populated for streaming UIs.
type Reporter = Int -> Effect Unit

-- | Termination predicate (e.g., on EOS). `const false` to disable.
type StopCond = Int -> Boolean

-- | Slice the last row of a `[seq, vocab]` logits tensor and call the
-- | continuation with the resulting `[vocab]` tensor. Both the input
-- | and the intermediate slice are disposed before/after the
-- | continuation runs.
-- |
-- | This folds the slice → reshape → flatten ritual into one place;
-- | every sampler call site used to repeat ~7 lines of plumbing.
withLastRow :: forall a. NDArray D2 -> (NDArray D1 -> Effect a) -> Effect a
withLastRow logits k = do
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
  result <- k flat
  dispose flat
  pure result

-- | The decode primitive. Prefill on the prompt, then loop one token
-- | at a time:
-- |   1. Forward through `forwardCachedWithHead` for the new token(s).
-- |   2. Sample via the supplied `Sampler`, threading its next-step state.
-- |   3. Report (for streaming) and append.
-- |   4. Stop on `stop`-true or after `maxNew` tokens.
-- |
-- | Stack-safe via `tailRecM`: the loop body returns `Step` values, so
-- | even at very large `maxNew` we don't grow the call stack.
decodeLoopWithHead
  :: ModelConfig
  -> ModelWeights
  -> NDArray D2       -- ^ LM-head projection [vocab, hidden]
  -> RoPETables
  -> Sampler
  -> Reporter
  -> StopCond
  -> Array Int        -- ^ prompt
  -> Int              -- ^ maxNew
  -> Effect (Array Int)
decodeLoopWithHead cfg weights head rope sampler0 report stop prompt maxNew = do
  cache0 <- emptyKVCacheStack cfg
  -- Prefill.
  promptIds <- arrayInt1D prompt
  { newCache: cacheAfterPrefill, logits: l1 } <-
    forwardCachedWithHead cfg weights head rope cache0 0 promptIds
  dispose promptIds
  { token: firstNew, next: sampler1 } <- withLastRow l1 (runSampler sampler0)
  report firstNew
  let startPos = Array.length prompt
  -- Decode loop, stack-safely. State carries (remaining, position,
  -- accumulated context, cache, sampler).
  finalCtxRef <- Ref.new (Array.snoc prompt firstNew)
  finalCacheRef <- Ref.new cacheAfterPrefill
  if stop firstNew then pure unit
  else do
    let
      stepInit =
        { remaining: maxNew - 1
        , pos: startPos + 1
        , cache: cacheAfterPrefill
        , sampler: sampler1
        , last: firstNew
        }
      step s
        | s.remaining <= 0 = do
            Ref.write s.cache finalCacheRef
            pure (Done unit)
        | otherwise = do
            tokIds <- arrayInt1D [ s.last ]
            { newCache, logits } <-
              forwardCachedWithHead cfg weights head rope s.cache s.pos tokIds
            dispose tokIds
            { token: nextTok, next: nextSampler } <- withLastRow logits (runSampler s.sampler)
            report nextTok
            Ref.modify_ (\xs -> Array.snoc xs nextTok) finalCtxRef
            if stop nextTok then do
              Ref.write newCache finalCacheRef
              pure (Done unit)
            else
              pure $ Loop
                { remaining: s.remaining - 1
                , pos: s.pos + 1
                , cache: newCache
                , sampler: nextSampler
                , last: nextTok
                }
    tailRecM step stepInit
  finalCache <- Ref.read finalCacheRef
  traverse_ disposeKVCache finalCache
  Ref.read finalCtxRef

-- | `decodeLoop` with weight-tied LM head (head = embedding).
decodeLoop
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Sampler
  -> Reporter
  -> StopCond
  -> Array Int
  -> Int
  -> Effect (Array Int)
decodeLoop cfg weights rope =
  decodeLoopWithHead cfg weights weights.embedding rope

disposeKVCache :: KVCache -> Effect Unit
disposeKVCache kv = do
  dispose kv.k
  dispose kv.v

silent :: Reporter
silent _ = pure unit

noStop :: StopCond
noStop _ = false

-- =============================================================================
-- Sampler builders
-- =============================================================================

-- | Greedy sampler: argmax of logits; key-irrelevant.
-- |
-- | Convention (shared by all samplers): the underlying `sample*`
-- | helpers (`sampleGreedy` etc.) leave the logits tensor at refcount
-- | 1 — they ref-bump internally so the caller still owns one
-- | reference. `withLastRow` is the sole owner of the flat logits and
-- | disposes after this continuation returns; samplers must NOT
-- | dispose, or we double-free.
greedy :: Sampler
greedy = Sampler \logits -> do
  token <- sampleGreedy logits
  pure { token, next: greedy }

-- | Temperature sampler: scales logits by 1/temperature, then draws a
-- | categorical sample. Splits the seed key on each invocation.
temperatureSampler :: Key -> Number -> Sampler
temperatureSampler key temp = Sampler \logits -> do
  ks <- splitKey2 key
  token <- sampleTemperature ks.a temp logits
  pure { token, next: temperatureSampler ks.b temp }

topKSampler :: Key -> Int -> Number -> Sampler
topKSampler key k temp = Sampler \logits -> do
  ks <- splitKey2 key
  token <- sampleTopK ks.a k temp logits
  pure { token, next: topKSampler ks.b k temp }

topPSampler :: Key -> Number -> Number -> Sampler
topPSampler key p temp = Sampler \logits -> do
  ks <- splitKey2 key
  token <- sampleTopP ks.a p temp logits
  pure { token, next: topPSampler ks.b p temp }

-- =============================================================================
-- Pre-baked generation recipes
-- =============================================================================

-- | Greedy generation, recomputing the full forward at every step
-- | (O(n²) total, no KV cache). Kept for testing parity with the
-- | cached variant; production code should use `generateGreedyCached`.
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
    nextId <- withLastRow logits \flat -> do
      idx <- sampleGreedy flat
      pure idx
    go (Array.snoc ctx nextId) (n - 1)

-- | KVCache-accelerated greedy generation. Prefill once, decode one
-- | token at a time against the cached K/V. O(n) decode.
generateGreedyCached
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateGreedyCached cfg weights rope prompt maxNew =
  decodeLoop cfg weights rope greedy silent noStop prompt maxNew

-- | KVCache-accelerated temperature-sampled generation.
generateTemperature
  :: Key
  -> Number
  -> ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateTemperature key temperature cfg weights rope =
  decodeLoop cfg weights rope (temperatureSampler key temperature) silent noStop

-- | KVCache-accelerated top-k sampling.
generateTopK
  :: Key
  -> Int
  -> Number
  -> ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateTopK key k temperature cfg weights rope =
  decodeLoop cfg weights rope (topKSampler key k temperature) silent noStop

-- | KVCache-accelerated top-p (nucleus) sampling.
generateTopP
  :: Key
  -> Number
  -> Number
  -> ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> Effect (Array Int)
generateTopP key p temperature cfg weights rope =
  decodeLoop cfg weights rope (topPSampler key p temperature) silent noStop

-- =============================================================================
-- Streaming variants
-- =============================================================================

-- | Greedy generation that streams each new token to a callback in order.
generateGreedyCachedStream
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Array Int
  -> Int
  -> (Int -> Effect Unit)
  -> Effect Unit
generateGreedyCachedStream cfg weights rope prompt maxNew onToken =
  generateGreedyCachedStreamUntil cfg weights rope Nothing prompt maxNew onToken

-- | Greedy streaming with optional EOS-aware early stop. The EOS token
-- | is delivered to the callback before the loop terminates.
generateGreedyCachedStreamUntil
  :: ModelConfig
  -> ModelWeights
  -> RoPETables
  -> Maybe Int
  -> Array Int
  -> Int
  -> (Int -> Effect Unit)
  -> Effect Unit
generateGreedyCachedStreamUntil cfg weights rope eosId prompt maxNew onToken =
  generateGreedyCachedStreamUntilWithHead
    cfg weights weights.embedding rope eosId prompt maxNew onToken

-- | Greedy streaming with EOS + an explicit LM-head matrix (for
-- | checkpoints with `tie_word_embeddings: false`).
generateGreedyCachedStreamUntilWithHead
  :: ModelConfig
  -> ModelWeights
  -> NDArray D2
  -> RoPETables
  -> Maybe Int
  -> Array Int
  -> Int
  -> (Int -> Effect Unit)
  -> Effect Unit
generateGreedyCachedStreamUntilWithHead cfg weights head rope eosId prompt maxNew onToken =
  void $ decodeLoopWithHead cfg weights head rope greedy onToken stopFn prompt maxNew
  where
  stopFn = case eosId of
    Just e -> (_ == e)
    Nothing -> noStop
