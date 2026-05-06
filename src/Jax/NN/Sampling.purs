module Jax.NN.Sampling
  ( sampleGreedy
  , sampleTemperature
  , sampleTopK
  , sampleTopP
  ) where

import Prelude

import Data.Array (head, (!!), length)
import Data.Maybe (Maybe(..), fromMaybe)
import Effect (Effect)
import Effect.Exception (throw)
import Jax.Core
  ( D1
  , NDArray
  , argmax
  , cumsum
  , dispose
  , mulScalar
  , ref
  , shape
  , sliceLastAxis
  , softmax
  , toJs
  , topK
  )
import Jax.Random (Key, sampleCategorical)
import Unsafe.Coerce (unsafeCoerce)

-- | Greedy sampling: index of the largest logit. Input is borrowed.
sampleGreedy :: NDArray D1 -> Effect Int
sampleGreedy logits = do
  logitsR <- ref logits
  idxArr <- argmax logitsR 0
  raw <- toJs idxArr
  dispose idxArr
  pure (unsafeCoerce raw :: Int)

-- | Temperature sampling. Scales logits by `1/temperature` before drawing
-- | a categorical sample. Lower temperature → sharper; higher → flatter.
-- | Consumes the `Key`. The `logits` tensor is borrowed.
sampleTemperature
  :: Key
  -> Number
  -> NDArray D1
  -> Effect Int
sampleTemperature key temperature logits = do
  logitsR <- ref logits
  scaled <- mulScalar logitsR (1.0 / temperature)
  sampleCategorical key scaled

-- | Top-k temperature sampling. Restricts the categorical distribution to
-- | the `k` largest logits, scaled by `1/temperature`. With `temperature`
-- | very small the result approaches argmax; with k=vocab_size it
-- | degenerates to plain `sampleTemperature`.
-- |
-- | Consumes the `Key`; `logits` is borrowed (the underlying topK
-- | normally consumes its argument, so we ref-bump first).
sampleTopK
  :: Key
  -> Int       -- ^ k (must be ≥ 1 and ≤ vocab_size)
  -> Number    -- ^ temperature
  -> NDArray D1
  -> Effect Int
sampleTopK key k temperature logits = do
  -- topK consumes its argument; ref so the caller's logits stays alive
  logitsR <- ref logits
  top <- topK logitsR k 0
  -- Scale the top-k values by 1/temperature
  vR <- ref top.values
  scaled <- mulScalar vR (1.0 / temperature)
  -- Sample within the top-k (returns position in [0, k))
  localPos <- sampleCategorical key scaled
  -- Read top.indices into a JS Array Int and look up the actual vocab id
  raw <- toJs top.indices
  dispose top.values
  dispose top.indices
  let allIndices = unsafeCoerce raw :: Array Int
  case allIndices !! localPos of
    Just idx -> pure idx
    Nothing -> throw $ "sampleTopK: localPos " <> show localPos
      <> " out of range for k=" <> show k

-- | Top-p (nucleus) temperature sampling. Restricts the categorical
-- | distribution to the smallest set of tokens whose cumulative softmax
-- | mass exceeds `p`. With `p=1.0` it degenerates to plain temperature
-- | sampling; with very small `p` it approaches argmax.
-- |
-- | Implementation: sort scaled logits descending, compute softmax+cumsum
-- | over the sorted distribution, find the smallest k whose cumulative
-- | mass crosses `p` (CPU-side scan), then categorical-sample from the
-- | top-k slice.
-- |
-- | Vocab is read from the input's first dim (the only dim of a 1D
-- | logits tensor). `key` is consumed; `logits` is borrowed.
sampleTopP
  :: Key
  -> Number     -- ^ p (typically 0.9)
  -> Number     -- ^ temperature
  -> NDArray D1
  -> Effect Int
sampleTopP key p temperature logits = do
  -- Read the vocab size; jax-js's topK errors if k > size, so we can't
  -- just pass a huge k.
  shapeArr <- shape logits
  let vocab = fromMaybe 1 (head shapeArr)
  -- Apply temperature.
  logitsR <- ref logits
  scaled <- mulScalar logitsR (1.0 / temperature)
  scaledR <- ref scaled
  top <- topK scaledR vocab 0
  dispose scaled
  -- Read sorted values to softmax + cumsum.
  vR1 <- ref top.values
  probs <- softmax vR1 0
  cumprobs <- cumsum probs 0
  cumForeign <- toJs cumprobs
  dispose cumprobs
  let
    cumArr = unsafeCoerce cumForeign :: Array Number
    n = length cumArr
    k = findThresholdK cumArr p
  -- Truncate the sorted values to the nucleus and sample.
  vR2 <- ref top.values
  truncated <- sliceLastAxis vR2 0 (k + 1)
  localPos <- sampleCategorical key truncated
  -- Look up the actual token index.
  idxForeign <- toJs top.indices
  dispose top.values
  dispose top.indices
  let allIndices = unsafeCoerce idxForeign :: Array Int
  case allIndices !! localPos of
    Just idx -> pure idx
    Nothing -> throw $ "sampleTopP: localPos " <> show localPos
      <> " out of range (n=" <> show n <> ", k=" <> show k <> ")"

-- | Smallest index `k` such that `cumprobs[k] >= p`. If p > all sums,
-- | returns the last index. Linear CPU scan; runs over the full vocab
-- | per sample, but vocab is typically ≤ 1e5 so it's O(vocab) per step.
findThresholdK :: Array Number -> Number -> Int
findThresholdK cumprobs p = go 0
  where
  n = length cumprobs
  go i
    | i >= n - 1 = i
    | otherwise = case cumprobs !! i of
        Just c | c >= p -> i
        _ -> go (i + 1)
