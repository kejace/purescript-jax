-- | Faithful microGPT-spirit port — pure PureScript, no `Jax.*`.
-- |
-- | This file is the *pedagogy* sibling: every gradient is hand-walked
-- | by `Value.backward`. The Reimagined demo (../jax/) does the same
-- | end-to-end task with `Jax.Autodiff.valueAndGradT`; reading the two
-- | side by side shows what JAX hides.
-- |
-- | To keep the port readable in a single afternoon, this version
-- | builds a *bigram-conditioned MLP* for next-character prediction
-- | rather than the full transformer:
-- |
-- |     forward(c) = silu(emb[c] @ W1 + b1) @ W2 + b2
-- |
-- | Same training shape as Karpathy's gist (cross-entropy loss,
-- | scalar autograd, hand-rolled Adam, temperature sampling). The
-- | full transformer's attention block translates to ~150 more lines
-- | of the same shape; we leave that as an exercise and let the
-- | Reimagined demo show the scaled-up version.
-- |
-- | Run from this directory: `bunx spago run`.
module Microgpt.Faithful.Main where

import Prelude

import Data.Array as Array
import Data.Foldable (foldM, for_, sum, traverse_)
import Data.Int (toNumber, floor)
import Data.Maybe (Maybe(..), fromMaybe)
import Data.Number as N
import Data.String.CodeUnits (toCharArray, singleton) as Str
import Data.Traversable (traverse)
import Data.Tuple (Tuple(..))
import Effect (Effect)
import Effect.Console (log)
import Effect.Random (random)
import Effect.Ref (Ref)
import Effect.Ref as Ref
import Microgpt.Faithful.Value (Value)
import Microgpt.Faithful.Value as V

-- =============================================================================
-- Section 0: corpus (inline; see README for how to swap in names.txt)
-- =============================================================================

namesCorpus :: String
namesCorpus = """ada
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
evelyn
abigail
emily
elizabeth
mila
ella
avery
sofia
camila
aria
scarlett
victoria
madison
luna
grace
chloe
penelope
layla
riley
zoey
"""

-- =============================================================================
-- Section 1: char tokenizer
-- =============================================================================

-- First-occurrence ordering — stable across calls.
buildVocab :: String -> Array Char
buildVocab text = Array.foldl step [] (Str.toCharArray text)
  where
  step acc c
    | Array.elem c acc = acc
    | otherwise = Array.snoc acc c

charToId :: Array Char -> Char -> Int
charToId vocab c = fromMaybe 0 (Array.elemIndex c vocab)

idToChar :: Array Char -> Int -> Char
idToChar vocab i = fromMaybe ' ' (Array.index vocab i)

-- =============================================================================
-- Section 2: parameter init
-- =============================================================================

-- Box-Muller box-1: two uniform draws → one standard normal.
randn :: Effect Number
randn = do
  u1 <- random
  u2 <- random
  pure (N.sqrt (-2.0 * N.log u1) * N.cos (2.0 * N.pi * u2))

-- Vector of n Values from N(0, sigma²).
randVec :: Int -> Number -> Effect (Array Value)
randVec n sigma = traverse (\_ -> do
  z <- randn
  V.mk (z * sigma)) (Array.range 0 (n - 1))

-- Glorot-ish [rows, cols] matrix init.
randMat :: Int -> Int -> Effect (Array (Array Value))
randMat rows cols = do
  let sigma = N.sqrt (2.0 / toNumber (rows + cols))
  traverse (\_ -> randVec cols sigma) (Array.range 0 (rows - 1))

-- Model parameters.
type Params =
  { emb :: Array (Array Value)   -- [vocab, hidden]
  , w1  :: Array (Array Value)   -- [intermediate, hidden]
  , b1  :: Array Value           -- [intermediate]
  , w2  :: Array (Array Value)   -- [vocab, intermediate]
  , b2  :: Array Value           -- [vocab]
  }

mkParams :: Int -> Int -> Int -> Effect Params
mkParams vocab hidden intermediate = do
  emb <- randMat vocab hidden
  w1  <- randMat intermediate hidden
  b1  <- randVec intermediate 0.0
  w2  <- randMat vocab intermediate
  b2  <- randVec vocab 0.0
  pure { emb, w1, b1, w2, b2 }

-- Flatten to a single list — the optimizer iterates over this.
flatten :: Params -> Array Value
flatten p =
  Array.concat p.emb
    <> Array.concat p.w1
    <> p.b1
    <> Array.concat p.w2
    <> p.b2

-- =============================================================================
-- Section 3: model — silu(emb[c] @ W1 + b1) @ W2 + b2
-- =============================================================================

-- linear: y[i] = Σⱼ x[j] · W[i][j] + b[i]
-- W is [out, in], x is [in], b is [out], output is [out].
linear :: Array Value -> Array (Array Value) -> Array Value -> Effect (Array Value)
linear x w b = traverse step (Array.zip w b)
  where
  step (Tuple wRow bi) = do
    d <- dot x wRow
    V.add d bi

-- Inner product. Folds across pairs; seeds with the first product to
-- keep the autograd graph sized to N-1 adds (instead of N).
dot :: Array Value -> Array Value -> Effect Value
dot xs ys = case Array.uncons (Array.zip xs ys) of
  Nothing -> V.mk 0.0
  Just { head: Tuple x0 y0, tail } -> do
    p0 <- V.mul x0 y0
    foldM
      (\acc (Tuple a b) -> do
        p <- V.mul a b
        V.add acc p)
      p0
      tail

-- silu(x) = x · σ(x) = x / (1 + e^-x). Composed from primitives so
-- the Value graph captures the right derivative.
silu :: Value -> Effect Value
silu x = do
  nx <- V.neg x
  ex <- V.exp nx
  one <- V.mk 1.0
  denom <- V.add one ex
  recipD <- V.divv one denom
  V.mul x recipD

-- Numerically stable softmax. Uses scalar values for the max so we
-- don't spam the autograd graph with a max-Value op (the max is a
-- detached subtraction constant).
softmaxV :: Array Value -> Effect (Array Value)
softmaxV xs = do
  vs <- traverse V.value xs
  let
    maxV = fromMaybe 0.0 (Array.head (Array.sortBy (\a b -> compare b a) vs))
  maxC <- V.mk maxV
  shifted <- traverse (\x -> V.sub x maxC) xs
  exps <- traverse V.exp shifted
  total <- case Array.uncons exps of
    Nothing -> V.mk 1.0
    Just { head, tail } -> foldM V.add head tail
  traverse (\e -> V.divv e total) exps

-- Forward: int token id → logits over vocab.
forward :: Params -> Int -> Effect (Array Value)
forward p tokenId = case Array.index p.emb tokenId of
  Nothing -> pure []
  Just embRow -> do
    h0 <- linear embRow p.w1 p.b1
    h1 <- traverse silu h0
    linear h1 p.w2 p.b2

-- Cross-entropy: -log(softmax(logits)[targetId]).
crossEntropy :: Array Value -> Int -> Effect Value
crossEntropy logits targetId = do
  probs <- softmaxV logits
  case Array.index probs targetId of
    Nothing -> V.mk 1.0e9
    Just p -> do
      lp <- V.log p
      V.neg lp

-- =============================================================================
-- Section 4: training loop with hand-rolled Adam
-- =============================================================================

type AdamSlot = { m :: Ref Number, v :: Ref Number }

mkSlot :: Effect AdamSlot
mkSlot = do
  m <- Ref.new 0.0
  v <- Ref.new 0.0
  pure { m, v }

-- Mirror Karpathy's loop verbatim:
--   m[i] = β₁·m[i] + (1-β₁)·g
--   v[i] = β₂·v[i] + (1-β₂)·g²
--   m̂   = m[i] / (1 - β₁^t)
--   v̂   = v[i] / (1 - β₂^t)
--   p   -= lr · m̂ / (√v̂ + ε)
adamStep
  :: Number -> Number -> Number -> Number -> Int
  -> Array (Tuple Value AdamSlot)
  -> Effect Unit
adamStep lr b1 b2 eps t paramsAndSlots = traverse_ stepOne paramsAndSlots
  where
  stepOne (Tuple p slot) = do
    g <- V.gradV p
    mPrev <- Ref.read slot.m
    vPrev <- Ref.read slot.v
    let
      mNext = b1 * mPrev + (1.0 - b1) * g
      vNext = b2 * vPrev + (1.0 - b2) * g * g
      mHat  = mNext / (1.0 - N.pow b1 (toNumber t))
      vHat  = vNext / (1.0 - N.pow b2 (toNumber t))
    pVal <- V.value p
    V.setValue p (pVal - lr * mHat / (N.sqrt vHat + eps))
    Ref.write mNext slot.m
    Ref.write vNext slot.v
    V.zeroGrad p

-- =============================================================================
-- Section 5: sampling — temperature softmax + categorical
-- =============================================================================

sampleCategorical :: Array Number -> Effect Int
sampleCategorical probs = do
  u <- random
  pure (go 0 0.0 u)
  where
  go i acc u
    | i >= Array.length probs = Array.length probs - 1
    | otherwise = case Array.index probs i of
        Just p ->
          let acc' = acc + p
          in if u <= acc' then i else go (i + 1) acc' u
        Nothing -> i

sampleSeq :: Params -> Array Char -> Number -> Int -> Int -> Effect String
sampleSeq p vocab temperature maxLen startId = go startId maxLen ""
  where
  go cur n acc
    | n <= 0 = pure (acc <> "…")
    | otherwise = do
        logits <- forward p cur
        vs <- traverse V.value logits
        let
          scaled = map (_ / temperature) vs
          maxV = fromMaybe 0.0 (Array.head (Array.sortBy (\a b -> compare b a) scaled))
          expd = map (\x -> N.exp (x - maxV)) scaled
          total = sum expd
          probs = map (_ / total) expd
        next <- sampleCategorical probs
        let c = idToChar vocab next
        if c == '\n'
          then pure (acc <> "\n")
          else go next (n - 1) (acc <> Str.singleton c)

-- =============================================================================
-- main: end-to-end glue.
-- =============================================================================

main :: Effect Unit
main = do
  log "[microgpt-faithful] start"
  let
    vocab = buildVocab namesCorpus
    vocabSize = Array.length vocab
    hidden = 8
    intermediate = 16
    numSteps = 200
    lr0 = 0.05
  log $ "[microgpt-faithful] vocab=" <> show vocabSize
    <> " hidden=" <> show hidden <> " intermediate=" <> show intermediate
  -- Encode the corpus as (input, target) char-pair examples. One pair
  -- is sampled per training step (matches Karpathy: one example per
  -- iteration; the whole-corpus accumulation would blow the autograd
  -- graph past 4 GB for any non-toy `numSteps`).
  let
    ids = map (charToId vocab) (Str.toCharArray namesCorpus)
    pairs = case Array.tail ids of
      Nothing -> []
      Just xs -> Array.zip ids xs
    nPairs = Array.length pairs
  log $ "[microgpt-faithful] training pairs available: " <> show nPairs
  params <- mkParams vocabSize hidden intermediate
  let allParams = flatten params
  slots <- traverse (\_ -> mkSlot) allParams
  let paramSlots = Array.zip allParams slots
  for_ (Array.range 1 numSteps) \t -> do
    -- Pick one (input, target) pair uniformly. The Value graph is
    -- discarded after backward + step; per-step working set is a
    -- few thousand allocations, GC'd immediately.
    u <- random
    let i = clamp 0 (nPairs - 1) (floor (u * toNumber nPairs))
    case Array.index pairs i of
      Nothing -> pure unit
      Just (Tuple inp tgt) -> do
        let lrT = lr0 * (1.0 - toNumber (t - 1) / toNumber numSteps)
        logits <- forward params inp
        loss <- crossEntropy logits tgt
        V.backward loss
        adamStep lrT 0.9 0.999 1.0e-8 t paramSlots
        when (mod t 20 == 0 || t == 1) do
          lossN <- V.value loss
          log $ "  step " <> show t <> " · loss " <> show lossN
  log "[microgpt-faithful] sampling 5 names:"
  let startId = charToId vocab 'a'
  for_ (Array.range 1 5) \_ -> do
    s <- sampleSeq params vocab 1.0 16 startId
    log $ "  → " <> s
  log "[microgpt-faithful] done"
