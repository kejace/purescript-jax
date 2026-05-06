-- | Heterogeneous traversal of `ModelWeights`-shaped records.
-- |
-- | The `heterogeneous` package handles records via `RowToList` to
-- | iterate over fields at compile time. Per-field handlers are written
-- | as `Folding` typeclass instances on a singleton tag — one instance
-- | per leaf-type pattern. To recurse into nested records / arrays we
-- | add instances that delegate to `hfoldl` / `foldl` respectively.
-- |
-- | This replaces the `unsafeCoerce`-shaped trust we'd otherwise put
-- | in jax-js's pytree assumption — `countTensors` and friends are
-- | type-checked, so adding a new field to `ModelWeights` is either a
-- | compile-time addition (new instance needed) or works for free
-- | (the field's type already matches an existing instance).
module Jax.Pytree
  ( -- * Counting
    countTensors
  , countParams
  -- * Statistics
  , sumSquaredL2
  -- * Tags (re-export so callers can use the same instances)
  , CountTensor(..)
  , CountParam(..)
  , SumSquaredL2(..)
  ) where

import Prelude

import Data.Foldable (foldl) as Foldable
import Effect (Effect)
import Heterogeneous.Folding (class Folding, class HFoldl, folding, hfoldl)
import Jax.Coerce (asNumber)
import Jax.Core (NDArray, dispose, ref, shape, square, sum, toJs)

-- =============================================================================
-- countTensors
-- =============================================================================

-- | Singleton tag for "count NDArray leaves".
data CountTensor = CountTensor

-- Folding instance: every NDArray leaf adds 1 to the running count.
instance foldCountTensorLeaf :: Folding CountTensor Int (NDArray d) Int where
  folding _ acc _ = acc + 1

-- Nested-record descent: when a field is itself a record, recurse via
-- hfoldl. The constraint says "if f folds over { | r } to produce b",
-- which is satisfied by the built-in `hfoldlRecord` instance.
instance foldCountTensorRec ::
  HFoldl CountTensor a { | r } b =>
  Folding CountTensor a { | r } b
  where
  folding f acc r = hfoldl f acc r

-- Array descent: fold each element through the same singleton folder.
instance foldCountTensorArr ::
  Folding CountTensor a el a =>
  Folding CountTensor a (Array el) a
  where
  folding f acc xs = Foldable.foldl (folding f) acc xs

-- | Count the number of tensor leaves anywhere in a record-of-tensors
-- | tree. Works on `ModelWeights`, `LayerWeights`, `AttentionWeights`,
-- | etc., because `heterogeneous` walks records by row-list.
-- |
-- | Example: a 6-layer Llama-arch model with tied embedding has:
-- |   embedding (1) + finalNorm (1)
-- |   + 6 × (attnNorm + mlpNorm + 4 attn projections + 3 mlp projections)
-- |   = 2 + 6 × 9 = 56
countTensors
  :: forall r
   . HFoldl CountTensor Int { | r } Int
  => { | r }
  -> Int
countTensors r = hfoldl CountTensor 0 r

-- =============================================================================
-- countParams
-- =============================================================================

-- | Singleton tag for "sum the product of every NDArray's shape".
data CountParam = CountParam

-- Accumulator is `Effect Int` so we can call `shape` (which is Effect)
-- without escaping the type.
instance foldCountParamLeaf :: Folding CountParam (Effect Int) (NDArray d) (Effect Int) where
  folding _ accE x = do
    acc <- accE
    sh <- shape x
    pure (acc + Foldable.foldl (*) 1 sh)

instance foldCountParamRec ::
  HFoldl CountParam a { | r } b =>
  Folding CountParam a { | r } b
  where
  folding f acc r = hfoldl f acc r

instance foldCountParamArr ::
  Folding CountParam a el a =>
  Folding CountParam a (Array el) a
  where
  folding f acc xs = Foldable.foldl (folding f) acc xs

-- | Total parameter count: sum of (product of shape) across every
-- | NDArray leaf in the tree. For a 101M-param model this returns
-- | ~101 million.
countParams
  :: forall r
   . HFoldl CountParam (Effect Int) { | r } (Effect Int)
  => { | r }
  -> Effect Int
countParams r = hfoldl CountParam (pure 0 :: Effect Int) r

-- =============================================================================
-- sumSquaredL2 — aggregate sum-of-squares across all weight tensors
-- =============================================================================

-- | Singleton tag for "accumulate Σᵢ wᵢ²" across all NDArray leaves.
data SumSquaredL2 = SumSquaredL2

instance foldSumSquaredL2Leaf
  :: Folding SumSquaredL2 (Effect Number) (NDArray d) (Effect Number)
  where
  folding _ accE x = do
    acc <- accE
    -- x.square().sum() is the per-tensor sum-of-squares; we accumulate
    -- those into a Number on the host. The intermediate squared
    -- tensor is consumed by `sum` (one allocation per leaf, freed
    -- before we read the scalar back).
    xR <- ref x
    sq <- square xR
    s <- sum sq
    sF <- toJs s
    dispose s
    pure (acc + asNumber sF)

instance foldSumSquaredL2Rec ::
  HFoldl SumSquaredL2 a { | r } b =>
  Folding SumSquaredL2 a { | r } b
  where
  folding f acc r = hfoldl f acc r

instance foldSumSquaredL2Arr ::
  Folding SumSquaredL2 a el a =>
  Folding SumSquaredL2 a (Array el) a
  where
  folding f acc xs = Foldable.foldl (folding f) acc xs

-- | Aggregate Σᵢ wᵢ² across every NDArray leaf in the tree. The square
-- | root of this is the global L2 norm of the parameter vector;
-- | dividing by `countParams` gives mean-square magnitude. Useful for
-- | sanity-checking a freshly-loaded checkpoint (random-init ≈ 0.02
-- | per param; trained checkpoints typically 0.001–0.05).
sumSquaredL2
  :: forall r
   . HFoldl SumSquaredL2 (Effect Number) { | r } (Effect Number)
  => { | r }
  -> Effect Number
sumSquaredL2 r = hfoldl SumSquaredL2 (pure 0.0 :: Effect Number) r
