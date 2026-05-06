-- | First-class optics over `ModelWeights` and its sub-records.
-- |
-- | Why these aren't a typeclass-derived thing: PureScript's
-- | `Data.Lens.Record.prop` is the standard way to project a record
-- | field as a `Lens'`, and the boilerplate per field is one line. The
-- | gain is at the *use* site — composing path lenses with `<<<`
-- | gives `view` / `set` / `over` / `traverseOf` for free, and the
-- | composition checks at the type level.
-- |
-- | `_layer i` is an `AffineTraversal'` (not a `Lens'`) because the
-- | array index might be out of bounds. Use `preview` to read, `over`
-- | to modify; both are no-ops if the index doesn't exist (the array
-- | is returned unchanged).
-- |
-- | Example:
-- |
-- |     -- Pull the embedding tensor out
-- |     let emb = view _embedding weights
-- |
-- |     -- Compute the L2 norm of just one layer's MLP weights
-- |     case preview (_layer 3 <<< _mlp) weights of
-- |       Just mlpW -> Pytree.sumSquaredL2 mlpW
-- |       Nothing -> pure 0.0
-- |
-- |     -- Apply a transformation to a specific tensor
-- |     let weights' = over (_layer 0 <<< _attn <<< _wq) freezeFn weights
-- |
-- | The composition rule: lenses compose with `<<<`. A path through
-- | a record has the same composition order as the field-access path
-- | you'd write by hand.
module Jax.Optics
  ( -- * ModelWeights fields
    _embedding
  , _layers
  , _finalNorm
  -- * LayerWeights fields
  , _attnNorm
  , _attn
  , _mlpNorm
  , _mlp
  -- * AttentionWeights fields
  , _wq
  , _wk
  , _wv
  , _wo
  -- * MLPWeights fields
  , _gateProj
  , _upProj
  , _downProj
  -- * Composite
  , _layer
  ) where

import Prelude

import Data.Lens (Lens')
import Data.Lens.AffineTraversal (AffineTraversal')
import Data.Lens.Index (ix)
import Data.Lens.Record (prop)
import Type.Proxy (Proxy(..))
import Jax.Core (D1, D2, NDArray)
import Jax.NN.Block (AttentionWeights, LayerWeights, MLPWeights, ModelWeights)

-- =============================================================================
-- ModelWeights
-- =============================================================================

_embedding :: Lens' ModelWeights (NDArray D2)
_embedding = prop (Proxy :: Proxy "embedding")

_layers :: Lens' ModelWeights (Array LayerWeights)
_layers = prop (Proxy :: Proxy "layers")

_finalNorm :: Lens' ModelWeights (NDArray D1)
_finalNorm = prop (Proxy :: Proxy "finalNorm")

-- =============================================================================
-- LayerWeights
-- =============================================================================

_attnNorm :: Lens' LayerWeights (NDArray D1)
_attnNorm = prop (Proxy :: Proxy "attnNorm")

_attn :: Lens' LayerWeights AttentionWeights
_attn = prop (Proxy :: Proxy "attn")

_mlpNorm :: Lens' LayerWeights (NDArray D1)
_mlpNorm = prop (Proxy :: Proxy "mlpNorm")

_mlp :: Lens' LayerWeights MLPWeights
_mlp = prop (Proxy :: Proxy "mlp")

-- =============================================================================
-- AttentionWeights
-- =============================================================================

_wq :: Lens' AttentionWeights (NDArray D2)
_wq = prop (Proxy :: Proxy "wq")

_wk :: Lens' AttentionWeights (NDArray D2)
_wk = prop (Proxy :: Proxy "wk")

_wv :: Lens' AttentionWeights (NDArray D2)
_wv = prop (Proxy :: Proxy "wv")

_wo :: Lens' AttentionWeights (NDArray D2)
_wo = prop (Proxy :: Proxy "wo")

-- =============================================================================
-- MLPWeights
-- =============================================================================

_gateProj :: Lens' MLPWeights (NDArray D2)
_gateProj = prop (Proxy :: Proxy "gateProj")

_upProj :: Lens' MLPWeights (NDArray D2)
_upProj = prop (Proxy :: Proxy "upProj")

_downProj :: Lens' MLPWeights (NDArray D2)
_downProj = prop (Proxy :: Proxy "downProj")

-- =============================================================================
-- Composite
-- =============================================================================

-- | Affine traversal to the i-th layer. `Lens'` would be wrong because
-- | the index might be out of bounds; `AffineTraversal'` represents
-- | "0-or-1 focus".
-- |
-- | Use `preview (_layer i)` to read (returns `Maybe LayerWeights`),
-- | `over (_layer i)` to modify (no-op if out of bounds), or compose
-- | with deeper lenses for path-based access:
-- |
-- |     preview (_layer 3 <<< _attn <<< _wq) weights :: Maybe (NDArray D2)
_layer :: Int -> AffineTraversal' ModelWeights LayerWeights
_layer i = _layers <<< ix i
