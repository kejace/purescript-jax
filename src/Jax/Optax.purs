module Jax.Optax
  ( Transformation
  , OptState
  , adam
  , adamW
  , sgd
  , init
  , update
  , applyUpdates
  , initT
  , updateT
  , applyUpdatesT
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried
  ( EffectFn1
  , EffectFn2
  , EffectFn3
  , runEffectFn1
  , runEffectFn2
  , runEffectFn3
  )
import Jax.Core (NDArray)

-- | Opaque jax-js/optax optimizer (`{ init, update }` pair). Pure data
-- | from the user's perspective; lifetime managed by JS GC, not refcount.
foreign import data Transformation :: Type

-- | Opaque optimizer state (e.g. Adam's first/second moments). Carries
-- | NDArrays internally — disposal is handled inside the transformation.
foreign import data OptState :: Type

foreign import adamImpl :: EffectFn1 Number Transformation
foreign import adamWImpl :: EffectFn1 Number Transformation
foreign import sgdImpl :: EffectFn1 Number Transformation

foreign import initImpl
  :: forall d. EffectFn2 Transformation (NDArray d) OptState

foreign import updateImpl
  :: forall d
   . EffectFn3
       Transformation
       (NDArray d)
       OptState
       { updates :: NDArray d, state :: OptState }

foreign import applyUpdatesImpl
  :: forall d. EffectFn2 (NDArray d) (NDArray d) (NDArray d)

-- | Adam optimizer with the given learning rate.
adam :: Number -> Effect Transformation
adam = runEffectFn1 adamImpl

-- | AdamW (Adam with decoupled weight decay).
adamW :: Number -> Effect Transformation
adamW = runEffectFn1 adamWImpl

-- | Plain SGD with the given learning rate.
sgd :: Number -> Effect Transformation
sgd = runEffectFn1 sgdImpl

-- | Initialize the optimizer state from the parameter shape.
init :: forall d. Transformation -> NDArray d -> Effect OptState
init = runEffectFn2 initImpl

-- | Compute parameter updates and the next optimizer state from the
-- | gradients and current state.
update
  :: forall d
   . Transformation
  -> NDArray d
  -> OptState
  -> Effect { updates :: NDArray d, state :: OptState }
update = runEffectFn3 updateImpl

-- | Apply updates to params (typically `params + updates`).
applyUpdates
  :: forall d
   . NDArray d
  -> NDArray d
  -> Effect (NDArray d)
applyUpdates = runEffectFn2 applyUpdatesImpl

-- =============================================================================
-- Pytree-aware variants
-- =============================================================================
--
-- jax-js/optax accepts any `JsTree<numpy.Array>` (nested records/arrays
-- of NDArrays) for params/grads. The PS bindings below are typed over
-- arbitrary `a` so callers can plug in records like `ModelWeights`.

foreign import initImpl_ :: forall a. EffectFn2 Transformation a OptState

foreign import updateImpl_
  :: forall a
   . EffectFn3 Transformation a OptState { updates :: a, state :: OptState }

foreign import applyUpdatesImpl_ :: forall a. EffectFn2 a a a

-- | Pytree-shaped optimizer init.
initT :: forall a. Transformation -> a -> Effect OptState
initT = runEffectFn2 initImpl_

-- | Pytree-shaped optimizer update.
updateT
  :: forall a
   . Transformation
  -> a
  -> OptState
  -> Effect { updates :: a, state :: OptState }
updateT = runEffectFn3 updateImpl_

-- | Pytree-shaped applyUpdates.
applyUpdatesT :: forall a. a -> a -> Effect a
applyUpdatesT = runEffectFn2 applyUpdatesImpl_
