-- | High-level training-loop helpers.
-- |
-- | Wraps the canonical "compute grad → optimizer update → return new
-- | state" pattern behind a single function. Hides the
-- | `EffectFn1`/`runEffectFn1` ceremony, the `valueAndGradT + updateT
-- | + applyUpdatesT` triple, and the loss-readback boilerplate.
-- |
-- | What you write:
-- |
-- |     vagFn <- valueAndGradT lossFn
-- |     state <- pure { weights: w0, optState: s0, lastLoss: 0.0/0.0 }
-- |     finalState <- stepN opt vagFn 1000 reportLoss state
-- |
-- | What this replaces (per step, was ~10 lines including dispose dance):
-- |
-- |     vag <- runEffectFn1 vagFn weights
-- |     lossF <- toJs vag.value
-- |     dispose vag.value
-- |     { updates, state: newState } <- Optax.updateT opt vag.grad state
-- |     newWeights <- Optax.applyUpdatesT weights updates
-- |     pure { weights: newWeights, optState: newState
-- |          , lastLoss: asNumber lossF }
-- |
-- | Why this lives outside `Jax.NN.Train`: `Jax.NN.Train` defines the
-- | (model-specific) cross-entropy loss; this module is about the
-- | (model-agnostic) optimizer loop scaffolding. They sit on opposite
-- | sides of the autograd boundary.
module Jax.Train
  ( TrainState
  , initial
  , step
  , stepN
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Jax.Coerce (asNumber)
import Jax.Core (D1, NDArray, dispose, toJs)
import Jax.Optax as Optax

-- | Training-loop state. Carries the previous step's gradient so the
-- | next step can `updateT(grad) → applyUpdatesT → vagFn(newWeights)`
-- | without consuming `weights` twice (jax-js's `valueAndGrad` and
-- | `applyUpdatesT` both want to take ownership; staggering them like
-- | this lets each call see a fresh tensor tree).
-- |
-- | Threaded by re-binding; nothing mutates in place.
type TrainState p =
  { weights  :: p
  , optState :: Optax.OptState
  , grad     :: p
  , lastLoss :: Number
  }

-- | Bootstrap a `TrainState` from the freshly-built parameters and a
-- | pre-built `optState` (you'll have called `Optax.initT` on a
-- | ref-bumped copy of weights yourself, since `initT` consumes its
-- | argument). Runs the model once to compute the initial gradient
-- | and loss, so that the very first call to `step` has something to
-- | feed into `updateT`.
-- |
-- | Why this exists: `valueAndGrad`'s wrapped function traces on its
-- | first call. We force that here, before `step` enters the loop,
-- | so the trace is cleanly tied to the original parameters and not
-- | to whatever's left after `step`'s own consumption pattern.
initial
  :: forall p
   . EffectFn1 p { value :: NDArray D1, grad :: p }
  -> p
  -> Optax.OptState
  -> Effect (TrainState p)
initial vagFn weights optState = do
  vag <- runEffectFn1 vagFn weights
  lossF <- toJs vag.value
  dispose vag.value
  pure
    { weights, optState, grad: vag.grad, lastLoss: asNumber lossF }

-- | One iteration. Order is load-bearing:
-- |
-- |   1. updateT consumes the gradient from the previous step.
-- |   2. applyUpdatesT consumes the previous weights + the updates,
-- |      returning fresh `newWeights`.
-- |   3. vagFn is called on `newWeights`. Its result is the gradient
-- |      that the NEXT step's updateT will consume.
-- |
-- | Each tensor is consumed exactly once per step. The single host
-- | round-trip is the loss readback (`toJs`).
step
  :: forall p
   . Optax.Transformation
  -> EffectFn1 p { value :: NDArray D1, grad :: p }
  -> TrainState p
  -> Effect (TrainState p)
step opt vagFn s = do
  { updates, state: newState } <- Optax.updateT opt s.grad s.optState
  newWeights <- Optax.applyUpdatesT s.weights updates
  vag <- runEffectFn1 vagFn newWeights
  lossF <- toJs vag.value
  dispose vag.value
  pure
    { weights: newWeights
    , optState: newState
    , grad: vag.grad
    , lastLoss: asNumber lossF
    }

-- | Run `step` `n` times, calling `report stepIdx state` after each
-- | iteration (1-indexed). Tail-recursive.
-- |
-- | Idiomatic:
-- |
-- |     stepN opt vagFn 1000 (\i s -> when (i `mod` 50 == 0) (log ...)) initial
stepN
  :: forall p
   . Optax.Transformation
  -> EffectFn1 p { value :: NDArray D1, grad :: p }
  -> Int
  -> (Int -> TrainState p -> Effect Unit)
  -> TrainState p
  -> Effect (TrainState p)
stepN opt vagFn nSteps report = go 0
  where
  go i s
    | i >= nSteps = pure s
    | otherwise = do
        s' <- step opt vagFn s
        report (i + 1) s'
        go (i + 1) s'
