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
  , step
  , stepN
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Jax.Coerce (asNumber)
import Jax.Core (D1, NDArray, dispose, toJs)
import Jax.Optax as Optax

-- | Mutable-only-by-replacement training state. Threaded through `step`
-- | by re-binding; nothing mutates in place. `lastLoss` is for
-- | reporting only; the real source of truth is the next gradient
-- | computation.
type TrainState p =
  { weights  :: p
  , optState :: Optax.OptState
  , lastLoss :: Number
  }

-- | One iteration of the standard autograd training loop:
-- |
-- |   1. forward the loss + accumulate gradients (one call to the
-- |      jit'd valueAndGrad function),
-- |   2. read the scalar loss back to the host,
-- |   3. dispose the loss tensor,
-- |   4. ask the optimizer for parameter updates,
-- |   5. apply them.
-- |
-- | The valueAndGrad function is passed in (not constructed here) so
-- | the caller can build it once outside the loop and reuse the jit
-- | trace across all iterations.
-- |
-- | Output gradient `vag.grad` is consumed by `updateT`, so the only
-- | tensor we explicitly dispose is the scalar loss.
step
  :: forall p
   . Optax.Transformation
  -> EffectFn1 p { value :: NDArray D1, grad :: p }
  -> TrainState p
  -> Effect (TrainState p)
step opt vagFn s = do
  vag <- runEffectFn1 vagFn s.weights
  lossF <- toJs vag.value
  dispose vag.value
  { updates, state: newState } <- Optax.updateT opt vag.grad s.optState
  newWeights <- Optax.applyUpdatesT s.weights updates
  pure
    { weights: newWeights
    , optState: newState
    , lastLoss: asNumber lossF
    }

-- | Run `step` `n` times, calling `report stepIdx state` after each
-- | iteration (1-indexed). Idiomatic uses:
-- |
-- |     stepN opt vagFn 1000 (\i s -> when (i `mod` 50 == 0) (log ...)) initial
-- |
-- | Tail-recursive (PureScript's compiler does TCO on this shape).
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
