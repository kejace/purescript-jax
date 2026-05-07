-- | Learning-rate schedules for `Jax.Optax`.
-- |
-- | A `Schedule` is just `Int -> Number` — given the current step
-- | count, return the multiplier in `[0, 1]` to apply to the gradient.
-- | Combine with `Optax.adam` (or any other transformation) via
-- | `chain`:
-- |
-- |     opt <- chain
-- |       [ Optax.adam 1.0e-3
-- |       , scaleBySchedule (linearDecay numSteps)
-- |       ]
-- |
-- | Backed by upstream `@jax-js/optax`'s `scaleBySchedule` and `chain`
-- | primitives — these are pre-existing Optax composition tools, not
-- | a new abstraction we're inventing.
module Jax.Optax.Schedule
  ( Schedule
  , linearDecay
  , cosineDecay
  , constant
  , scaleBySchedule
  , chain
  ) where

import Prelude

import Data.Int (toNumber)
import Data.Number (cos, pi)
import Effect (Effect)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Jax.Optax (Transformation)

-- | A learning-rate schedule. Given the step index (0-based, advanced
-- | by upstream once per call to the wrapped optimizer's `update`),
-- | return the multiplier to scale gradient updates by.
-- |
-- | Implemented as a function the JS side can call directly — no
-- | wrapping at the FFI boundary, so this is `Int -> Number` not
-- | `EffectFn1 Int Number`.
type Schedule = Int -> Number

-- | Linear decay from 1.0 to 0.0 over `numSteps`. After `numSteps`
-- | the schedule continues at 0.0 (no negative multipliers).
-- |
-- |     linearDecay 1000 0    == 1.0
-- |     linearDecay 1000 500  == 0.5
-- |     linearDecay 1000 1000 == 0.0
-- |     linearDecay 1000 2000 == 0.0  (clamped)
linearDecay :: Int -> Schedule
linearDecay numSteps step =
  let frac = toNumber step / toNumber numSteps
  in max 0.0 (1.0 - frac)

-- | Cosine decay from 1.0 to 0.0 over `numSteps`. The classic
-- | "warm-then-decay" curve without the warmup half. After `numSteps`
-- | the schedule stays at 0.0.
-- |
-- |     cosineDecay 1000 0    == 1.0
-- |     cosineDecay 1000 500  ≈ 0.5
-- |     cosineDecay 1000 1000 == 0.0
cosineDecay :: Int -> Schedule
cosineDecay numSteps step
  | step >= numSteps = 0.0
  | otherwise =
      let t = toNumber step / toNumber numSteps
      in 0.5 * (1.0 + cos (pi * t))

-- | A constant schedule. Useful as the identity element when chaining
-- | conditional schedules; no-op vs. just calling `Optax.adam` directly.
constant :: Number -> Schedule
constant c _ = c

-- | Scale gradients by a step-indexed schedule. Wraps upstream's
-- | `optax.scaleBySchedule`. The returned transformation has a
-- | one-int internal state (the current step counter) — it advances
-- | each time you call `Optax.updateT` against it.
foreign import scaleByScheduleImpl :: EffectFn1 Schedule Transformation

scaleBySchedule :: Schedule -> Effect Transformation
scaleBySchedule = runEffectFn1 scaleByScheduleImpl

-- | Compose a sequence of transformations. The result's `init` /
-- | `update` walk every member in order. Wraps upstream's
-- | `optax.chain` (variadic; we accept an array). Equivalent to
-- | nesting in PyTorch with `torch.optim.lr_scheduler` wrappers.
foreign import chainImpl :: EffectFn1 (Array Transformation) Transformation

chain :: Array Transformation -> Effect Transformation
chain = runEffectFn1 chainImpl
