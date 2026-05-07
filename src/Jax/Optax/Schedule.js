// Bindings to upstream optax composition primitives.
//
// `scaleBySchedule(stepSizeFn)` returns a GradientTransformation whose
// `update(grads, state, params)` scales every leaf of `grads` by
// `stepSizeFn(currentStep)` before returning. The step counter lives
// in the optimizer state; the user only ever sees a "Transformation"
// from PureScript.
//
// `chain(...transforms)` composes a sequence: each transform's update
// runs against the output of the previous, threading state through.
// This is how you stack `adam(lr) ∘ scaleBySchedule(decay)` etc.
//
// Both functions exist in @jax-js/optax already; we're just exposing
// them under the `Optax.Schedule` PS module.

import { scaleBySchedule, chain } from "@jax-js/optax";

export const scaleByScheduleImpl = (schedule) => scaleBySchedule(schedule);
export const chainImpl = (transforms) => chain(...transforms);
