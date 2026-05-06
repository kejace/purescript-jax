import { sgd, adam, adamw, applyUpdates } from "@jax-js/optax";

// jax-js/optax represents an optimizer as a `GradientTransformation`:
// a record `{ init, update }` of pure functions. Callers carry the
// optimizer state (`OptState`) explicitly.
//
//   const opt = adam(1e-3);
//   const state = opt.init(params);
//   const [updates, state2] = opt.update(grads, state);
//   const newParams = applyUpdates(params, updates);

export const adamImpl = (lr) => adam(lr);
export const adamWImpl = (lr) => adamw(lr);
export const sgdImpl = (lr) => sgd(lr);

export const initImpl = (transformation, params) => transformation.init(params);

export const updateImpl = (transformation, grads, state) => {
  const r = transformation.update(grads, state);
  return { updates: r[0], state: r[1] };
};

export const applyUpdatesImpl = (params, updates) => applyUpdates(params, updates);

// Pytree-aware variants. Same JS shape, broader PS types.
export const initImpl_ = (transformation, params) => transformation.init(params);
export const updateImpl_ = (transformation, grads, state) => {
  const r = transformation.update(grads, state);
  return { updates: r[0], state: r[1] };
};
export const applyUpdatesImpl_ = (params, updates) => applyUpdates(params, updates);
