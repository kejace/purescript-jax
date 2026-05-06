import { grad, valueAndGrad, jit, vmap, numpy as np } from "@jax-js/jax";

// Convenience: a sum-of-squares loss exposed as a pure JS function so
// JAX's autodiff trace doesn't see PureScript's mkEffectFn1 wrapper.
// Used by the Phase 4 training demo.
export const sumSquareLossImpl = (x) => np.square(x).sum();

// Pytree variant: sum(a²) + sum(b²) over a `{a, b}` record.
// Demonstrates jax-js's autodiff over nested records.
export const sumSquareTreeLossImpl = (p) =>
  np.square(p.a).sum().add(np.square(p.b).sum());

// jax-js inspects fn.length during tracing, so we must pass through real
// uncurried JS functions (which is what EffectFn1/2/3 already are at runtime).

export const gradImpl = (fn) => grad(fn);

export const valueAndGradImpl = (fn) => {
  const vag = valueAndGrad(fn);
  // Return a 1-arg JS fn that produces a {value, grad} record (PureScript
  // record reps natively as a JS object, so no constructor needed).
  return (x) => {
    const r = vag(x);
    // jax-js returns [value, grad] as a JS array.
    return { value: r[0], grad: r[1] };
  };
};

export const jitImpl = (fn) => jit(fn);

export const vmapImpl = (fn) => vmap(fn);

// Pytree-aware variants. Same JS shape, broader PS types.
export const gradImpl_ = (fn) => grad(fn);

export const valueAndGradImpl_ = (fn) => {
  const vag = valueAndGrad(fn);
  return (x) => {
    const r = vag(x);
    return { value: r[0], grad: r[1] };
  };
};
