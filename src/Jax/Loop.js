// Loops for traced code. jax-js has no `while_loop` / `scan`; these unroll
// at trace time, so under `jit` (or `jitWithBarriers`) a loop becomes one
// fused graph instead of one dispatch per iteration.
import { numpy as np } from "@jax-js/jax";

// Run `body` on the lanes where `cond` holds, at most n times. State is an
// array of arrays; cond(state) gives a bool array (per lane); a lane whose
// condition is false keeps its state (np.where), so lanes finish at
// different iterations without diverging. Consumes `init`.
// With `barrier` (from Jax.Jit.jitWithBarriers), the state is materialised
// every `every` iterations: an unrolled loop reads its state several times
// per step, so without barriers jax-js would fuse the whole loop into one
// expression tree that grows exponentially with the iteration count.
export const whileBoundedImpl = (n, cond, body, init, barrier, every) => {
  let s = init;
  for (let i = 0; i < n; i++) {
    if (barrier && every > 0 && i > 0 && i % every === 0) s = s.map((x) => barrier(x));
    const c = cond(s.map((x) => x.ref));
    const next = body(s.map((x) => x.ref));
    s = s.map((x, k) => np.where(k < s.length - 1 ? c.ref : c, next[k], x));
  }
  return s;
};

// The condition after the loop (to see which lanes are still running).
export const anyImpl = (a) => np.any(a);

// for i in [0, n): state = f(i, state).
export const foriImpl = (n, f, init, _unit) => {
  let s = init;
  for (let i = 0; i < n; i++) s = f(i, s);
  return s;
};

// scan over the leading axis of xs (static length n): carry, ys stacked.
export const scanImpl = (n, f, init, xs) => {
  let carry = init;
  const ys = [];
  for (let t = 0; t < n; t++) {
    const r = f(carry, xs.ref.slice(t));
    carry = r.carry;
    ys.push(r.y);
  }
  xs.dispose();
  return { carry, ys: np.stack(ys, 0) };
};

// Unbounded: run a (jitted) bounded step until no lane is running.
// step(state) -> state; running(state) -> bool array. Resolves to the state.
export const whileLoopAffImpl = async (step, running, init, maxChunks) => {
  let s = init;
  for (let k = 0; k < maxChunks; k++) {
    s = step(s);
    const live = await np.any(running(s.map((x) => x.ref))).data();
    if (!live[0]) break;
  }
  return s;
};
