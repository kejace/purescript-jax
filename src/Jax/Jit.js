// `jit` with fusion barriers.
//
// jax-js fuses every elementwise op into its consumer unless the value
// reaches more than one materialised node (an output, a reduction, ...).
// A deep elementwise graph with shared subexpressions and few outputs
// therefore becomes one kernel whose expression *tree* duplicates every
// shared node: exponential in depth (a 256-op, 73-layer integer circuit
// exhausted 4 GB). A barrier marks a value to be materialised: it becomes
// an extra output of the traced function, which jax-js always writes to
// memory, and the wrapper disposes those extra outputs after each call.
import { jit } from "@jax-js/jax";

// f(barrier, x) -> out; `x` and `out` are jax-js pytrees (arrays, nested JS
// arrays or records of arrays). Returns a jitted (x) -> out.
export const jitWithBarriersImpl = (f) => {
  const g = jit((x) => {
    const extras = [];
    const barrier = (v) => { extras.push(v.ref); return v; };
    return [f(barrier, x), extras];
  });
  return (x) => {
    const [out, extras] = g(x);
    for (const e of extras) e.dispose();
    return out;
  };
};

// The barrier is rank-agnostic at runtime.
export const unsafeBarrier = (f) => f;
