// Foreign-to-typed shims for jax-js inspection results.
//
// `Jax.Core.toJs` returns a `Foreign` whose JS-runtime shape depends on
// the rank/dtype of the producing NDArray:
//   - rank-0          → JS Number (or boolean)
//   - rank-1 float    → JS Array<number>
//   - rank-1 int32    → JS Array<number>  (still numbers; we widen to PS Int)
//   - rank-2 float    → JS Array<Array<number>>
//   - higher / mixed  → caller's responsibility
//
// Each export documents the precondition; the shim itself is a no-op
// reinterpretation. The single point of `unsafeCoerce`-equivalence is
// confined to this file so future fp-police audits can lock the
// pattern down via a project rule.

export const asIntImpl        = (x) => x | 0;             // truncate-toward-zero
export const asNumberImpl     = (x) => x;
export const asArray1DImpl    = (x) => x;
export const asArray1DIntImpl = (x) => x;
export const asArray2DImpl    = (x) => x;
