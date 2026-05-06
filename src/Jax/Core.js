// FFI shim for @jax-js/jax.
//
// jax-js semantics: every operation CONSUMES its arguments (refcount -1).
// `.ref` is a *property* (not method) that returns the same object with
// refcount +1. To keep a tensor alive across multiple ops, call refImpl
// before passing it in.
//
// Calling convention is op-specific: some are methods on the Array class
// (`a.add(b)`, `a.transpose()`), others are free functions on the `numpy`
// namespace (`np.matmul(a, b)`, `np.sqrt(a)`). Each shim below picks the
// correct form per the upstream API surface (verified against the
// installed jax-js dist).
//
// Inspection methods (`.js()`, `.dataSync()`) internally consume the
// receiver via `#dataInline`. We bump `.ref` before calling them so they
// read as non-consuming from the PureScript side.

import { numpy as np, nn, lax, devicePut, defaultDevice as setDefaultDeviceJs, init as initJs } from "@jax-js/jax";

// jax-js's init() returns Promise<Device[]>. We use ESM top-level await so
// that any code reaching into this module sees an initialized backend.
// Supported by Node ≥ 14.8, Bun, and modern bundlers (Vite).
await initJs();

// =============================================================================
// Leak canary
// =============================================================================
//
// FinalizationRegistry fires (best-effort, not guaranteed) when an object
// is garbage-collected. We use it as a *diagnostic*: if a tensor reaches
// GC without being disposed, log a warning. This is opt-in via
// `globalThis.JAX_DEBUG = true` to avoid spurious warnings in shipped
// builds (FR firing for tensors held inside jax-js's own caches looks
// like a leak from outside).
const __JAX_DEBUG =
  typeof globalThis !== "undefined" && globalThis.JAX_DEBUG === true;

const __jaxFinalizers =
  __JAX_DEBUG && typeof FinalizationRegistry !== "undefined"
    ? new FinalizationRegistry((label) => {
        // eslint-disable-next-line no-console
        console.warn(`[jax-leak] tensor reached GC undisposed: ${label}`);
      })
    : null;

const __track = (ndarr, label) => {
  if (__jaxFinalizers) __jaxFinalizers.register(ndarr, label, ndarr);
  return ndarr;
};
const __untrack = (ndarr) => {
  if (__jaxFinalizers) __jaxFinalizers.unregister(ndarr);
};

// --- Backend ---------------------------------------------------------------
// `init` is now redundant (top-level await already ran), but kept as a
// no-op for the PureScript surface. Re-running is idempotent.
export const initImpl = () => { initJs(); };
export const setDefaultDeviceImpl = (name) => setDefaultDeviceJs(name);
export const devicePutImpl = (a, name) => devicePut(a, name);

// --- Reference management --------------------------------------------------
export const refImpl = (a) => a.ref;
export const disposeImpl = (a) => { __untrack(a); a.dispose(); };
export const refCountImpl = (a) => a.refCount;

// --- Constructors ----------------------------------------------------------
export const array1DImpl = (xs) => __track(np.array(xs), "array1D");
export const arrayInt1DImpl = (xs) => __track(np.array(xs, { dtype: "int32" }), "arrayInt1D");
export const arrayNestedImpl = (nested) => __track(np.array(nested), "arrayNested");
export const zerosImpl = (shape) => __track(np.zeros(shape), `zeros[${shape}]`);
export const onesImpl = (shape) => __track(np.ones(shape), `ones[${shape}]`);
export const arangeImpl = (start, stop, step) => __track(np.arange(start, stop, step), "arange");

// numpy-style repeat-along-axis: each element is repeated `n` times
// consecutively along `axis` (i.e. torch.repeat_interleave). Distinct
// from `tile`, which replicates the whole pattern. We need this for
// GQA: HF Llama pairs q_head_i ↔ kv_head_(i // G), which corresponds to
// expanding kv heads via repeat-interleave (k0,k0,k0,k1,k1,k1,...) NOT
// tile (k0,k1,...,k0,k1,...). jax-js's `nn.dotProductAttention` does
// the latter internally — pre-expand here so its internal tile is
// skipped (N == K → no expansion).
export const repeatAxisImpl = (a, n, axis) => __track(np.repeat(a, n, axis), `repeatAxis[${n}]`);
export const linspaceImpl = (start, stop, num) => __track(np.linspace(start, stop, num), "linspace");

// --- Binary ops (consume both args) ----------------------------------------
// add/mul/sub are class methods; matmul is a free function on np.
export const addImpl = (a, b) => a.add(b);
export const mulImpl = (a, b) => a.mul(b);
export const subImpl = (a, b) => a.sub(b);
export const matmulImpl = (a, b) => np.matmul(a, b);

// --- Unary math (consume arg) ----------------------------------------------
// jax-js does not expose rsqrt directly — synthesize via reciprocal∘sqrt.
// Method-style ops: transpose, sum, mean.
// np-level ops: sigmoid, square, sqrt, sin, tanh, reciprocal.
export const rsqrtImpl = (a) => np.reciprocal(np.sqrt(a));
export const meanImpl = (a) => a.mean();
export const meanAxisKeepImpl = (a, axis) => a.mean(axis, { keepdims: true });
export const sumImpl = (a) => a.sum();
export const sumAxisKeepImpl = (a, axis) => a.sum(axis, { keepdims: true });
// Scalar broadcast variants — jax-js's a.add(other)/a.mul(other) accept
// number for `other` (TracerValue includes primitives).
export const addScalarImpl = (a, n) => a.add(n);
export const mulScalarImpl = (a, n) => a.mul(n);
export const transposeImpl = (a) => a.transpose();
export const sigmoidImpl = (a) => np.sigmoid(a);
export const siluImpl = (a) => nn.silu(a);
export const squareImpl = (a) => np.square(a);
export const sqrtImpl = (a) => np.sqrt(a);
export const sinImpl = (a) => np.sin(a);
export const tanhImpl = (a) => np.tanh(a);

// --- Shape ops -------------------------------------------------------------
export const reshapeImpl = (a, shape) => a.reshape(shape);
// jax-js Array.slice is variadic with one selector per axis. Each can be:
//   number (drops axis), [] (full axis), [i] (i..end), [i,j] (range),
//   null (insert size-1 axis), Tracer (advanced indexing).
// Trailing axes default to full-keep.
export const sliceImpl = (a, start, limit) => a.slice([start, limit]);
export const sliceAxisImpl = (a, axis, start, end) => {
  const n = a.shape.length;
  const ax = axis < 0 ? n + axis : axis;
  const args = [];
  for (let i = 0; i <= ax; i++) {
    args.push(i === ax ? [start, end] : []);
  }
  return a.slice(...args);
};
export const sliceLastAxisImpl = (a, start, end) => {
  const n = a.shape.length;
  const args = [];
  for (let i = 0; i < n - 1; i++) args.push([]);
  args.push([start, end]);
  return a.slice(...args);
};
// concatenate is np-level; default axis 0. Add an axis-aware variant if needed.
export const concatImpl = (arrs) => np.concatenate(arrs, 0);
export const concatAxisImpl = (arrs, axis) => np.concatenate(arrs, axis);

// --- Indexing / gather -----------------------------------------------------
// np.take(a, indices, axis): consumes both a and indices.
export const takeImpl = (a, indices, axis) => np.take(a, indices, axis);

// --- Reduction along axis to int32 -----------------------------------------
export const argmaxImpl = (a, axis) => np.argmax(a, axis);
export const argminImpl = (a, axis) => np.argmin(a, axis);

// --- Top-k -----------------------------------------------------------------
// lax.topK(a, k, axis) returns [values, indices]. We unwrap into a record.
export const topKImpl = (a, k, axis) => {
  const r = lax.topK(a, k, axis);
  return { values: r[0], indices: r[1] };
};

// --- Softmax / cumsum / one-hot --------------------------------------------
export const softmaxImpl = (a, axis) => nn.softmax(a, axis);
export const logSoftmaxImpl = (a, axis) => nn.logSoftmax(a, axis);
export const oneHotImpl = (idx, n) => nn.oneHot(idx, n);
export const cumsumImpl = (a, axis) => np.cumsum(a, axis);
export const sumAxisImpl = (a, axis) => a.sum(axis);

// --- Inspection -----------------------------------------------------------
// `shape` is a property — non-consuming.
// `js()` / `dataSync()` are method calls and *do* consume the receiver
// internally (see Array#dataInline → this.dispose() in jax-js source). We
// auto-bump via `.ref` so these read like ordinary inspectors from the
// PureScript side, leaving the caller's reference intact.
export const shapeImpl = (a) => a.shape;
export const dataSyncImpl = (a) => a.ref.dataSync();
export const jsImpl = (a) => a.ref.js();
