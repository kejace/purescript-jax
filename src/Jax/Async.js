// Async readback (no OffscreenCanvas needed): jax-js's synchronous reads
// (`dataSync`, `js`) only work on WebGPU in browsers, through
// OffscreenCanvas; Deno and workers without it need the async forms.
//
// `data()` consumes its receiver, so each shim takes a `.ref` first:
// reads are non-consuming from the PureScript side, like `Jax.Core`'s.
export const dataAffImpl = (a) => a.ref.data().then((d) => Array.from(d));
export const toJsAffImpl = (a) => a.ref.jsAsync();
// `blockUntilReady` does not consume (it returns the same array).
export const blockUntilReadyImpl = (a) => a.blockUntilReady().then(() => undefined);
