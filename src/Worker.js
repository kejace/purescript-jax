// Worker-side FFI: self.onmessage / self.postMessage.
//
// Wire format: JSON strings (the PS-side codec handles ser/de).

import { defaultDevice, init } from "@jax-js/jax";

// jax-js's `defaultDevice("wasm")` flips a flag but does *not* warm up
// the backend itself. Hitting an op afterwards then blocks waiting for
// the wasm module to load — which on the worker side manifests as a
// generate request that never produces tokens. Top-level `await init()`
// runs once at module load and gives us a hot wasm/webgpu pool before
// any message handler is even installed.
const _initializedDevices = await init();
console.log(`[jax-js] backends initialized:`, _initializedDevices);

export const selfOnMessageImpl = (cb) => {
  self.onmessage = (e) => cb(e.data)();
};

export const selfPostMessageImpl = (msg) => { self.postMessage(msg); };

export const performanceNowImpl = () => performance.now();

export const trySetDeviceImpl = (name) => {
  try {
    if (!_initializedDevices.includes(name)) return false;
    defaultDevice(name);
    return true;
  } catch (_e) {
    return false;
  }
};

export const hasWebGpuImpl = () =>
  typeof navigator !== "undefined" && "gpu" in navigator;

export const arrayLengthImpl = (xs) => xs.length;

export const replaceImpl = (search, replacement, input) =>
  input.split(search).join(replacement);

// String slicing for the streaming-decode delta calculation. Counts
// UTF-16 code units (matching PureScript's Data.String.length / take).
export const dropImpl = (n) => (s) => s.slice(n);
