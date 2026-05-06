// Worker-side FFI: self.onmessage / self.postMessage.
// Both bound as EffectFn1 — the JS form is `(arg) => result` (no extra
// thunk wrapper).

export const selfOnMessageImpl = (cb) => {
  self.onmessage = (e) => cb(e.data)();
};

export const selfPostMessageImpl = (msg) => { self.postMessage(msg); };

export const performanceNowImpl = () => performance.now();

import { defaultDevice } from "@jax-js/jax";

export const trySetDeviceImpl = (name) => {
  try {
    defaultDevice(name);
    return true;
  } catch (_e) {
    return false;
  }
};

export const hasWebGpuImpl = () =>
  typeof navigator !== "undefined" && "gpu" in navigator;

export const arrayLengthImpl = (xs) => xs.length;

export const replaceImpl = (search) => (replacement) => (input) =>
  input.split(search).join(replacement);

// String slicing for the streaming-decode delta calculation. Counts
// UTF-16 code units (matching PureScript's Data.String.length / take).
export const dropImpl = (n) => (s) => s.slice(n);
