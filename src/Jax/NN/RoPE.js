import { numpy as np } from "@jax-js/jax";

// Precompute RoPE sin/cos tables in one JS pass and ship the result as a
// pair of NDArrays. Done in JS rather than via tensor ops so we avoid an
// outer-product/broadcast dance and only allocate the two final tensors.
//
// Input:
//   dim       - the per-head dimension (must be even)
//   maxSeqLen - number of positions to precompute
//   theta     - base of the geometric frequency progression (e.g. 10000)
// Output:
//   { cos, sin } - NDArray D2 each of shape [maxSeqLen, dim/2], float32.
export const precomputeRoPEImpl = (dim, maxSeqLen, theta) => {
  const halfDim = dim >> 1;
  const total = maxSeqLen * halfDim;
  const cosArr = new Array(total);
  const sinArr = new Array(total);
  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const freq = 1.0 / Math.pow(theta, (2 * i) / dim);
      const angle = pos * freq;
      const idx = pos * halfDim + i;
      cosArr[idx] = Math.cos(angle);
      sinArr[idx] = Math.sin(angle);
    }
  }
  const cosFlat = np.array(cosArr, { dtype: "float32" });
  const sinFlat = np.array(sinArr, { dtype: "float32" });
  return {
    cos: cosFlat.reshape([maxSeqLen, halfDim]),
    sin: sinFlat.reshape([maxSeqLen, halfDim]),
  };
};
