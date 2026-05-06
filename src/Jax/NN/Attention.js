import { nn } from "@jax-js/jax";

// jax-js's nn.dotProductAttention computes softmax((Q @ K^T) / sqrt(d)) @ V
// with optional causal masking. Q is [B?, L, N, H], K/V are [B?, S, K, H]
// (K may differ from N — this is GQA support: fewer KV heads than Q heads).
//
// Consumes q, k, v (all three pass through opts); options object is plain JS
// data (no NDArrays inside) so no extra refcount handling is needed there.
export const dotProductAttentionImpl = (q, k, v, isCausal) =>
  nn.dotProductAttention(q, k, v, { isCausal });
