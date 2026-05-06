import { random } from "@jax-js/jax";

// random.key(seed) → an Array (NDArray) representing a PRNG state.
// random.split(key, num) → array of `num` derived keys.
// random.categorical(key, logits, opts) → sampled indices.

export const mkKeyImpl = (seed) => random.key(seed);

export const splitKey2Impl = (key) => {
  // random.split(key, 2) → shape [2, keyLen]. To get two 1D keys we
  // pass a number index (drops the indexed axis), not a range (keeps it).
  // The first call needs to bump split's refcount so the second has
  // something to consume.
  const split = random.split(key, 2);
  const a = split.ref.slice(0);  // [keyLen]
  const b = split.slice(1);      // [keyLen]; consumes split
  return { a, b };
};

// random.categorical samples one int per row of logits along `axis`.
// For 1D logits with axis=-1, output is rank 0 (a scalar int32 NDArray).
export const sampleCategoricalImpl = (key, logits) =>
  random.categorical(key, logits);
