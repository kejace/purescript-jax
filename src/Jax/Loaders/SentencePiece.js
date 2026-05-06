import { tokenizers } from "@jax-js/loaders";

const { Unigram } = tokenizers;

// SentencePiece Unigram is the tokenizer family used by Llama, Gemma,
// T5, etc. Binary `.model` files are the standard format.
//
// Construction from raw bytes is synchronous; the caller fetches the
// file via their preferred mechanism (e.g. `fetch(url).arrayBuffer()`)
// and passes the bytes here.

export const fromBinaryImpl = (bytes) => Unigram.fromBinary(bytes);
export const encodeImpl = (sp, text) => sp.encode(text);
export const decodeImpl = (sp, tokens) => sp.decode(tokens);
export const bosTokenImpl = (sp) => sp.bosToken;
export const eosTokenImpl = (sp) => sp.eosToken;
export const vocabSizeImpl = (sp) => sp.vocabSize;
