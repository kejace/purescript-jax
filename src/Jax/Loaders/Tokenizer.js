import { tokenizers } from "@jax-js/loaders";

// jax-js's `getBpe` uses an OPFS-backed cache to fetch tiktoken vocabs.
// OPFS doesn't exist in Node, so we replicate `loadTiktokenBpe`'s logic
// using plain `fetch` + manual `BpeEncoding` construction.
//
// This is invariant across runtimes. In a browser we could fall back to
// jax-js's `getBpe` to get the OPFS-cached path; for v1 we use the same
// direct-fetch path everywhere.

const { BpeEncoding } = tokenizers;

const cl100kPattern =
  /'(?:[sdmtSDMT]|[lL]{2}|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s/gu;

const cl100kSpecialTokens = {
  "<|endoftext|>": 100257,
  "<|fim_prefix|>": 100258,
  "<|fim_middle|>": 100259,
  "<|fim_suffix|>": 100260,
  "<|endofprompt|>": 100276,
};

async function loadCl100kBase() {
  const url = "https://cdn.jsdelivr.net/npm/gpt-tokenizer@3.0.1/data/cl100k_base.tiktoken";
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch cl100k_base.tiktoken: ${response.status}`);
  }
  const data = new Uint8Array(await response.arrayBuffer());
  const encoder = new Map();
  const text = new TextDecoder().decode(data);
  for (const line of text.split("\n")) {
    if (!line) continue;
    const [token, rank] = line.split(/\s+/);
    const bytes = Uint8Array.from(atob(token), (c) => c.charCodeAt(0));
    let hex = "";
    for (let i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, "0");
    encoder.set(hex, parseInt(rank, 10));
  }
  return new BpeEncoding(encoder, cl100kSpecialTokens, cl100kPattern);
}

const defaultBpe = await loadCl100kBase();

export const defaultTokenizerImpl = defaultBpe;

export const encodeImpl = (tok, text) => tok.encode(text);
export const decodeImpl = (tok, tokens) => tok.decode(tokens);
