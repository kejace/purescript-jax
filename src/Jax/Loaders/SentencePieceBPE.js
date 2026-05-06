import { fromBinary } from "@bufbuild/protobuf";
import {
  ModelProtoSchema,
  ModelProto_SentencePiece_Type,
} from "sentencepiece-buf/model";

// SentencePiece BPE encoder/decoder. The Llama / Mistral / TinyLlama
// family ships `tokenizer.model` files in this format. jax-js bundles
// only Unigram (Viterbi-based) decoding, so we hand-roll BPE here from
// the proto schema.
//
// Algorithm (matches sentencepiece/src/bpe_model.cc):
//   1. Normalize: replace " " with U+2581 ("▁"), optionally prepend a
//      single ▁ (the "dummy prefix" — Llama-style).
//   2. Decompose into per-codepoint tokens. Codepoints absent from the
//      vocab fall back to byte-tokens `<0xHH>` when `byte_fallback` is
//      set, otherwise to `<unk>`.
//   3. Repeatedly merge the adjacent pair whose concatenated string is
//      a vocab piece with the highest score, until no such pair exists.

const SPIECE_UNDERLINE = "▁";

const TYPE_BYTE = ModelProto_SentencePiece_Type.BYTE;

class SentencePieceBPE {
  constructor(model) {
    this.unkId = model.trainerSpec?.unkId ?? 0;
    this.bosId = model.trainerSpec?.bosId ?? 1;
    this.eosId = model.trainerSpec?.eosId ?? 2;
    this.byteFallback = model.trainerSpec?.byteFallback ?? false;
    this.addDummyPrefix = model.normalizerSpec?.addDummyPrefix ?? true;
    this.removeExtraWhitespaces =
      model.normalizerSpec?.removeExtraWhitespaces ?? false;

    this.pieceToId = new Map();
    this.decoder = new Array(model.pieces.length);
    this.types = new Array(model.pieces.length);
    this.byteToId = new Map();

    for (let i = 0; i < model.pieces.length; i++) {
      const p = model.pieces[i];
      this.decoder[i] = p.piece;
      this.types[i] = p.type;
      // Multiple pieces could share a string in theory; first wins.
      if (!this.pieceToId.has(p.piece)) {
        this.pieceToId.set(p.piece, { id: i, score: p.score });
      }
      if (p.type === TYPE_BYTE) {
        const m = p.piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
        if (m) this.byteToId.set(m[1].toUpperCase(), i);
      }
    }
  }

  static fromBinary(bytes) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const model = fromBinary(ModelProtoSchema, data);
    return new SentencePieceBPE(model);
  }

  vocabSize() {
    return this.decoder.length;
  }

  normalize(text) {
    if (this.removeExtraWhitespaces) {
      text = text.replace(/\s+/g, " ").trim();
    }
    // Short-circuit on empty input *before* the dummy prefix runs;
    // otherwise we'd emit a stray ▁ token that the reference encoder
    // never produces.
    if (text.length === 0) return "";
    if (this.addDummyPrefix) text = " " + text;
    return text.replace(/ /g, SPIECE_UNDERLINE);
  }

  encode(text) {
    const norm = this.normalize(text);
    if (norm.length === 0) return [];

    const tokens = [];
    for (const ch of norm) {
      const piece = this.pieceToId.get(ch);
      if (piece !== undefined) {
        tokens.push({ id: piece.id, str: ch });
        continue;
      }
      if (this.byteFallback) {
        const utf8 = new TextEncoder().encode(ch);
        for (const b of utf8) {
          const hex = b.toString(16).toUpperCase().padStart(2, "0");
          const id = this.byteToId.get(hex);
          if (id !== undefined) {
            tokens.push({ id, str: `<0x${hex}>` });
          } else {
            tokens.push({ id: this.unkId, str: this.decoder[this.unkId] });
          }
        }
      } else {
        tokens.push({ id: this.unkId, str: this.decoder[this.unkId] });
      }
    }

    // Iterative merge. O(n * m) where m = #merges performed (≤ n).
    // Adequate for typical prompt sizes; a heap-based variant is a
    // straightforward perf upgrade if needed.
    while (tokens.length > 1) {
      let bestIdx = -1;
      let bestScore = -Infinity;
      let bestId = -1;
      for (let i = 0; i < tokens.length - 1; i++) {
        const merged = tokens[i].str + tokens[i + 1].str;
        const p = this.pieceToId.get(merged);
        if (p !== undefined && p.score > bestScore) {
          bestScore = p.score;
          bestIdx = i;
          bestId = p.id;
        }
      }
      if (bestIdx < 0) break;
      const mergedStr = tokens[bestIdx].str + tokens[bestIdx + 1].str;
      tokens.splice(bestIdx, 2, { id: bestId, str: mergedStr });
    }

    const ids = new Array(tokens.length);
    for (let i = 0; i < tokens.length; i++) ids[i] = tokens[i].id;
    return ids;
  }

  decode(ids) {
    const parts = [];
    let byteBuf = [];
    const decoder = new TextDecoder("utf-8", { fatal: false });

    const flushBytes = () => {
      if (byteBuf.length === 0) return;
      parts.push(decoder.decode(new Uint8Array(byteBuf)));
      byteBuf = [];
    };

    for (const id of ids) {
      if (id < 0 || id >= this.decoder.length) {
        flushBytes();
        continue;
      }
      const piece = this.decoder[id];
      const t = this.types[id];
      if (t === TYPE_BYTE) {
        const m = piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
        if (m) {
          byteBuf.push(parseInt(m[1], 16));
          continue;
        }
      }
      flushBytes();
      // Skip CONTROL (1) / UNKNOWN (2) tokens — BOS/EOS/PAD are control
      // and shouldn't render in user-visible output. Matches Python
      // sentencepiece's `DecodeIds` default. UNUSED (5) and USER_DEFINED
      // (4) are kept verbatim.
      if (t === 2 || t === 3) continue;
      parts.push(piece);
    }
    flushBytes();

    let result = parts.join("").replace(new RegExp(SPIECE_UNDERLINE, "g"), " ");
    if (this.addDummyPrefix && result.startsWith(" ")) {
      result = result.substring(1);
    }
    return result;
  }
}

export const fromBinaryImpl = (bytes) => SentencePieceBPE.fromBinary(bytes);
export const encodeImpl = (sp, text) => sp.encode(text);
export const decodeImpl = (sp, ids) => sp.decode(ids);
export const bosTokenImpl = (sp) => sp.bosId;
export const eosTokenImpl = (sp) => sp.eosId;
export const unkTokenImpl = (sp) => sp.unkId;
export const vocabSizeImpl = (sp) => sp.vocabSize();

// Toggle the SP `add_dummy_prefix` normalization. SP-trained models have
// it enabled in the proto (Llama family included), but HF's Llama
// tokenizer with `legacy: false` (the default for newer tokenizers) does
// NOT prepend the dummy ▁ to the first word — the trained model expects
// IDs *without* the leading ▁ for word-initial tokens. Call
// `setAddDummyPrefix(sp, false)` to match HF's non-legacy behavior.
export const setAddDummyPrefixImpl = (sp, value) => {
  sp.addDummyPrefix = value;
};

