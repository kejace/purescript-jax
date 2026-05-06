// Build a Llama-formatted safetensors blob in memory for testing the
// real-model loading path. Uses PyTorch's `[out, in]` convention for
// linear weights (LlamaAdapter transposes on load).

const cfg = {
  vocab: 8,
  hidden: 8,
  nHeads: 2,
  nKvHeads: 2,
  headDim: 4,
  intermediate: 16,
  nLayers: 1,
};

function genFloat32(shape) {
  const n = shape.reduce((a, b) => a * b, 1);
  const arr = new Float32Array(n);
  for (let i = 0; i < n; i++) arr[i] = ((i / n) * 0.2) - 0.1;
  return arr;
}

function buildSafetensors(tensors) {
  // tensors: { name: { dtype, shape, data: TypedArray } }, in insertion order.
  const headerEntries = {};
  let offset = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const byteLength = t.data.byteLength;
    headerEntries[name] = {
      dtype: t.dtype,
      shape: t.shape,
      data_offsets: [offset, offset + byteLength],
    };
    offset += byteLength;
  }
  headerEntries.__metadata__ = {};
  let header = JSON.stringify(headerEntries);
  // Pad the header with spaces so the data section starts at an 8-byte
  // boundary. This is required for float32/int32 reads to align cleanly.
  while ((8 + header.length) % 8 !== 0) header += " ";
  const headerBytes = new TextEncoder().encode(header);
  const totalSize = 8 + headerBytes.length + offset;
  const buf = new Uint8Array(totalSize);
  const dv = new DataView(buf.buffer);
  dv.setBigUint64(0, BigInt(headerBytes.length), true);
  buf.set(headerBytes, 8);
  let pos = 8 + headerBytes.length;
  for (const [name, t] of Object.entries(tensors)) {
    buf.set(
      new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength),
      pos,
    );
    pos += t.data.byteLength;
  }
  return buf;
}

export const makeLlamaFixtureImpl = () => {
  const tensors = {};
  const add = (name, shape) => {
    tensors[name] = { dtype: "F32", shape, data: genFloat32(shape) };
  };

  add("model.embed_tokens.weight", [cfg.vocab, cfg.hidden]);
  add("model.norm.weight", [cfg.hidden]);

  for (let i = 0; i < cfg.nLayers; i++) {
    const p = `model.layers.${i}.`;
    add(p + "input_layernorm.weight", [cfg.hidden]);
    // PyTorch convention: [out_features, in_features]
    add(p + "self_attn.q_proj.weight", [cfg.nHeads * cfg.headDim, cfg.hidden]);
    add(p + "self_attn.k_proj.weight", [cfg.nKvHeads * cfg.headDim, cfg.hidden]);
    add(p + "self_attn.v_proj.weight", [cfg.nKvHeads * cfg.headDim, cfg.hidden]);
    add(p + "self_attn.o_proj.weight", [cfg.hidden, cfg.nHeads * cfg.headDim]);
    add(p + "post_attention_layernorm.weight", [cfg.hidden]);
    add(p + "mlp.gate_proj.weight", [cfg.intermediate, cfg.hidden]);
    add(p + "mlp.up_proj.weight", [cfg.intermediate, cfg.hidden]);
    add(p + "mlp.down_proj.weight", [cfg.hidden, cfg.intermediate]);
  }
  return buildSafetensors(tensors);
};

export const llamaFixtureCfgImpl = cfg;
