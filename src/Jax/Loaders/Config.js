// Parse a HuggingFace `config.json` for a Llama-arch model into our
// internal `ModelConfig` shape. We pluck only the fields we need;
// unsupported config quirks (e.g. RoPE scaling, sliding-window
// attention) are silently ignored — extend if a target model needs them.

export const parseLlamaConfigImpl = (json) => {
  const c = JSON.parse(json);
  const hidden = c.hidden_size;
  const nHeads = c.num_attention_heads;
  return {
    hidden,
    nHeads,
    nKvHeads: c.num_key_value_heads ?? nHeads,
    headDim: c.head_dim ?? Math.floor(hidden / nHeads),
    intermediate: c.intermediate_size,
    nLayers: c.num_hidden_layers,
    maxSeqLen: c.max_position_embeddings ?? 2048,
    vocabSize: c.vocab_size,
    ropeTheta: c.rope_theta ?? 10000.0,
    normEps: c.rms_norm_eps ?? 1.0e-6,
  };
};
