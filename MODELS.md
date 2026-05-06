# Supported model architectures

`Jax.Loaders.LlamaAdapter` is wired for the **Llama-2 family weight
layout**: pre-norm RMSNorm, GQA-capable self-attention with bias-free
`q/k/v/o_proj`, SwiGLU MLP with `gate/up/down_proj`, RoPE on Q/K. Anything
that ships safetensors in this exact shape will load; deviations mean
either fail-fast (architectures we explicitly reject) or quietly-wrong
output (architectures with subtle differences we don't yet enforce).

The compat check lives in `Jax.Loaders.Config.compatibleArchs`. The worker
calls `probeRawExtras` before committing to a load and throws a clear
error for anything off-list.

## Verified

| Model | model_type | Status | Notes |
|---|---|---|---|
| `Felladrin/Smol-Llama-101M-Chat-v1` | llama | ✅ end-to-end | Default demo checkpoint. Untied lm_head. |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama | should work | Same shape as Smol-Llama. ~2.2 GB. Not yet load-tested in browser. |

## Compatible (should work, untested in this repo)

| Model | model_type | Notes |
|---|---|---|
| Llama-2-7B family | llama | Standard reference. Memory may be tight in browser. |
| `mistralai/Mistral-7B-Instruct-v0.2` | mistral | We accept `mistral` because the weight layout is identical when `sliding_window` is null. **Mistral with sliding-window set is silently wrong on long prompts** — we run full-context attention, the model was trained with windowed. We log a note in `[worker]`. |
| `mistralai/Mixtral-*` (MoE) | mixtral | **Untested.** MoE routing is an architectural gap; would need new code in the MLP path. Not in `compatibleArchs`. |

## Incompatible (refused at load by `probeRawExtras`)

| Model | model_type | Blocker |
|---|---|---|
| Phi family | phi | `qk_layernorm` (LayerNorm on Q/K) + partial-RoPE — different attention. |
| Qwen2 / Qwen2.5 | qwen2 | Q/K/V projections carry biases. Adapter would need to load + apply them. Tied embeddings (no lm_head.weight) are already supported. |
| GPT-2 / GPT-Neo | gpt2 / gpt_neo | Learned positional embeddings (no RoPE). Different MLP (no SwiGLU). |
| Gemma | gemma | RMSNorm placement and tied-embedding scaling differ subtly. |

## Adding a new architecture

The minimum needed:

1. Confirm `config.json` exposes (or implies) all fields in
   `Jax.Loaders.Config.RawConfig`. If not, extend `RawConfig` and
   the `toCfg`/`fromCfg` mapping.
2. Inspect the safetensors header (range-fetch the first ~200 KB and
   parse the JSON header) to confirm tensor names match the
   `model.layers.N.{self_attn,mlp}.*` convention. If different,
   either add a new adapter module or extend `LlamaAdapter` with
   per-naming-convention switches.
3. Verify the *math* matches: same RMSNorm formula, same RoPE half-split
   convention, no per-layer biases in projections, no extra norms inside
   attention.
4. Add `model_type` to `Jax.Loaders.Config.compatibleArchs` and document
   here.

A safetensors range probe to inspect tensor names without downloading
the whole file:

```bash
curl -sS -L --max-time 30 -o /tmp/h.bin -r 0-200000 \
  "https://huggingface.co/<repo>/resolve/main/model.safetensors"
node -e "
  const all = require('fs').readFileSync('/tmp/h.bin');
  const hdr = JSON.parse(all.subarray(8, 8 + Number(all.readBigUInt64LE(0))).toString('utf8'));
  Object.keys(hdr).filter(k => k.includes('layers.0.')).forEach(k =>
    console.log(k, hdr[k].dtype, JSON.stringify(hdr[k].shape)));
"
```

That's the same probe used to seed this document.

## Why Mistral but not Qwen2

Both have `model_type` ≠ `llama`. The difference is the **weight shape**:

- Mistral's per-layer tensors are byte-identical to Llama-2's. Loading
  works without changes; sliding-window is the only behavioral
  divergence and we surface it as a log warning.
- Qwen2 adds `q_proj.bias`, `k_proj.bias`, `v_proj.bias` per layer.
  Loading them ignores the biases (the adapter looks up only
  `.weight`), so the projections would be off by a constant per-head
  vector — silently wrong outputs. Better to refuse at load.

If you want Qwen2, the right move is a small extension to
`AttentionWeights` (`Maybe (NDArray D1)` for each bias) plus an
`add` after each projection in `attentionForward` and
`attentionForwardCached`. ~30 lines of code; happy to take a PR.
