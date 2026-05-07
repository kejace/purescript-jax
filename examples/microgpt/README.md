# microGPT × purescript-jax

Two side-by-side ports of [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):

- **`faithful/`** — pure PureScript, custom `Value` autograd, no `Jax.*`. Pedagogy match.
- **`jax/`** — same pipeline, real tensor ops, JAX autograd. Reads like normal user code on top of the framework.

Both demos:

- inline a small subset of [Karpathy's `names.txt`](https://github.com/karpathy/makemore/blob/master/names.txt) (30 names) so they're self-contained,
- tokenize at the character level,
- train a small next-character model,
- sample 5 continuations.

Run either with `bunx spago run` from its directory.

## Why two demos

Karpathy's microGPT is *zero-dependency* by design. Every line of autograd is on the page, including the topological sort and the chain-rule walk. That pedagogy gets lost the moment you call into a framework. So:

- `faithful/` keeps it. `Value.backward` is six readable lines (DFS topo + reverse propagation through `localGrads`). The cost: PureScript can't operator-overload on `Value`, so what's `a * b` in Python becomes `mul a b` here. The model is a 1-layer bigram-conditioned MLP rather than a full transformer — porting attention scalar-by-scalar would add another ~150 lines without changing what the demo demonstrates.

- `jax/` shows how the same pipeline reads when you have a framework. Section structure is identical. Where Faithful does `V.backward loss` and a hand-rolled Adam, Reimagined does `Train.stepN`. The trade: JAX's autograd is opaque — you can't read `valueAndGradT` the way you can read `Value.backward`. If you want to know what JAX is doing, the Faithful sibling is your reference.

Both files have **the same five sections** (corpus → tokenizer → init → model + loss → training → sampling) so you can read them side by side.

## What you're seeing

### Faithful (~270 LoC)

```
[microgpt-faithful] vocab=22 hidden=8 intermediate=16
  step 1   · loss 3.16
  step 200 · loss ~2.8
  → bevotvehetamicur…
```

Architecture: `forward(c) = silu(emb[c] @ W1 + b1) @ W2 + b2`. One (input, target) pair per step, hand-rolled Adam, temperature softmax sampling. Loss is single-example-SGD noisy and 200 steps isn't enough to produce coherent names; the point is "the autograd works end to end". Bump `numSteps` to 5000 and add mini-batching for real output.

The Faithful demo deliberately accumulates per-pair loss across the whole corpus *would* OOM the autograd graph at this scale (we tried; the Value tape blows past 4 GB on a 197-pair fold) — so we follow Karpathy's exact pattern of one example per step.

### Reimagined (~250 LoC)

```
[microgpt-jax] vocab=22 hidden=32 nHeads=4 nLayers=1
  step 0   · loss 4.37
  step 100 · loss 0.014
  → ada
emma
olivisa
```

Architecture: 1-layer transformer (RMSNorm + GQA-style attention + SwiGLU MLP). Trains on the first 32 characters of the corpus as one long sequence — by step 100 the model has *memorized* this window, so you see the training data play back. Real names-of-the-world generation needs:

- training over a *window* that slides across the corpus rather than a fixed prefix,
- 1000+ steps,
- temperature > 0.8 to break out of memorized completions.

That's a one-screen extension of the demo; left out for size.

## How they map

| Step | Faithful | Reimagined |
|------|----------|-----------|
| Section 1 (tokenizer) | `buildVocab` + `charToId` + `idToChar` (~15 lines) | `CharTokenizer.fromText` (3 lines) |
| Section 2 (init) | `randn` + Glorot scaling, layer-by-layer | `Random.normal` + `mulScalar`, key-tree split |
| Section 3 (model) | `linear` + `softmax` + `crossEntropy`, all on `Value`s | `Jax.NN.Block.transformerStack` + `Jax.NN.Train.makeCrossEntropyLoss` |
| Section 3 (autograd) | `Value.backward` (DFS topo + reverse-mode chain rule) | `Jax.Autodiff.valueAndGradT` (one call; the work is hidden inside JAX) |
| Section 4 (optimizer) | hand-rolled Adam loop with `m/v` Refs and a per-step `lr_t` | `Optax.adam` chained with `Schedule.scaleBySchedule(linearDecay)` |
| Section 4 (loop) | inline `for_ [1..numSteps]` with backward + adam | `Train.stepN` (one call) |
| Section 5 (sampling) | hand-rolled temperature softmax + categorical | `Jax.NN.Generate.generateTemperature` |

## Drawbacks worth being honest about

1. **Faithful is uglier than Karpathy's Python**, and that's structural. PureScript has no operator overloading on user types, so `a * b` becomes `V.mul a b`. `Effect` threading shows up wherever Python silently mutates. Expect ~1.5× the line count for any port of this style. The pedagogy survives — every autograd op is still on the page — but it's noisier.

2. **Reimagined hides the autograd**, by design. `valueAndGradT` is one call and the rest is a black box. If you're trying to *teach* autograd, this is the wrong demo; use the Faithful sibling and read `Value.backward`. If you're trying to teach what a real framework looks like, this is the right one.

3. **The framework still leaks `EffectFn1` once.** `valueAndGradT lossFn` returns an `EffectFn1 ModelWeights …`. We hide that inside `Train.step`; you still need `runEffectFn1` if you go around `Train`. Documented; not removed because the wrap cost would lose JAX's arity inference.

4. **No streaming sampling in the demo.** `generateTemperature` returns a final `Array Int`. The framework has `generateGreedyCachedStream` for token-by-token output; threading that into the Reimagined demo's `for_` loop is a 5-line change we didn't make to keep the file scannable.

## Framework additions made for these demos

The Reimagined demo would have been ceremonial without four small framework extras (all upstreamed in the same branch):

- `Jax.Loaders.CharTokenizer` — char-level encode/decode (pre-existed only as SentencePiece BPE).
- `Jax.Train.{initial, step, stepN}` — collapses the `valueAndGrad → updateT → applyUpdatesT` triple per iteration.
- `Jax.Optax.Schedule.{linearDecay, cosineDecay, scaleBySchedule, chain}` — bindings to upstream optax composition primitives.
- `Jax.Loaders.Fetch.fetchTextLines` — `fetchText` + split + filter empty.
- `Jax.Random.normal` — FFI binding to `random.normal` for parameter init.
- `Jax.NN.Block.refModelWeights` — promoted to public so the demo can ref-bump before `Optax.initT`.

If you peel these back, the demo gets ~50 lines longer and significantly more ceremonial, but nothing essential is hidden.

## Running

```sh
# Faithful
cd examples/microgpt/faithful && bunx spago run

# Reimagined
cd examples/microgpt/jax && bunx spago run
```

Both run on Bun + Node. Reimagined uses `setDefaultDevice "wasm"` (the universal fallback in jax-js); WebGPU isn't available in Node, but for a 1-layer 32-hidden model it doesn't matter.
