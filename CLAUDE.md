# Execution Plan: nanoGPT to PureScript via jax-js

## 1. Project Objective & Context
This document serves as the master blueprint for an autonomous coding agent to port the `nanoGPTJAX` architecture into PureScript, utilizing the `@jax-js/jax` library via Foreign Function Interface (FFI). 

The goal is to produce a strongly-typed, purely functional implementation of a modern Large Language Model (LLM) architecture running natively in the browser, with **WebGPU as the perf target and Wasm as the portable fallback**. To ensure maximum reliability, you will apply the same FFI architecture and memory-management patterns used to implement fully automatically differentiable math primitives in our other PureScript web applications.

---

## 2. Agent Directives & Constraints
When executing this plan, the coding agent must adhere to the following rules:
- **Immutability vs. Memory:** PureScript assumes garbage collection, but `jax-js` requires manual reference counting. Implement a `Managed` monad (via `ContT Unit Effect`) to allow flat `do`-notation bindings that automatically guarantee `.dispose()` is called upon exit. In dev builds, register every allocated tensor with a JS `FinalizationRegistry` whose callback logs a leak warning — the spec does *not* guarantee FR fires (V8 skips it under memory pressure), so this is a diagnostic, not a safety net. Leaks must still be fixed at the source. Note that jax-js ops *consume their arguments* (refcount −1 per pass-in) and `.ref` is a property, not a method — `Managed` allocations must `ref` before handing a tensor to an op. **Inspection methods (`.js()`, `.dataSync()`) also consume the receiver** — jax-js's internal `#dataInline` path calls `this.dispose()` to migrate AluExp-sourced arrays to CPU. The FFI shim must wrap those with `.ref` (i.e. `(a) => a.ref.js()`) so they read as non-consuming from the PureScript side; users should not have to ref-bump just to read values.
- **Long-Lived Tensors:** Model weights, RoPE precomputed tables, and embedding matrices outlive any `Managed` scope. Allocate them in a top-level `Effect` with a distinct `LongLived (NDArray d)` newtype so the type system flags accidental disposal. The decode loop borrows from a `LongLived` pool; only intermediates flow through `Managed`.
- **Currying Bridge:** You must use `Data.Function.Uncurried` / `Effect.Uncurried` when binding to `jax-js`'s `jit`, `vmap`, and `grad` functions, as XLA tracing relies on JavaScript function arity (`fn.length`).
- **Shape Typing:** For v1, avoid full dimension tracking, but implement **Rank Tracking** via Phantom Types (`foreign import data NDArray :: Type -> Type` with `D1`, `D2`, `D3`). This catches rank-mismatch errors at compile-time without the overhead of full type-level integer arithmetic.
- **Determinism / RNG:** All sampling and stochastic init flows through a single `Jax.Random` module. If jax-js exposes splittable PRNG keys, use them; otherwise seed a userspace splittable PRNG (e.g. Philox in JS). Test fixtures depend on reproducibility — never call `Math.random()` directly inside model code.
- **No External NN Frameworks:** Do not attempt to port Flax or Equinox. `nanoGPTJAX` is built from scratch; replicate its pure mathematical forward passes directly.

---

## 3. Implementation Phases

### Phase 1: Core FFI & Memory Management (The Foundation)
1. **Initialize Project:** Set up a standard Spago project with Vite/Esbuild for browser bundling.
2. **Bind Core `jax-js` Primitives:**
   - Write `src/Jax/Core.js` and `src/Jax/Core.purs` with rank-parameterized types (`NDArray D1`, `NDArray D2`, …).
   - Constructors live under `numpy` (imported as `np`): bind `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`. All return rank-typed `NDArray d` and are effectful (allocation hits the device). Ensure explicit `dtype` selection (prefer `float32`).
   - Calling convention is op-specific (verified against the installed jax-js dist; the surface is *not* uniform between methods and free functions):
     - **Array methods** (`a.method(...)`): `add`, `mul`, `sub`, `div`, `neg`, `sum`, `mean`, `transpose`, `reshape`, `slice`.
     - **`np`-level free functions** (`np.func(...)`): `matmul`, `concatenate`, `reciprocal`, `sigmoid`, `square`, `sqrt`, `sin`, `tanh`.
     - **Not in upstream — synthesize**: `rsqrt` ≔ `np.reciprocal(np.sqrt(x))`. Document any further synthesized op alongside its definition.
   - Mismatches between method-style and free-function form throw `TypeError: a.method is not a function` at runtime — the type system can't catch this across the FFI boundary, so each shim's calling form must be verified once against upstream.
   - `.ref` is a *property*, not a method: `refImpl = (a) => a.ref`.
3. **Implement Resource Management:**
   - Create `Jax.Managed` using `ContT Unit Effect`.
   - Expose `allocate :: Effect (NDArray d) -> Managed (NDArray d)` to ensure tensors are disposed after the continuation runs.
4. **Bind Autodiff:**
   - Implement `grad`, `value_and_grad`, and `jit` using `EffectFn` and `runEffectFn`.
5. **Weight & Tokenizer Loading (`Jax.Loaders`):**
   - Bind `@jax-js/loaders` for safetensors (`loadSafetensors :: URL -> Effect (Map String (NDArray d))`), HTTP-cached fetches, and BPE tokenizer (`loadBPE :: URL -> Effect Tokenizer`).
   - Verify upstream uses `ArrayBuffer` slicing (not JSON) before assuming. Only hand-roll if upstream is missing a needed feature (streaming, dtype coercion, sharded files); document which.

### Phase 2: Neural Network Primitives
Port the exact mathematical definitions from `nanoGPTJAX`.
1. **RMSNorm:**
   - Formula: `x * rsqrt(mean(x^2) + eps) * weight`.
   - Ensure epsilon addition is broadcasted correctly.
2. **Token Embedding & LM Head:**
   - `embed :: NDArray D2 -> NDArray D1 -> NDArray D2` via gather/take (shapes: `vocab×embed` table, `seq` ids → `seq×embed`).
   - `unembed`: linear projection from hidden → vocab logits. Support optional weight-tying with the embedding table (shared `NDArray`, fed in via `LongLived`).
3. **RoPE (Rotary Positional Embeddings):**
   - Create the precomputed sine/cosine frequency tensors (held as `LongLived` for the lifetime of the model).
   - Implement the rotary mix function, ensuring complex number arithmetic is handled via slicing and concatenation if `jax-js` lacks native complex tensor support.
4. **GQA (Grouped Query Attention):**
   - Port the causal mask generation.
   - Implement multi-headed attention.
   - Define `KVCache` (per-layer) and `KVCacheStack = Array KVCache` (whole model). The forward pass: `GPTWeights -> KVCacheStack -> NDArray -> Tuple NDArray KVCacheStack`. Preallocate at `max_seq_len` rather than growing — verify whether jax-js exposes in-place slice-update; if not, accept allocate-new-and-dispose-old per decode step for v1.
5. **SwiGLU / MLP Block:**
   - Implement the `silu` (Swish) activation function: `x * sigmoid(x)`.
   - Chain the gate and up-projection matrices.
6. **Block & Stack:**
   - Compose RMSNorm + GQA + RMSNorm + MLP into a `TransformerBlock`. Stack to depth `n_layers`, threading `KVCacheStack` through each layer.

### Phase 3: Inference Pipeline (Pretrained Checkpoint)
Wire the primitives into an end-to-end run before tackling training — inference on a pretrained checkpoint is the demo; training is a stretch.
1. **Tokenizer:** Encode prompts via `Jax.Loaders` BPE; decode generated token IDs back to text.
2. **Forward pass:** Stack `n_layers` of `TransformerBlock` over an embedded sequence, project to logits via the LM head.
3. **Decode loop:** `prefill` (process the prompt to populate `KVCacheStack`), then `decodeStep :: KVCacheStack -> TokenId -> Effect (KVCacheStack, Logits)` jit-compiled at the per-step boundary.
4. **Sampling:** `argmax`, `temperature`, `top-k` via `Jax.Random`.
5. **Streaming:** Emit each decoded token via a callback / `Aff` channel — never block the event loop on the full generation.

### Phase 4: Autodiff, Optimizer & Training (Optional / Stretch)
`@jax-js/optax` provides Adam and SGD; bind it via FFI rather than hand-rolling. Only fall back to a PureScript reimplementation if a needed option (decoupled weight decay, gradient clipping, schedule) is unsupported upstream. Defer until inference (Phase 3) is solid.
1. **Optimizer Bindings (`Jax.Optax`):**
   - FFI for `optax.adam(lr, ...opts)` and `optax.sgd(...)` returning an opaque `Optimizer` handle plus `init :: Params -> OptState` and `update :: Grads -> OptState -> Params -> { params, state }`.
   - If AdamW (decoupled weight decay) is missing upstream, layer it on top as a thin PureScript wrapper rather than reimplementing Adam.
2. **Model State Record:**
   - Define the `GPTConfig` record (vocab size, layers, heads, dimensions).
   - Define the hierarchical `GPTWeights` record.
3. **Training Step (The Jitted Loop):**
   - Write a `lossFn :: GPTWeights -> NDArray -> NDArray -> NDArray` (weights, inputs, targets -> loss).
   - Use `value_and_grad` on `lossFn`.
   - Wrap the entire update step (forward, backward, optimizer step) in `jit`.

---

## 4. Testing Infrastructure (In-Browser)
Testing tensor operations requires a DOM/Browser environment to access WebGL/WebGPU.
1. **Test Runner Setup:**
   - Use **Vitest in browser mode** (`@vitest/browser` + Playwright provider) to run `Test.Main` compiled output in headless Chromium. This reuses the Vite config from §5 and gives us WebGPU-capable headless runs out of the box. Karma is end-of-life; do not use it.
2. **Unit Tests (Math Parity):**
   - Export known input arrays and expected output arrays from Python JAX (saved as JSON).
   - Write PureScript tests that load these fixtures, run the `jax-js` FFI functions, and assert that outputs match.
   - *Crucial:* Set tolerance dynamically based on the backend (e.g., `1e-5` for CPU, but `1e-3` for WebGL/float16).
3. **Module Tests:**
   - Test `RMSNorm`, `RoPE`, and `GQA` forward passes in isolation using the fixture strategy.
4. **Memory Leak Tests:**
   - Write a test that runs a `matmul` in a loop 10,000 times.
   - Query the `jax-js` backend memory profiler (if exposed) or monitor for WebGL context crashes to verify `dispose()` is working.

---

## 5. Benchmarking Suite (WebGL/WebGPU)
1. **UI Dashboard:**
   - Build a minimal HTML/CSS interface served via Vite.
   - Create a realtime dashboard tracking memory usage and compute times.
2. **Forward Pass Benchmarks (Tokens / Sec):**
   - JIT compile the text generation loop.
   - Run a prompt through the model, capturing `performance.now()` before and after generation.
   - Calculate Tokens Per Second (TPS).
3. **Backend Toggle:**
   - jax-js exposes `defaultDevice("wasm" | "webgpu" | "webgl" | "cpu")`. Treat **WebGPU** as the perf target and **Wasm** as the universal fallback (Safari/older browsers). Run the TPS benchmark on all supported backends present at runtime; surface a feature-detect ladder `webgpu → wasm` (not webgpu → webgl).
   - Print a table of backend × prefill-tps × decode-tps × peak-MB.
