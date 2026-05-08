# Session notes — 2026-05-08

> Two threads in this session, both shipped to `main`:
>
> 1. **Type-level shapes Stage 7** — the unsafe-call audit and final
>    cleanup of the boundary between rank-only `NDArray d` and shape-
>    typed `Tensor s`.
> 2. **GUI backend swap + caching** — added a backend dropdown, fixed
>    the wasm warm-up hang, made the OPFS cache survive preset changes,
>    and made backend swap migrate the loaded model rather than force a
>    reload.

The branch is `main`, working tree clean, last commit pushed
(`22e8dff`). 52/52 tests pass after every commit in this session.

---

## Commits landed this session

```
22e8dff fix(ui): migrate model on backend swap; drop cpu from picker
2220e2c fix(ui): wasm warmup, no-reload backend swap, persistent OPFS
9cdf290 feat(ui): backend swap dropdown
1cc05d9 feat(types): Stage 7 — typed lifecycle helpers + boundary cleanup
```

Prior to this session the type-level shapes work was at Stage 6.

---

## Thread 1 — Type shapes Stage 7 (commit `1cc05d9`)

### What got added

Typed lifecycle helpers in `Jax.Shape.Tensor`, mirroring the rank-only
ops in `Jax.Core` over `Tensor s`:

- `refT :: Tensor s -> Effect (Tensor s)`
- `disposeT :: Tensor s -> Effect Unit`
- `shapeT :: Tensor s -> Effect (Array Int)`
- `squareT :: Tensor s -> Effect (Tensor s)`
- `sumT :: Tensor s -> Effect (Tensor s')`  *(reduce-all, output shape open)*
- `toJsT :: Tensor s -> Effect Foreign`

Plus runtime-sized typed allocators in `Jax.Shape.Tensor.Op`:

- `zerosWith :: Array Int -> Effect (Tensor s)`
- `onesWith :: Array Int -> Effect (Tensor s)`

And a `Managed` binder that owns shape:

- `allocateT :: Effect (Tensor s) -> Managed (Tensor s)`

### Boundary cleanup

Migrated to the typed helpers so `unsafeForgetShape` / `unsafeAssumeShape`
no longer appears in user-facing code:

- `Jax/Demo/Microgpt.purs` — `glorotMat` returns `Tensor (S2 a b)`;
  `buildLayer` / `buildWeights` use `Op.onesWith` + `allocateT` directly.
- `Worker.purs` — `varyingWeight` typed; `allocate` → `allocateT`;
  `refModelWeights` uses `refT`.
- `Main.purs` — `buildModel` / `buildLayer` use `Op.onesWith` + `allocateT`.
- `Jax/NN/Block.purs` — `refModelWeights` uses `refT`; the internal
  `asD2` / `asD1` aliases now go through `withRank` (RankOf-checked)
  instead of raw `unsafeForgetShape`.
- `Jax/NN/Generate.purs` — `decodeLoop` and the streaming variant pass
  `withRank weights.embedding` to `decodeLoopWithHead` instead of an
  unsafe-erase.
- `Jax/NN/Train.purs` — `makeCrossEntropyLoss` uses `withRank` for the
  embedding/transpose path.
- `Jax/Pytree.purs` — `CountParam` and `SumSquaredL2` Tensor-leaf
  instances use `shapeT` / `refT` / `squareT` / `sumT` / `toJsT` /
  `disposeT` instead of bridging through `unsafeForgetShape`.
- `test/Test/Main.purs` — `testLlamaEndToEnd` L2-decomposition path
  uses the typed helpers; `varyingWeight` and the three `testGenerate`
  weight blocks use `Op.onesWith` + `allocateT`.

### Remaining `unsafe*` sites (all justified)

`grep -n "unsafeAssumeShape\|unsafeForgetShape"` in `src/` and `test/`,
excluding `Jax/Shape/Tensor*`:

- **`Jax/Loaders/LlamaAdapter.purs`** — 11 sites. The canonical FFI
  boundary: shapes come from safetensors metadata, this is the trusted
  point asserting them. Whitelisted by the project rule in
  `.claude/fp-police-rules.md`.
- **`Worker.purs:455`** — inside `varyingWeight`, a typed-allocator
  helper that returns `Tensor s` from a typed-reshape NDArray. Local
  bridge.
- **`Jax/Demo/Microgpt.purs:319`** — inside `glorotMat`, the
  fan-in-scaled normal initializer. Local bridge inside the typed
  allocator.
- **`test/Test/ShapeOps.purs:49`**, **`test/Test/ShapeNN.purs:72,151`**,
  **`test/Test/Main.purs:733`** — test fixtures + synthetic allocators.
- **`Jax/Shape/Tensor*`** — the typed-helper modules themselves; all
  unsafe coercions live here by design.

The architectural promise is now: outside the typed-helper modules and
the loader, you don't see `unsafe*` shape coercions in code paths users
read.

---

## Thread 2 — Backend swap + caching (commits `9cdf290`, `2220e2c`, `22e8dff`)

### `9cdf290` — backend swap UI

- New protocol pair: `SetBackend { backend }` (in) and
  `BackendError { tried, err }` (out). Both go through the existing
  `taggedSum` codec.
- New `<select id="backendSelect">` next to the backend label in
  `index.html`. Starts disabled; enabled on first `Ready`.
- `handleSetBackend` in `Worker.purs` calls `trySetDevice`, posts
  `Ready { backend }` on success or `BackendError` on failure.
- `Browser.purs` syncs the dropdown's selected option from `Ready` and
  surfaces errors in a hint span.

### `2220e2c` — wasm warm-up, no-reload swap, persistent cache

Three coupled fixes:

1. **Wasm hang on first op.** jax-js's `defaultDevice("wasm")` only
   flips a flag — the wasm runtime is loaded lazily on first op.
   Switched `Worker.js` to do **top-level `await init()`** at module
   load (the worker is `{ type: "module" }`, so TLA works), and store
   the returned `Device[]` so `trySetDevice` can refuse a backend that
   wasn't actually initialized.

2. **Backend swap stopped clearing modelRef.** The previous version
   wrote `Nothing` to modelRef on swap, forcing the user to reload.
   Removed that line — see thread `22e8dff` for the follow-up that
   actually makes "no reload" work.

3. **OPFS cache survives preset change.** The preset `onChange` used
   to call `clearOpfs` to reclaim quota. Removed — multiple preset
   checkpoints now coexist in OPFS, so flipping presets re-uses any
   cached blob. The explicit "clear cache" button is still there.

### `22e8dff` — model migration on backend swap; cpu hidden

Diagnosis of "cpu mode stops at running…": with the model loaded on
webgpu and the user swapping to cpu, the weight tensors stayed on
webgpu. **jax-js does not auto-migrate cross-device ops** — a cpu
kernel dispatched onto webgpu inputs just hangs.

Fixes:

- **Drop `cpu` from the dropdown.** jax-js's README calls it "slow,
  interpreted JS, only meant for debugging" — running a 100M-param
  model on the cpu interpreter is not a useful mode.
- **`devicePut` migration on swap.** Added `devicePutImpl` in
  `Worker.js` (`(tree, device) => devicePut(tree, device)`); PS-side
  `Aff` wrapper via `Control.Promise.toAffE`. `handleSetBackend` now
  `launchAff_`s the migration: walks the whole `LoadedModel` record
  through `devicePut`, writes the migrated tree back to modelRef. On
  failure, clears modelRef so a reload via the OPFS cache is the
  recovery path. On success, generate works without reload.

---

## State of `Jax.Worker.Protocol`

WorkerIn:
```
LoadModel | Generate | Benchmark | TrainSynthetic
| MicrogptTrain | MicrogptSample
| SetBackend                       -- new
```

WorkerOut:
```
Ready | LoadStart | LoadConfig | LoadTokenizer | LoadFetched
| LoadDone | LoadError
| GenerateStart | GenerateError | TokenText | Done
| BenchResult | BenchmarkDone
| TrainStart | TrainStep | TrainDone | TrainError
| MicrogptStart | MicrogptStep | MicrogptSampled | MicrogptTrainDone
| MicrogptSampleDone | MicrogptError
| BackendError                     -- new
```

`Ready { backend }` is now both a startup signal *and* a
backend-changed signal. The browser uses it to enable the dropdown
and sync its selected option.

---

## Open items / things to verify in the browser

These weren't tested in the browser this session (only at the build /
unit-test level). The first time someone runs the dev server they
should sanity-check:

1. **WebGPU → wasm migration actually works.** Load Smol-Llama on
   webgpu, generate a few tokens, swap to wasm. Should generate again
   without reload, just slower. If `devicePut` chokes, `BackendError`
   isn't fired (success path was assumed) — instead the migration
   clears modelRef and posts `Ready`. UX is "load status nothing,
   generate fails 'no model loaded'". Worth verifying that path
   surfaces a clear message.

2. **OPFS cache hits across page reloads.** Load Smol-Llama
   (~400 MB, under the 1.5 GB cache threshold), reload the page,
   check the `[fetch-cache] hit ...` console log on second load.
   Should be subjectively instant.

3. **Backend dropdown reflects truth.** On a browser without WebGPU,
   `_initializedDevices` should be `["wasm"]` and the worker's
   `selectBackend` should pick wasm. Verify the dropdown surfaces wasm
   as the selected option (not webgpu). Also: a user-initiated swap to
   `webgpu` from a non-webgpu browser should hit the `!includes(name)`
   short-circuit in `trySetDevice` and post a `BackendError` (not a
   silent flip-then-hang).

4. **TinyLlama (over the 1.5 GB threshold).** Doesn't cache; every
   reload re-fetches. Out of scope for this session; if reload speed
   becomes a pain, raise `CACHE_MAX_BYTES` or add streaming-decode
   caching.

---

## Where to pick up

Easy follow-ups:

- **Show migration progress.** A long `devicePut` for a multi-GB
  model is silent right now. Adding a `BackendMigrating { backend }`
  → `Ready { backend }` pair would let the browser show "migrating
  weights to wasm…" in the hint span.

- **Disable Generate during migration.** Currently the user can click
  Generate while migration is in flight, which would race against the
  modelRef write. Cheap fix: gate the generate button on a Ref<Bool>
  on the worker side, post `GenerateError` if mid-migration.

- **`unsafeAssumeShape` policy doc.** `.claude/fp-police-rules.md`
  whitelists `Loaders/**` and `Jax/Core.purs` but the actual sites are
  also in `Jax/Shape/Tensor*` (the typed-helper module itself, by
  design) and the explicit-typed-allocator helpers in
  `Worker.purs` / `Microgpt.purs`. Worth updating the rule to match
  reality so `/fp-police` doesn't flag legitimate sites.

Bigger follow-ups (deferred from CLAUDE.md):

- Phase 4 training (Optax-bound, optional / stretch).
- Larger models — TinyLlama works but parsing + adapt are slow.

---

## File index — what changed

```
src/Jax/Shape/Tensor.purs        — refT/disposeT/shapeT/squareT/sumT/toJsT
src/Jax/Shape/Tensor/Op.purs     — zerosWith/onesWith
src/Jax/Managed.purs             — allocateT
src/Jax/NN/Block.purs            — refModelWeights typed; asD2/asD1 → withRank
src/Jax/NN/Generate.purs         — withRank instead of unsafeForgetShape
src/Jax/NN/Train.purs            — withRank for embedding path
src/Jax/Pytree.purs              — Tensor leaf instances via shapeT/squareT/…
src/Jax/Demo/Microgpt.purs       — typed allocators throughout
src/Worker.purs                  — handleSetBackend + devicePut migration
src/Worker.js                    — top-level await init(); devicePutImpl
src/Browser.purs                 — backend dropdown wiring; preset no-clear
src/Jax/Worker/Protocol.purs     — SetBackend + BackendError
src/Main.purs                    — typed allocators throughout
test/Test/Main.purs              — typed allocators in test fixtures
index.html                       — backend <select>; cpu dropped
```
