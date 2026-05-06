# Refactor plan — making purescript-jax more idiomatic

> Consolidated 2026-05-05 from the original architectural plan and the
> `/fp-police` audit findings. Now informed by the
> `purescript` + `purescript-ecosystem` skills, which sharpen several
> of the recommendations.

## What's actually wrong with the current code

The FFI density is fine in isolation — the real cost is that **the
refcount discipline leaks into every call site**. Look at `RMSNorm`
(canonical small case):

```purescript
rmsnorm eps x weight = do
  xR1 <- ref x
  xSq <- square xR1
  m <- meanAxisKeep xSq (-1)
  mEps <- addScalar m eps
  invRms <- rsqrt mEps
  xR2 <- ref x
  scaled <- mul xR2 invRms
  weightR <- ref weight
  mul scaled weightR
```

The math is 4 lines. The plumbing is 5 of those plus 9 explicit names.
Multiply this across `RoPE` (~30 lines for ~6 lines of math),
`Block.attentionForward` (~50 lines for ~10), and the picture's clear:
**the abstraction "every op consumes both arguments" is at the wrong
level**. It belongs inside a tensor-algebra DSL, not at the call site.

Other recurring pain points:

- `Generate.purs`: five near-identical 30-line decode loops
  (greedy / cached / temperature / top-k / top-p / streaming).
- `Sampling.purs`: `sampleLastRow{,Temp,TopK,TopP}` repeat the
  slice-last-row-then-flatten-then-sample ritual four times.
- `Worker.purs` / `Browser.purs`: messages typed via `unsafeCoerce` to
  god-object records; `parseLlamaConfig` is a one-line FFI shim that
  `JSON.parse`s without validation.
- `Test.Main.purs`: 1300+ lines of hand-rolled asserts in one module,
  no isolation, no parallelism.
- `Block.purs`: `(cfg, weights, rope, seqLen)` threaded through ~10
  functions. Half the noise is parameter passing.
- `Sampling.purs` / `Random.purs`: 7 occurrences of `unsafeCoerce` to
  decode `Foreign` from `toJs` calls. Same pattern, no shared shim.
- `Jax.Loaders.Fetch`: callback-style (`onOk` / `onErr`) forces the
  worker into 5-level-nested CPS.

## Phase 0 — Mechanical quick wins (~1 hour, no design changes)

These are the high-confidence fp-police fixes. Land first, since they
shrink the surface for the bigger refactors and lock in regressions
via a project-specific audit rule.

### 0.1 Add `Jax.Coerce` — typed shims for jax-js `Foreign` exits

Replaces 7 `unsafeCoerce` violations with named, locally-checked
operations. Single-file consolidation of an unsafe pattern that
recurs across `Sampling.purs`, `Random.purs`, and `Worker.purs`.

```purescript
module Jax.Coerce
  ( asInt
  , asNumber
  , asArray1D
  , asArray1DInt
  , asArray2D
  ) where

-- Each function documents the precondition (rank/dtype on the
-- producing NDArray). The `unsafeCoerce` is confined to the .js side.
asInt        :: Foreign -> Int                   -- rank-0 int32
asNumber     :: Foreign -> Number                -- rank-0 float
asArray1D    :: Foreign -> Array Number          -- rank-1 float
asArray1DInt :: Foreign -> Array Int             -- rank-1 int32
asArray2D    :: Foreign -> Array (Array Number)  -- rank-2 float
```

Migrate 6 application call sites (Random.purs:42, Sampling.purs:38/79/121/132,
Worker.purs:370-371). **Ecosystem note:** an even better long-term
move is `safe-coerce`'s `coerce` for any case that's a true newtype
unwrap (compiler-checked); `Jax.Coerce` covers the residual
JS-runtime-shape coercions where `coerce` doesn't apply.

**Effort:** ~30 minutes. No public API changes.

### 0.2 Fix the two curried `foreign import`s

```purescript
-- src/Worker.purs:235 — before
foreign import replaceImpl :: String -> String -> String -> String

-- after (FFI rule: use Fn3 for uncurried JS, runFn3 fully saturated)
import Data.Function.Uncurried (Fn3, runFn3)
foreign import replaceImpl :: Fn3 String String String String
replace search replacement input = runFn3 replaceImpl search replacement input
```

Same for `test/Test/Main.purs:146` (`allCloseImpl`).

**Effort:** ~10 minutes total.

### 0.3 `Impl`-suffix the bare test helper + audit rule

`test/Test/Main.purs:145`:

```purescript
foreign import lengthImpl :: forall a. Array a -> Int
length :: forall a. Array a -> Int
length = lengthImpl
```

Then add `.claude/fp-police-rules.md`:

```
## E1. Direct unsafeCoerce outside Jax.Coerce / test helpers (VIOLATION)
All `Foreign → typed` conversions in application code must go through
`Jax.Coerce`. Direct `unsafeCoerce` is allowed only in
`src/Jax/Coerce.{purs,js}` and `test/Test/Main.purs` rank-coercion
helpers.
Pattern: "unsafeCoerce"
Glob: src/**/*.purs
```

Re-run `/fp-police` and confirm zero application-code `unsafeCoerce`.

**Effort:** ~15 minutes.

---

## Phase 1 — `decodeLoop` combinator + `withLastRow` (~2 hours, no new deps)

Pure refactor. Most concentrated payoff in the inference path; no
ecosystem additions.

Right now `generateGreedy`, `generateGreedyCached`,
`generateTemperature`, `generateTopK`, `generateTopP`,
`generateGreedyCachedStream`, and `generateGreedyCachedStreamUntilWithHead`
all repeat:

```
prefill → first sample → loop { encode last → forwardCachedWithHead → sample → snoc → check EOS }
```

Replace with one primitive:

```purescript
type DecodeStep =
  { sample :: NDArray D1 -> Effect Int       -- pluggable: greedy / temp / top-k / top-p
  , onToken :: Int -> Effect Unit             -- streaming hook (mempty for batch)
  , stop :: Int -> Boolean                    -- EOS predicate (const false for none)
  }

decodeLoop
  :: ModelConfig
  -> ModelWeights
  -> NDArray D2          -- LM head
  -> RoPETables
  -> DecodeStep
  -> Array Int           -- prompt
  -> Int                 -- maxNew
  -> Effect (Array Int)
decodeLoop cfg w lmHead rope step prompt maxNew = ...
```

Use `Data.Tailrec.tailRecM` for the loop body (`Step Int -> Effect (Step (Either Int Int))`)
so it's stack-safe regardless of `maxNew`.

All current `generate*` functions become 5–10-line builders that fill
in a `DecodeStep` and call `decodeLoop`. **~250 LOC saved**, plus the
EOS / streaming / explicit-LM-head variants stop diverging.

Add `withLastRow :: NDArray D2 -> (NDArray D1 -> Effect a) -> Effect a`
to fold the slice-last-row-then-flatten-then-sample ritual into one
place. `sampleLastRow{,Temp,TopK,TopP}` all become one-liners over it.

**Effort:** ~2 hours.

---

## Phase 2 — Type-safe message protocol + HF config codec (~3 hours)

Adds `argonaut-codecs` + `argonaut-core` (single dep cluster). Kills
2 application-code `unsafeCoerce` and replaces a one-line `JSON.parse`
FFI with a validated codec.

### Worker / Browser protocol

```purescript
data WorkerIn
  = LoadModel { url :: String, tokenizerUrl :: String }
  | Generate { prompt :: String, maxNew :: Int, debug :: Boolean }
  | Benchmark { benchPrompt :: Array Int, maxNew :: Int }

data WorkerOut
  = Ready { backend :: String }
  | LoadStart { url :: String }
  | LoadTokenizer { vocabSize :: Int }
  | LoadFetched { url :: String, fetchMs :: Number }
  | LoadDone { url :: String, tensorCount :: Int, parseMs :: Number, adaptMs :: Number, totalMs :: Number }
  | LoadError { url :: String, err :: String }
  | GenerateStart { promptIds :: Array Int, promptTokens :: Int }
  | GenerateError { err :: String }
  | TokenText { value :: Int, text :: String, isEos :: Boolean }
  | Done { count :: Int, prefillMs :: Number, decodeMs :: Number, totalMs :: Number, text :: String, stoppedAtEos :: Boolean }
  | BenchResult { backend :: String, ok :: Boolean, count :: Int, prefillMs :: Number, decodeMs :: Number, totalMs :: Number }
  | BenchmarkDone

derive instance Generic WorkerIn _
derive instance Generic WorkerOut _
```

Codec values via `codec-argonaut` (the skill is explicit:
**codec values, not `EncodeJson`/`DecodeJson` instances** — the latter
create orphan-instance problems and invisible behavior):

```purescript
import Data.Codec.Argonaut as CA
import Data.Codec.Argonaut.Record as CAR
import Data.Codec.Argonaut.Variant as CAV

workerInCodec :: CA.JsonCodec WorkerIn
workerInCodec = CA.taggedSum "WorkerIn"
  { "LoadModel": CAR.record { url: CA.string, tokenizerUrl: CA.string }
  , "Generate":  CAR.record { prompt: CA.string, maxNew: CA.int, debug: CA.boolean }
  , "Benchmark": CAR.record { benchPrompt: CA.array CA.int, maxNew: CA.int }
  }
  toVariant fromVariant
```

`postMessage` and `onmessage` get encoder / decoder shims:
`postMessage worker = postMessageImpl worker <<< CA.encode workerInCodec`
and
`onMessage = decode workerOutCodec >=> handle`. `printJsonDecodeError` for
diagnostics on bad messages.

### `parseLlamaConfig` → codec

`Jax.Loaders.Config` currently FFI's `JSON.parse` and plucks fields.
Replace with a codec:

```purescript
modelConfigCodec :: CA.JsonCodec ModelConfig
modelConfigCodec = CAR.record
  { hidden: prop "hidden_size" CA.int
  , nHeads: prop "num_attention_heads" CA.int
  , nKvHeads: prop "num_key_value_heads" CA.int
  , headDim: optionalProp "head_dim" CA.int (computeFromHiddenAndHeads ...)
  , intermediate: prop "intermediate_size" CA.int
  , nLayers: prop "num_hidden_layers" CA.int
  , maxSeqLen: prop "max_position_embeddings" CA.int
  , vocabSize: prop "vocab_size" CA.int
  , ropeTheta: optionalProp "rope_theta" CA.number 10000.0
  , normEps: optionalProp "rms_norm_eps" CA.number 1.0e-6
  }
```

(We'll need a small `optionalProp` helper or use `CAR.optional` with `withDefault`.)

This eliminates ~5 `unsafeCoerce` instances total and gives clear
failure modes ("missing field `hidden_size`" instead of `undefined`
silently propagating).

**Effort:** ~3 hours including dep wiring + spago.lock regen.

---

## Phase 3 — `Aff`-based load pipeline (~1.5 hours)

Replaces `Jax.Loaders.Fetch`'s callback-style API and unwinds the
5-level-nested CPS in `Worker.handleLoadModel`.

### Bridge `fetch` to `Aff`

`web-fetch` (low-level) wraps the browser Fetch API. Combined with
`js-promise-aff`'s `toAff` (or `aff-promise`'s equivalent), we get:

```purescript
fetchBytes :: String -> Aff Foreign       -- ArrayBuffer
fetchText  :: String -> Aff String
```

In Node (for tests), the same surface backed by `node-fetch` or just
`affjax-node`.

### Linearize the worker

```purescript
loadModel :: String -> String -> Aff LoadedModel
loadModel weightsUrl tokenizerUrl = do
  configJson <- fetchText (configUrlFromWeights weightsUrl)
  cfg <- liftEither $ CA.decode modelConfigCodec configJson
  tokBytes <- fetchBytes tokenizerUrl
  tokenizer <- liftEffect $ SBPE.fromBinary tokBytes
  liftEffect $ SBPE.setAddDummyPrefix tokenizer false
  weightsBytes <- fetchBytes weightsUrl
  liftEffect do
    parsed <- parseSafetensors weightsBytes
    ckpt <- loadLlamaWeights cfg parsed
    rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
    pure { weights: ckpt.weights, lmHead: ckpt.lmHead, rope, cfg, tokenizer }
```

That's the entire load pipeline — flat `do`, errors short-circuit
through `Aff`, no nested error handlers. Worker entry becomes
`launchAff_` with `attempt` for surfacing errors as `LoadError` messages.

**Streaming generate** also lifts cleanly: `generateText` becomes
`Aff Unit`, the per-token callback stays as `Effect Unit` via
`liftEffect`.

**Effort:** ~1.5 hours. Mostly mechanical translation; the trickiest
part is `Foreign → Aff` for the fetch (one binding via `js-promise-aff`).

---

## Phase 4 — `Tensor` DSL via `qualified-do` + Free Applicative (~1 day)

The big-impact one. Best done after Phase 1 / 2 / 3 so we know the
surface we're abstracting.

The skill's note on idioms says: **"JSON: codec values, not type
class instances"** — the same logic applies here. We want explicit,
inspectable DSL values, not hidden coercions in `Semiring` instances.

### Design (revised after skill review)

Use `Free` (or hand-rolled equivalent) over a small functor that
captures the operations + a `qualified-do` interpreter:

```purescript
-- src/Jax/Tensor.purs
module Jax.Tensor
  ( T                                       -- abstract; carries rank tag d
  , lit                                     -- :: NDArray d -> T d
  , scalar                                  -- :: Number   -> T D1     (broadcast)
  , add, mul, sub, matmul                   -- arithmetic
  , rsqrt, square, sqrt, sigmoid, silu     -- unary math
  , meanAxis, sumAxis                      -- reductions (rank-changing)
  , reshape, transpose                      -- shape ops
  , build                                   -- :: T d -> Effect (NDArray d)
  , bind, pure, discard                    -- for `Tensor.do`
  ) where
```

Internal representation: a Free Applicative AST. `build` walks it once,
counts uses of each `lit`, emits `ref`-bumps where needed, runs the
underlying `Effect`. That's the whole story — refcount discipline
disappears from user code.

### What the user writes

```purescript
import Jax.Tensor as T

rmsnorm :: forall d. Number -> NDArray d -> NDArray D1 -> Effect (NDArray d)
rmsnorm eps x weight = T.build T.do
  let xT = T.lit x
      wT = T.lit weight
      invRms = T.rsqrt (T.meanAxis (-1) (xT * xT) + T.scalar eps)
  T.pure (xT * invRms * wT)
```

`*` and `+` come from `Semiring` instance on `T`. `T.do` is a
qualified-do block (skill note: "PureScript supports qualified
`do`/`ado` natively") that uses `T.bind` / `T.pure` so we control the
syntax inside without inventing operators.

**Comparison:** RoPE (~30 lines today) → ~10 lines. `attentionForward`
(~50 lines) → ~20. `transformerBlock` (~25 lines) → ~10.

### Why this beats the original two proposals

The original plan offered "Free Applicative" or "use-counting newtype".
The use-counting version was tempting but fragile (the count depends
on the consuming context, not the producer). With a real AST + a
single interpretation pass at `build`, we get the ref-counts right by
construction and we can also fuse trivial chains (e.g. constant
folding, eliminating `(-) x x` etc.) — though we don't have to.

### Migration order

1. Build the DSL with primitive ops + `build` interpreter.
2. Port `RMSNorm` (smallest function) — stress-test the design.
3. Port `RoPE` (2nd smallest, distinct shape).
4. Port `MLP` (rank-uniform, demonstrates compositionality).
5. Port `Block.attentionForward` + `attentionForwardCached`.
6. Audit the remaining Effect-soup spots.

**Effort:** ~1 day. ~30% LOC reduction in NN modules. The DSL itself
is ~150 LOC.

---

## Phase 5 — `purescript-spec` migration (~2 hours, orthogonal)

Move `test/Test/Main.purs` from a 1300-line `do` block of hand-rolled
asserts to BDD-style `Test.Spec`. Add `spec-discovery` so we don't
maintain a master list.

```purescript
-- test/Test.Main.purs
import Test.Spec.Discovery (discover)
import Test.Spec.Reporter.Console (consoleReporter)
import Test.Spec.Runner (runSpec)

main :: Effect Unit
main = launchAff_ do
  Jax.Core.init
  liftEffect $ Jax.Core.setDefaultDevice "wasm"
  specs <- discover "Test\\..*Spec"
  runSpec [consoleReporter] specs

-- test/Test/RoPESpec.purs
spec :: Spec Unit
spec = describe "RoPE" do
  it "cos table shape" do ...
  it "sin table shape" do ...
  it "applyRoPE @pos=0 is identity" do ...
```

Each `testFoo` becomes a `describe` / `it` pair with isolated failure
reporting. Fixture loading (`BpeFixture`) becomes a `beforeAll` hook.
Long-running training tests can be marked `pending` or
`skip`-tagged. **`parallel` blocks** for tests that don't share state.

~1500 LOC turns into ~600 with better diagnostics and a 5-second
turnaround on individual failures (spago test currently re-runs the
entire suite).

**Effort:** ~2 hours.

---

## Smaller improvements bundled across phases

These are too small for their own phase but should land while we're in
the relevant area.

- **`ReaderT (Env := { cfg, weights, rope, lmHead }) Effect`** for
  forward-pass functions (or replace `cfg` threading with `MonadAsk`
  via `Effect.Reader`-like patterns). Removes 3 args from every
  internal helper. ~50 LOC saved. *Bundle into Phase 4.*
- **`Tagged` newtypes** for `TokenId`, `Position`, `Layer`, `VocabSize`
  instead of bare `Int`. Skill confirms: "newtype is zero-cost but
  type-safe". Use `safe-coerce`'s `coerce` for unwrap, not pattern
  matching. *Bundle into Phase 1 (decode loop touches all of these).*
- **`dimAt :: NDArray d -> Int -> Effect Int`** helper. Replaces
  ~20 instances of `Array.head` + `fromMaybe 0`. *Bundle into Phase 4.*
- **`Debug.spy` over `Console.log`** for diagnostics. `Console.log`
  stays for entry-point logging (worker boot). Skill recommends
  `debug` package for tracing. *Bundle into Phase 2.*
- **`filterMap` from `filterable`** wherever we have `mapMaybe`-shaped
  patterns (none in src/, several in tests). *Bundle into Phase 5.*
- **Drop `LongLived`** if it's not catching bugs. Used at one boundary;
  replace with comments. *Bundle into Phase 4.*
- **Move leak-canary diagnostic** out of `Jax/Core.js` into
  `Jax.Debug`. Keep core surface minimal. *Bundle into Phase 4.*
- **`ado` notation** for the prefill+postfill computations in
  `forwardLogits` etc. — they're independent and `ado` is the right
  tool per the skill. *Bundle into Phase 4.*
- **Polymorphic `MonadEffect m`** constraints on training helpers so
  the same code can run in `Aff` for browser or `Effect` for Node tests.
  Skill: "polymorphic constraints work in Aff, HalogenM, etc." *Bundle
  into Phase 3.*

## What we punt on

- **Type-level dimension arithmetic** (turning `D1/D2/D3/D4` into a
  `Nat` index). Real shape-typing is fiddly without
  `prim-typeable`-style equipment. The phantom rank tags catch most
  real bugs; concrete dims would be nice but the cost/benefit is bad.
- **A linear-types story** for resource ownership. PureScript doesn't
  have linear types; the Tensor DSL in Phase 4 handles "consumed
  once" by interpretation, not by types.
- **An MTL stack with `MonadJax`** abstracting over `Effect`/`Aff`.
  The `Aff`-lifted helpers in Phase 3 already buy most of the
  ergonomic win without the abstraction cost.
- **`Run` extensible-effects.** The skill describes Run as "more
  ergonomic for multi-effect scenarios" — we have one effect (tensor
  alloc/dispose), so Run is overkill. Free Applicative + `qualified-do`
  fits better.
- **`profunctor-lenses` for ModelWeights navigation.** Tempting for
  per-layer parameter updates during training, but our training path
  uses pytree gradients, not field-by-field updates. Reach for it
  only when we add online fine-tuning.

## Suggested order + total effort

| # | Phase | Effort | Adds dep? | Prereqs |
|---|-------|--------|-----------|---------|
| 0 | Quick wins (Coerce + Fn-eta + Impl + audit rule) | 1 hr | none | none |
| 1 | `decodeLoop` + `withLastRow` | 2 hr | none | none |
| 2 | Codec'd protocol + config | 3 hr | argonaut-codecs, argonaut-core | none |
| 3 | `Aff` load pipeline | 1.5 hr | aff (already), js-promise-aff *or* aff-promise | Phase 2 |
| 4 | `Tensor` DSL | ~1 day | none | Phases 0-3 ideally |
| 5 | `purescript-spec` tests | 2 hr | spec, spec-discovery | none (orthogonal) |

**Total: ~2 sessions of focused work, plus the Phase-4 day.** Each
phase delivers visible-from-outside improvements, and Phases 0-3 land
without any architectural risk.

## How to use this document

- **Don't** treat this as a binding sequence; revisit before each
  phase. The order assumes nothing else changes.
- **Do** re-run `/fp-police` after each phase. Phase 0's audit rule
  enforces the `Foreign`-coercion convention going forward.
- **Do** snapshot `bunx spago test` results before/after each phase —
  ~50 cases on wasm, current baseline is all-passing.
- **Do** keep `fp-police.md` separately for the audit-derived
  granular checklist; this plan is the consolidated architecture doc.
