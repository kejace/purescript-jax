# FP Police Audit — Findings

> **Superseded by `refactor_plan.md` (Phase 0).** The audit's
> granular fixes have been folded into the consolidated plan.
> This file is preserved as a record of the original findings.

## Audit summary (2026-05-05)

**12 violations, 4 warnings, ~20 style issues across ~7 files in
application code.**

The violations cluster into two structural patterns:

1. **`Foreign → typed` decoding at FFI boundaries** (Sampling, Random,
   Worker diagnostic) — 7 occurrences of `unsafeCoerce`.
2. **Worker/Browser message protocol decoding** — 2 occurrences of
   `unsafeCoerce`.

Plus:

- 1 curried `foreign import` (`replaceImpl` in Worker.purs).
- 1 bare `foreign import length` in test code (no `Impl` suffix).
- 1 curried `foreign import` (`allCloseImpl` in test code).
- ~20 point-free `runEffectFn`-style bindings (cosmetic).

## Where each finding is addressed

| Finding | Plan phase |
|---------|-----------|
| Foreign-decode `unsafeCoerce` (7) | **Phase 0.1** — `Jax.Coerce` module |
| Worker/Browser message `unsafeCoerce` (2) | **Phase 2** — codec'd protocol |
| Curried `foreign import` (Worker, test) | **Phase 0.2** — `Fn3` migration |
| Bare `foreign import length` | **Phase 0.3** — `Impl`-suffix |
| `Effect.Ref` in streaming-buffer | **Phase 4** — Tensor DSL refactor |
| Point-free `runEffectFn` (cosmetic) | Not addressed (no perf evidence) |

## Re-running the audit

After Phase 0 lands, the project rule
`.claude/fp-police-rules.md` (added in Phase 0.3) will fail any new
`unsafeCoerce` introduced outside `Jax.Coerce` or test rank-coercion
helpers. Run `/fp-police` periodically during the larger phases to
confirm no regressions slip in.
