# Project-Specific FP Police Rules

## E1. Direct `unsafeCoerce` outside `Jax.Coerce` / test helpers (VIOLATION)

All `Foreign → typed` conversions in application code must go through
`Jax.Coerce`. Direct `unsafeCoerce` is allowed only inside
`src/Jax/Coerce.{purs,js}` and in `test/Test/Main.purs` rank-coercion
helpers (`asNumber`, `asArray1D`, `asArray2D` near line 135) which own
the rank invariants of the tests they support.

Pattern: `unsafeCoerce`
Glob: `src/**/*.purs`
Allowed-in: `src/Jax/Coerce.purs`

## E2. No direct `Console.log` outside entry-point modules (STYLE)

Worker / Browser / Main are app boundaries — `log` from `Effect.Console`
is correct there. Anywhere else, prefer `Debug.spy` / `Debug.trace` /
`Debug.traceM` so diagnostics survive being moved between effect
contexts (Aff, HalogenM, etc.) and don't need a `Console` import in
library modules.

Pattern: `Console\.log|import.*Effect\.Console`
Glob: `src/**/*.purs`
Allowed-in: `src/Main.purs`, `src/Browser.purs`, `src/Worker.purs`
