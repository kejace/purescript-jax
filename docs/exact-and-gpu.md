# Exact integers, async reads, loops, fusion barriers and custom kernels

Five additions, made for [Eigenbird](https://github.com/kejace/eigenbird)'s
exact integer circuits, and useful wherever jax-js runs integer programs or
needs WebGPU outside a browser tab.

| Module | What it adds | Why |
|---|---|---|
| `Jax.Async` | `dataAff`, `toJsAff`, `blockUntilReady` | jax-js reads WebGPU arrays synchronously only through `OffscreenCanvas`, which Deno and some workers lack; `Jax.Core.dataSync` / `toJs` throw there. Non-consuming, like `Jax.Core`'s readers. |
| `Jax.Int` | int32 arrays (`arrayInt32`, `fullInt32`, `toInt32`), `floorDivide`, `remainder`, shifts, bitwise ops, `maximum` / `minimum`, comparisons and `where_` | Integer programs exact to 2^31 (float32 stops at 2^24). int32 arithmetic wraps like i32 on every backend. |
| `Jax.Jit` | `jitWithBarriers` | jax-js fuses an elementwise value into its consumer unless it reaches more than one materialised node, so a deep graph with shared subexpressions and few outputs becomes one kernel whose expression tree duplicates every shared node: exponential in depth (a 256-op, 73-layer integer circuit exhausted 4 GB). A barrier materialises a value (it becomes an extra output, disposed after each call). |
| `Jax.Loop` | `whileBounded`, `whileBoundedWith`, `fori`, `scan`, `whileLoop` | jax-js has no `while_loop` / `scan`. These unroll at trace time, so under `jit` a loop is one program, not one dispatch per iteration. `whileBounded` is a per-lane while loop (a lane whose condition fails keeps its state: a select, not a branch); `whileBoundedWith` adds a barrier every k iterations (an unrolled loop reading its state twice per step is exactly the exponential case above); `whileLoop` repeats a bounded step until no lane runs. |
| `Jax.WebGPU` | `device`, `gpuBuffer`, `runKernel` | Custom WGSL kernels on jax-js arrays with no copies: inputs are the arrays' own GPU buffers; outputs are fresh jax-js arrays whose buffers the kernel writes, which later jax-js ops read in place. |

`runKernel` writes into the buffers of freshly realised `np.zeros` arrays that
nobody else holds. jax-js's `gpuBuffer` documentation says not to mutate its
buffers; this is the one place that does, relying on jax-js 0.1.x giving a
fresh array its own buffer.

Tests: `Test.Exact` (wasm, `spago test`): int32 exactness and wrapping,
`floorDivide` / `remainder` signs, `where_`, per-lane Collatz lengths with
`whileBoundedWith` under `jitWithBarriers`, and a 40-deep shared chain
(2^40 nodes as a tree) that compiles with barriers. The WebGPU module is
checked in a browser by Eigenbird's `bench/jax-webgpu.html` (a custom kernel
reading and writing jax-js int32 arrays, its output used by jax-js, async
reads, the loop on WebGPU).
