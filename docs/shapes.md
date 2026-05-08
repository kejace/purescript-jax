# Shape-typed tensors in `purescript-jax`

> Status: Stages 1–5 of the [type-level shapes plan][plan] shipped.
> Stages 1–2 give you the kinds, the operations, and the deferred DSL.
> Stage 3 demonstrates the system end-to-end on NN-shaped operations.
> Stage 4 adds the documented escape hatches for shapes you can't fully
> prove. Stage 5 is this document.
>
> [plan]: ../.claude/plans/mossy-twirling-lemon.md (project-internal)

## What this gives you

Two layers of tensor types:

```purescript
-- Layer 0: rank-only, the FFI border (UNCHANGED from v1)
foreign import data NDArray :: Type -> Type
data D0 ; data D1 ; data D2 ; data D3 ; data D4

-- Layer 1: shape-typed, the user-facing surface
newtype Tensor (s :: Shape) = ...
```

`Shape` is a cons-list of `Dim`s. A `Dim` is either `Lit Int` (a
concrete size known at compile time) or `Var Symbol` (a named
polymorphic dimension that must unify with other occurrences of the
same Symbol).

```purescript
-- A 768x768 weight matrix:
type W = Tensor (S2 (Lit 768) (Lit 768))

-- A batch of sequences with statically-unknown length:
type X = Tensor (S2 (Var "seq") (Lit 768))

-- Same Var must agree at use sites:
attentionLike
  :: forall s. Tensor (S3 (Var s) (Lit 12) (Lit 64))
  -> Tensor (S3 (Var s) (Lit 12) (Lit 64))
  -> Effect (Tensor (S3 (Var s) (Lit 12) (Lit 64)))
-- Calling with two tensors whose seq dims are bound to different
-- Symbols is a type error.
```

## What's caught at compile time

| Bug class | Example | Caught? |
|---|---|---|
| Rank mismatch | `add :: Tensor (S2 _ _) -> Tensor (S3 _ _ _)` | ✓ |
| Matmul inner-dim mismatch (statically-known) | `Tensor (S2 _ (Lit 768)) **. Tensor (S2 (Lit 256) _)` | ✓ |
| Matmul inner-dim mismatch (Var) | `Tensor (S2 _ (Var "hidden")) **. Tensor (S2 (Var "qDim") _)` | ✓ |
| Reshape product mismatch (Lit-only) | `[3, 4]` → `[5, 6]` | ✓ |
| Transpose orientation | `Tensor (S2 m n) -> Effect (Tensor (S2 n m))` swap forgotten | ✓ |
| Broadcast incompatibility (NumPy rules) | `[3, 4]` ⊕ `[5, 4]` | ✓ |
| Broadcasting `Var` with non-`Lit-1` | `Var "seq"` ⊕ `Lit 3` | ✓ (rejected, conservative) |
| Last/Init/Head/Tail/Replace/Append witness | shape transformations | ✓ |
| Symbolic dim unification across function body | `Var "seq"` here, `Var "batch"` there | ✓ |

## What's NOT caught

Some bug classes need information the type system doesn't have:

| Bug class | Why not | Mitigation |
|---|---|---|
| `nHeads * headDim ≠ hidden` for runtime config | `nHeads`, `headDim`, `hidden` are runtime fields of `ModelConfig`, not type-level Ints | Lift `ModelConfig` to type-level Ints (separate refactor) |
| Slice `startPos + newSeq > maxSeqLen` | Indices are runtime; would need dependent types | Runtime bounds check + `unsafeAssumeShape` |
| Reshape from Var-led shape `[seq, hidden]` to `[seq, nHeads, headDim]` | Product of `[Var "seq", Lit 768]` isn't computable | `reshapeUnchecked` with runtime size array |
| `batch * seq * hidden` total element count for Var-led shapes | Same as above | Same |

These are honest limits. Lifting `ModelConfig` to type-level integers
is the path forward — it makes every model parametric over its
architecture (`forall (h :: Int) (nh :: Int) (hd :: Int). Mul nh hd h
=> Model h nh hd`). That's a project-scale refactor; whether it's
worth doing depends on how much the runtime-config flexibility costs
us in static safety.

## Module layout

```
src/Jax/Shape.purs              -- the kinds + type-level ops
                                --   (RankOf, Product, Append, Last,
                                --    Init, Head, Tail, Replace, DimEq,
                                --    DimMul)
src/Jax/Shape/Broadcast.purs    -- NumPy-style broadcasting class
src/Jax/Shape/Proxy.purs        -- SProxy / DProxy + ReflectShape
src/Jax/Shape/Tensor.purs       -- the Tensor newtype + escape hatches
src/Jax/Shape/Tensor/Op.purs    -- typed primitive ops (matmul, transpose,
                                --   reshape, slice, broadcast +/-/*, …)
src/Jax/Shape/Tensor/Expr.purs  -- typed deferred-allocation DSL
                                --   (T s, lit, run, +. -. *. **.)
```

## How the layers compose at the FFI boundary

`Tensor s` and rank-only `NDArray d` share a runtime representation:
both are opaque jax-js handles. The bridge is *phantom-only*
coercion — no JS work, no extra allocation:

```purescript
unsafeAssumeShape :: forall s d. NDArray d -> Tensor s     -- claim a shape
unsafeForgetShape :: forall s d. Tensor s -> NDArray d     -- discard the claim
withRank          :: forall s d. RankOf s d => Tensor s -> NDArray d
                                                           -- safe: rank derived
```

The first two are `unsafeCoerce` under the hood and confined to
`src/Jax/Shape/Tensor.purs` (carved out in `.claude/fp-police-rules.md`).
`withRank` derives the rank witness from the shape via `RankOf`, so
its only "unsafe" property is that the rank-tagged NDArray it returns
is *runtime-correct* by construction.

Typed ops in `Jax.Shape.Tensor.Op` wrap rank-only `Jax.Core` ops:

```purescript
matmul a b = do
  result <- Core.matmul (withRank a :: NDArray D2) (withRank b :: NDArray D2)
  pure (unsafeAssumeShape result)
```

Each typed op chooses the rank witness for its input and asserts the
output shape from the shape-correct signature.

## Choosing between `Lit` and `Var`

Use `Lit n` when the size is fixed by the program (a tokenizer's
vocab, a fixed-arity layer in a hand-rolled net, a constant attention
head count). The compiler can check products, fold dimensions, and
catch off-by-one errors.

Use `Var "name"` when the size is set by runtime config or input data
(sequence length, batch size, vocabulary loaded from a tokenizer).
Same-name `Var`s unify; different names don't. The compiler ensures
*consistency* across uses but can't compute with the size.

In production NN code most dimensions end up as `Var`. The `Lit`
machinery shines in:

- Hand-coded test fixtures and demos (see `test/Test/ShapeNN.purs`)
- Layers with fixed structure (`Linear`, `MultiHeadAttention` with
  literal hyperparameters)
- Loaders that target a specific architecture variant (e.g. Smol-
  Llama-101M with `hidden = 768`)

## Reshape policy

Three reshape variants, in order of strictness:

```purescript
-- Strict: total element count proved. Both shapes must be Lit-only.
reshape         :: Product s n => Product s' n => ReflectShape s'
                => SProxy s' -> Tensor s -> Effect (Tensor s')

-- Caller-asserted: runtime size array; result shape assertion. For
-- shapes containing Var dims (most NN code).
reshapeUnchecked :: Array Int -> Tensor s -> Effect (Tensor s')

-- Last-axis only: split or merge the last axis with statically-
-- known new size.
sliceLastAxis   :: Reflectable k Int => Init s inner
                => Append inner (S1 (Lit k)) s'
                => Proxy k -> Int -> Tensor s -> Effect (Tensor s')
```

A future Stage would add `reshapeAlongHead` (preserves the leading
Var axis, requires Product equality on the tail) and
`reshapeAlongLast` (preserves the leading axes, transforms the last
few). Both are doable; neither was load-bearing for the migration to
date so they're deferred.

## Escape-hatch policy

The unsafe boundaries are:

1. **`unsafeAssumeShape` / `unsafeForgetShape`** in
   `Jax.Shape.Tensor`. Used at FFI boundaries (loaders, allocators,
   readback). Carved out in `fp-police-rules.md`.

2. **`reshapeUnchecked`** for Var-containing reshapes. Documented as
   "caller asserts the runtime size array matches the result shape."
   Marked `Unchecked` so it stands out in code review.

3. **Rank-open op signatures** in `Jax.Core` (`add :: forall d e f.`,
   `reshape :: forall d e.`). Inherited from v1; not in scope for
   shape work.

When you reach for an escape, leave a one-line comment naming what
external fact (config dim, parser output, etc.) makes the assertion
true. fp-police flags un-commented uses on audit.

## What Stage 6+ might do

Roughly in increasing order of investment:

1. **Tightened reshape variants**: `reshapeAlongHead` and friends.
   Eliminates most uses of `reshapeUnchecked` in NN code where the
   leading dim is `Var`-symbolic but the tail products match.

2. **Type-level `ModelConfig`**: lift `hidden`, `nHeads`, `headDim`,
   `intermediate`, `vocabSize` to type-level `Int`s. With `Mul nh hd
   h` constraints, the attention reshapes in `Jax.NN.Block` become
   compile-time-checked. Migration cost: every `forwardLogits`-like
   function gains 5+ type parameters. Worth it for correctness,
   painful for ergonomics. Realistic with `forall (cfg ::
   ModelShape).` pseudo-records.

3. **Dependent-ish slice indices**: prove `startPos + newSeq ≤
   maxSeqLen` via `LessEq` constraints witnessed by `Reflectable`
   integers. Pretty tractable for prefill/decode boundaries; fiddly
   to plumb through.

4. **Migrate `Jax.NN.Block` to typed weights**: depends on (2). The
   prize the original plan listed.

5. **Migrate `Jax.NN.RoPE` internals**: low-value once (1) and (2)
   land — at that point RoPE's slice/concat dance is type-correct
   "for free" via the typed sliceLastAxis + concat ops.

The system as it stands is enough to write *new* shape-typed code
end-to-end (see `test/Test/ShapeNN.purs`). Existing rank-only code
keeps working unchanged.
