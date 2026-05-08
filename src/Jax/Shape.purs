-- | Type-level shape arithmetic for `Tensor`.
-- |
-- | A `Shape` is a type-level cons-list of `Dim`s; a `Dim` is either a
-- | concrete `Lit Int` or a polymorphic `Var Symbol` (a name that must
-- | unify consistently within a function's scope, e.g. `Var "seq"`).
-- |
-- |     Tensor (S2 (Lit 768) (Lit 768))               -- a 768x768 matrix
-- |     Tensor (S3 (Var "seq") (Lit 12) (Lit 64))     -- [seq, 12, 64]
-- |
-- | Operations on `Shape` are expressed as multi-parameter type classes
-- | with functional dependencies. The compiler solves them at the use
-- | site; if a constraint can't be discharged, that's a type error.
-- |
-- | The compiler doesn't bake in arithmetic for *shapes* — only for
-- | individual `Int` parameters via `Prim.Int.{Add, Mul, Compare}`.
-- | We lift those scalar operations to shapes here.
module Jax.Shape
  ( -- * The Shape kind
    Shape
  , SNil
  , SCons
  , type (:::)
    -- * The Dim kind
  , Dim
  , Lit
  , Var
    -- * Shape aliases (S0 = scalar, S1..S4 = rank 1..4)
  , S0
  , S1
  , S2
  , S3
  , S4
    -- * Type-level shape operations
  , class CountAxes
  , class RankWitness
  , class RankOf
  , class Product
  , class Append
  , class Last
  , class Init
  , class Head
  , class Tail
  , class Replace
    -- * Pair-level dim operations (used by Broadcast and shape transforms)
  , class DimEq
  , class DimMul
  ) where

import Prim.Int (class Add, class Mul)

import Jax.Core (D0, D1, D2, D3, D4)

-- =============================================================================
-- Kinds
-- =============================================================================

-- | The kind of shapes — a cons-list of `Dim`s.
foreign import data Shape :: Type

-- | The empty shape (a scalar tensor has shape `SNil`).
foreign import data SNil :: Shape

-- | Cons a `Dim` onto a `Shape`.
foreign import data SCons :: Dim -> Shape -> Shape

-- | Right-associative type operator for `SCons`. Reads inside-out as
-- | NumPy shape literals do: `Lit 3 ::: Lit 4 ::: SNil` is `[3, 4]`.
infixr 6 type SCons as :::

-- | The kind of a single axis size. Either a concrete literal `Int`
-- | (`Lit 768`) or a Symbol-named variable (`Var "seq"`) that the
-- | type system unifies across uses.
foreign import data Dim :: Type

-- | A statically-known axis size.
foreign import data Lit :: Int -> Dim

-- | A symbolic axis size: identical Symbols unify, distinct Symbols
-- | don't. Used for runtime-determined dims (sequence length, batch).
foreign import data Var :: Symbol -> Dim

-- =============================================================================
-- Shape aliases
-- =============================================================================

-- | Scalar shape (rank 0). Equivalent to `SNil`.
type S0 = SNil

-- | Rank-1 shape: `[a]`.
type S1 a = SCons a SNil

-- | Rank-2 shape: `[a, b]`.
type S2 a b = SCons a (SCons b SNil)

-- | Rank-3 shape: `[a, b, c]`.
type S3 a b c = SCons a (SCons b (SCons c SNil))

-- | Rank-4 shape: `[a, b, c, d]`.
type S4 a b c d = SCons a (SCons b (SCons c (SCons d SNil)))

-- =============================================================================
-- Rank computation: Shape -> rank witness
-- =============================================================================

-- | `CountAxes s n` ≡ "shape `s` has `n` axes". Pure type-level
-- | `length`, computed via `Prim.Int.Add`.
class CountAxes (s :: Shape) (n :: Int) | s -> n

instance countAxesNil :: CountAxes SNil 0

instance countAxesCons ::
  ( CountAxes rest k
  , Add k 1 n
  ) => CountAxes (SCons d rest) n

-- | `RankWitness n d` ≡ "the rank-`n` runtime witness type is `d`".
-- | Bridges type-level Int to the pre-existing D0..D4 phantoms.
-- | Stops at D4 because so does `Jax.Core` — bumping the ceiling is
-- | one line on each side.
class RankWitness (n :: Int) d | n -> d

instance rankWitness0 :: RankWitness 0 D0
instance rankWitness1 :: RankWitness 1 D1
instance rankWitness2 :: RankWitness 2 D2
instance rankWitness3 :: RankWitness 3 D3
instance rankWitness4 :: RankWitness 4 D4

-- | The composed alias: `RankOf s d` ≡ "shape `s` has runtime rank `d`".
-- | Lets a function that needs an `NDArray d` derive `d` from the shape
-- | alone: `forall s d. RankOf s d => Tensor s -> NDArray d`.
class RankOf (s :: Shape) d | s -> d

instance rankOfDerived ::
  ( CountAxes s n
  , RankWitness n d
  ) => RankOf s d

-- =============================================================================
-- Total element count: Shape -> Int
-- =============================================================================

-- | `Product s n` ≡ "the product of all dim literals in `s` is `n`".
-- | Used by `reshape` to enforce element-count preservation. Only
-- | resolves when every dim in `s` is a `Lit` — `Var`-containing
-- | shapes don't have a known product, so `Product` constraints don't
-- | discharge for them. This is by design: callers carrying `Var` dims
-- | through reshape must witness equality some other way (e.g. via
-- | `Append` / `Replace` rewrites).
class Product (s :: Shape) (n :: Int) | s -> n

instance productNil :: Product SNil 1

instance productConsLit ::
  ( Product rest k
  , Mul a k n
  ) => Product (SCons (Lit a) rest) n

-- =============================================================================
-- Shape transformations: append, last, init, head, tail, replace
-- =============================================================================

-- | `Append a b c` ≡ "`a ++ b = c`" at the shape level.
class Append (a :: Shape) (b :: Shape) (c :: Shape) | a b -> c

instance appendNil :: Append SNil b b

instance appendCons ::
  ( Append rest b cTail
  ) => Append (SCons d rest) b (SCons d cTail)

-- | `Last s d` ≡ "`d` is the last dim of `s`". Defined for non-empty
-- | shapes; a constraint over `SNil` won't discharge.
class Last (s :: Shape) (d :: Dim) | s -> d

instance lastSingleton :: Last (SCons d SNil) d

instance lastCons ::
  ( Last (SCons d2 rest) dLast
  ) => Last (SCons d1 (SCons d2 rest)) dLast

-- | `Init s s'` ≡ "`s'` is `s` with the last dim dropped". Symmetric
-- | partner to `Last`.
class Init (s :: Shape) (s' :: Shape) | s -> s'

instance initSingleton :: Init (SCons d SNil) SNil

instance initCons ::
  ( Init (SCons d2 rest) inner
  ) => Init (SCons d1 (SCons d2 rest)) (SCons d1 inner)

-- | `Head s d` ≡ "`d` is the first dim of `s`".
class Head (s :: Shape) (d :: Dim) | s -> d

instance headCons :: Head (SCons d rest) d

-- | `Tail s s'` ≡ "`s'` is `s` with the first dim dropped".
class Tail (s :: Shape) (s' :: Shape) | s -> s'

instance tailCons :: Tail (SCons d rest) rest

-- | `Replace n s d s'` ≡ "replacing axis `n` (0-indexed) of `s` with
-- | dim `d` yields `s'`". Used by `sliceAxis` and `oneHot`.
class Replace (n :: Int) (s :: Shape) (d :: Dim) (s' :: Shape) | n s d -> s'

instance replaceZero ::
  Replace 0 (SCons _old rest) dNew (SCons dNew rest)

else instance replaceSucc ::
  ( Add k 1 n
  , Replace k rest dNew restNew
  ) => Replace n (SCons d rest) dNew (SCons d restNew)

-- =============================================================================
-- Dim-level helpers (used by Broadcast in a sibling module)
-- =============================================================================

-- | `DimEq a b` ≡ "dims `a` and `b` are the same". Trivially solved
-- | by unification; exposed as a class so error messages can be hand-
-- | crafted at use sites (e.g. matmul's inner-dim check).
class DimEq (a :: Dim) (b :: Dim)

instance dimEqRefl :: DimEq a a

-- | `DimMul a b c` ≡ "`a * b = c`" at the dim level. Both must be
-- | `Lit`s — symbolic dim products aren't supported.
class DimMul (a :: Dim) (b :: Dim) (c :: Dim) | a b -> c

instance dimMulLit ::
  ( Mul x y z
  ) => DimMul (Lit x) (Lit y) (Lit z)
