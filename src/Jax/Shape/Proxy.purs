-- | Runtime witnesses for type-level shapes and dims.
-- |
-- | A `Tensor (s :: Shape)` carries no value-level shape info — `s` is
-- | a phantom. To talk to FFI ops that need an `Array Int` (e.g.
-- | `reshape`, `zeros`), we reflect the type-level shape to runtime
-- | via `reflectShape`.
-- |
-- | Reflection only works for shapes whose dims are all `Lit`s.
-- | `Var s` dims have no statically-known size, so a `Var`-containing
-- | shape produces a "no instance" error if you try to reflect it.
-- | That's by design: the caller must witness the dynamic size some
-- | other way (a runtime int passed alongside, an `Array Int` argument
-- | to a less-typed sibling op, etc.).
module Jax.Shape.Proxy
  ( SProxy
  , DProxy
  , sProxy
  , dProxy
  , class ReflectShape
  , reflectShape
  , class ReflectDim
  , reflectDim
  ) where

import Prelude

import Data.Reflectable (class Reflectable, reflectType)
import Type.Proxy (Proxy(..))

import Jax.Shape (Shape, SNil, SCons, Dim, Lit)

-- =============================================================================
-- Singletons (kind-polymorphic Proxies, named for clarity)
-- =============================================================================

-- | Runtime witness for a type-level `Shape`. Just `Type.Proxy.Proxy`
-- | with a tighter kind annotation; the alias makes signatures read
-- | naturally.
type SProxy (s :: Shape) = Proxy s

-- | Constructor convenience: `sProxy :: forall s. SProxy s`.
sProxy :: forall (s :: Shape). SProxy s
sProxy = Proxy

-- | Runtime witness for a type-level `Dim`.
type DProxy (d :: Dim) = Proxy d

-- | Constructor convenience.
dProxy :: forall (d :: Dim). DProxy d
dProxy = Proxy

-- =============================================================================
-- Dim reflection
-- =============================================================================

-- | `ReflectDim d` ≡ "dim `d` has a known runtime size". Instance
-- | exists for `Lit n` (any concrete `n`); not for `Var s`.
class ReflectDim (d :: Dim) where
  reflectDim :: DProxy d -> Int

instance reflectDimLit :: Reflectable n Int => ReflectDim (Lit n) where
  reflectDim _ = reflectType (Proxy :: Proxy n)

-- =============================================================================
-- Shape reflection
-- =============================================================================

-- | `ReflectShape s` ≡ "shape `s` is fully concrete (all dims are
-- | `Lit`s)". Walks the cons-list, reflecting each dim. Yields the
-- | runtime `Array Int` the FFI ops expect.
class ReflectShape (s :: Shape) where
  reflectShape :: SProxy s -> Array Int

instance reflectShapeNil :: ReflectShape SNil where
  reflectShape _ = []

instance reflectShapeCons ::
  ( ReflectDim d
  , ReflectShape rest
  ) => ReflectShape (SCons d rest) where
  reflectShape _ =
    [ reflectDim (Proxy :: Proxy d) ]
      <> reflectShape (Proxy :: Proxy rest)
