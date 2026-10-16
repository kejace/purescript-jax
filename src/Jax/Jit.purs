-- | `jit` with fusion barriers, for deep elementwise programs.
-- |
-- | jax-js fuses an elementwise value into its consumer unless it reaches
-- | more than one materialised node, so a deep graph with shared
-- | subexpressions and few outputs becomes one kernel whose expression tree
-- | duplicates every shared node (exponential in depth; a 256-op, 73-layer
-- | integer circuit exhausted 4 GB). Marking values with the `Barrier`
-- | materialises them (they become extra outputs, disposed after each call),
-- | which bounds the fused kernels.
module Jax.Jit
  ( Barrier
  , jitWithBarriers
  ) where

import Prelude

import Effect (Effect)
import Effect.Uncurried (EffectFn1, EffectFn2, mkEffectFn1, mkEffectFn2, runEffectFn1)
import Jax.Core (NDArray)

-- | Materialise a value inside a traced function; returns it unchanged.
type Barrier = forall d. NDArray d -> Effect (NDArray d)

foreign import jitWithBarriersImpl
  :: forall a b. EffectFn2 (EffectFn1 (NDArray Unit) (NDArray Unit)) a b -> EffectFn1 a b

-- | `jitWithBarriers f`: a jitted `x -> out`, where `f` receives the
-- | barrier. `x` and `out` are jax-js pytrees.
jitWithBarriers :: forall a b. (Barrier -> a -> Effect b) -> (a -> Effect b)
jitWithBarriers f =
  let
    g = jitWithBarriersImpl (mkEffectFn2 \barrier x -> f (barrierOf barrier) x)
  in
    runEffectFn1 g
  where
  barrierOf :: EffectFn1 (NDArray Unit) (NDArray Unit) -> Barrier
  barrierOf b = unsafeBarrier (runEffectFn1 b)

foreign import unsafeBarrier :: (NDArray Unit -> Effect (NDArray Unit)) -> Barrier
