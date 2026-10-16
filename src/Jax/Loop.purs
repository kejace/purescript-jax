-- | Loops for traced code. jax-js has no `while_loop` / `scan`: these unroll
-- | at trace time, so under `jit` (or `Jax.Jit.jitWithBarriers`) a loop is
-- | one fused graph rather than one dispatch per iteration.
-- |
-- | `whileBounded` is a per-lane while loop with a fixed bound: a lane whose
-- | condition fails keeps its state (a select), so lanes end at different
-- | iterations without branching. `whileLoop` repeats a bounded step until
-- | no lane is running, reading one bool between steps.
module Jax.Loop
  ( whileBounded
  , whileBoundedWith
  , fori
  , scan
  , whileLoop
  ) where

import Prelude

import Control.Promise (Promise, toAffE)
import Effect (Effect)
import Effect.Aff (Aff)
import Data.Nullable (Nullable, notNull, null)
import Effect.Uncurried (EffectFn1, EffectFn2, EffectFn4, EffectFn6, mkEffectFn1, mkEffectFn2, runEffectFn4, runEffectFn6)
import Jax.Jit (Barrier)
import Jax.Core (NDArray)

foreign import whileBoundedImpl
  :: forall d c. EffectFn6 Int (EffectFn1 (Array (NDArray d)) (NDArray c)) (EffectFn1 (Array (NDArray d)) (Array (NDArray d))) (Array (NDArray d)) (Nullable (EffectFn1 (NDArray d) (NDArray d))) Int (Array (NDArray d))

foreign import foriImpl
  :: forall s. EffectFn4 Int (EffectFn2 Int s s) s Unit s

foreign import scanImpl
  :: forall c x y ys. EffectFn4 Int (EffectFn2 c (NDArray x) { carry :: c, y :: NDArray y }) c (NDArray ys) { carry :: c, ys :: NDArray ys }

foreign import whileLoopAffImpl
  :: forall d c. EffectFn4 (EffectFn1 (Array (NDArray d)) (Array (NDArray d))) (EffectFn1 (Array (NDArray d)) (NDArray c)) (Array (NDArray d)) Int (Promise (Array (NDArray d)))

-- | `whileBounded n cond body init`: at most `n` steps of `body` on the lanes
-- | where `cond` holds. Consumes `init`; `cond` and `body` receive references
-- | they may consume.
whileBounded
  :: forall d c
   . Int
  -> (Array (NDArray d) -> Effect (NDArray c))
  -> (Array (NDArray d) -> Effect (Array (NDArray d)))
  -> Array (NDArray d)
  -> Effect (Array (NDArray d))
whileBounded n cond body init = runEffectFn6 whileBoundedImpl n (mkEffectFn1 cond) (mkEffectFn1 body) init null 0

-- | `whileBounded` that materialises the state every `every` iterations with
-- | the barrier of `Jax.Jit.jitWithBarriers`. Without it, an unrolled loop
-- | whose step reads its state more than once fuses into an expression tree
-- | exponential in the iteration count.
whileBoundedWith
  :: forall d c
   . Barrier
  -> Int
  -> Int
  -> (Array (NDArray d) -> Effect (NDArray c))
  -> (Array (NDArray d) -> Effect (Array (NDArray d)))
  -> Array (NDArray d)
  -> Effect (Array (NDArray d))
whileBoundedWith barrier every n cond body init =
  runEffectFn6 whileBoundedImpl n (mkEffectFn1 cond) (mkEffectFn1 body) init (notNull (mkEffectFn1 barrier)) every

-- | `fori n f init`: `f 0`, ..., `f (n - 1)` threaded through the state.
fori :: forall s. Int -> (Int -> s -> Effect s) -> s -> Effect s
fori n f init = runEffectFn4 foriImpl n (mkEffectFn2 f) init unit

-- | `scan n f init xs`: thread a carry over the `n` slices of `xs` along its
-- | leading axis, stacking each step's `y`. Consumes `xs`.
scan
  :: forall c x y ys
   . Int
  -> (c -> NDArray x -> Effect { carry :: c, y :: NDArray y })
  -> c
  -> NDArray ys
  -> Effect { carry :: c, ys :: NDArray ys }
scan n f init xs = runEffectFn4 scanImpl n (mkEffectFn2 f) init xs

-- | `whileLoop step running init maxChunks`: apply `step` (typically a
-- | jitted `whileBounded`) until `running` holds on no lane, at most
-- | `maxChunks` times.
whileLoop
  :: forall d c
   . (Array (NDArray d) -> Effect (Array (NDArray d)))
  -> (Array (NDArray d) -> Effect (NDArray c))
  -> Array (NDArray d)
  -> Int
  -> Aff (Array (NDArray d))
whileLoop step running init maxChunks = toAffE (runEffectFn4 whileLoopAffImpl (mkEffectFn1 step) (mkEffectFn1 running) init maxChunks)
