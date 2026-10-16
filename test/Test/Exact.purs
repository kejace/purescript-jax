-- | Integers, async reads, loops and fusion barriers (wasm backend; the
-- | WebGPU module is tested in a browser).
module Test.Exact (spec) where

import Prelude hiding (add, mul)

import Data.Array as Array
import Data.Int (toNumber)
import Effect.Aff (Aff)
import Effect.Class (liftEffect)
import Effect.Exception (throw)
import Jax.Async (dataAff)
import Jax.Core (NDArray, D1, ref, dispose, add, mul)
import Jax.Int as I
import Jax.Jit (jitWithBarriers)
import Jax.Loop (fori, whileBoundedWith)
import Test.Spec (Spec, describe, it)

foreign import collatzLengths :: Array Int -> Array Int

expect :: String -> Boolean -> Aff Unit
expect what ok = when (not ok) (liftEffect (throw what))

spec :: Spec Unit
spec = describe "Exact integers, async reads, loops, barriers" do
  it "int32 arithmetic is exact past 2^24 and wraps like i32" do
    a <- liftEffect (I.arrayInt32 [ 16777217, 2147483647, -7, 7 ])
    b <- liftEffect (I.arrayInt32 [ 1, 1, 2, -2 ])
    s <- liftEffect (ref a >>= \a' -> ref b >>= \b' -> add a' b') :: Aff (NDArray D1)
    xs <- dataAff s
    expect ("sum " <> show xs) (xs == [ 16777218.0, -2147483648.0, -5.0, 5.0 ])
    q <- liftEffect (ref a >>= \a' -> ref b >>= \b' -> I.floorDivide a' b') :: Aff (NDArray D1)
    r <- liftEffect (I.remainder a b) :: Aff (NDArray D1)
    qs <- dataAff q
    rs <- dataAff r
    -- floor division and remainder as numpy: -7 // 2 = -4, -7 % 2 = 1; 7 // -2 = -4, 7 % -2 = -1
    expect ("floorDivide " <> show qs) (Array.drop 2 qs == [ -4.0, -4.0 ])
    expect ("remainder " <> show rs) (Array.drop 2 rs == [ 1.0, -1.0 ])
    liftEffect (dispose s *> dispose q *> dispose r)

  it "where and comparisons select per element" do
    x <- liftEffect (I.arrayInt32 [ 3, -4, 0, 9 ])
    z <- liftEffect (I.fullInt32 [ 4 ] 0)
    c <- liftEffect (ref x >>= \x' -> ref z >>= \z' -> I.greater x' z') :: Aff (NDArray D1)
    w <- liftEffect (ref x >>= \x' -> I.negative x >>= \n -> I.where_ c x' n) :: Aff (NDArray D1)
    ws <- dataAff w
    -- where (x > 0) x (-x): |x|
    expect ("where " <> show ws) (ws == [ 3.0, 4.0, 0.0, 9.0 ])
    liftEffect (dispose z *> dispose w)

  it "whileBounded under jit: Collatz lengths per lane, lanes finishing at different steps" do
    let starts = [ 1, 2, 3, 6, 7, 27, 97, 871 ]
    n0 <- liftEffect (I.arrayInt32 starts)
    s0 <- liftEffect (I.fullInt32 [ Array.length starts ] 0)
    let
      running st = case st of
        [ n, _ ] -> I.fullInt32 [ Array.length starts ] 1 >>= \one -> I.greater n one
        _ -> throw "state"
      step st = case st of
        [ n, s ] -> do
          two <- I.fullInt32 [ Array.length starts ] 2
          r <- ref n >>= \n' -> I.remainder n' two
          zero <- I.fullInt32 [ Array.length starts ] 0
          even <- I.equal r zero
          two' <- I.fullInt32 [ Array.length starts ] 2
          half <- ref n >>= \n' -> I.floorDivide n' two'
          up <- I.mulScalarInt n 3 >>= \m -> I.addScalarInt m 1
          n' <- I.where_ even half up
          s' <- I.addScalarInt s 1
          pure [ n', s' ]
        _ -> throw "state"
      loop = jitWithBarriers \barrier st -> whileBoundedWith barrier 4 200 running step st
    out <- liftEffect (loop [ n0, s0 ])
    case out of
      [ n, s ] -> do
        lens <- dataAff s
        expect ("lengths " <> show lens) (lens == map toNumber (collatzLengths starts))
        liftEffect (dispose n *> dispose s)
      _ -> liftEffect (throw "state")

  it "jitWithBarriers bounds fusion: a 40-deep shared chain (2^40 nodes as a tree) compiles" do
    x0 <- liftEffect (I.arrayInt32 [ 1, 2, 3 ])
    let
      -- x := (x * x + x) mod 97, each level using x twice
      level barrier i x = do
        x' <- if i `mod` 4 == 0 then barrier x else pure x
        sq <- ref x' >>= \a -> ref x' >>= \b -> mul a b
        s <- add sq x'
        m <- I.fullInt32 [ 3 ] 97
        I.remainder s m
      f = jitWithBarriers \barrier x -> fori 40 (level barrier) x
    y <- liftEffect (f x0) :: Aff (NDArray D1)
    ys <- dataAff y
    let ref' v = Array.foldl (\a _ -> (a * a + a) `mod` 97) v (Array.range 1 40)
    expect ("chain " <> show ys) (ys == map (toNumber <<< ref') [ 1, 2, 3 ])
    liftEffect (dispose y)
