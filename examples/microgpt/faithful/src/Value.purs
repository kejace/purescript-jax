-- | Scalar autograd, ported faithfully from Karpathy's microGPT gist.
-- |
-- | A `Value` is a single number that knows what created it. Every
-- | arithmetic op constructs a new `Value` whose `children` field
-- | holds the inputs and `localGrads` holds the chain-rule
-- | coefficients (∂out/∂child). `backward` walks the tape in
-- | reverse topological order, distributing the upstream gradient
-- | through `localGrads`.
-- |
-- | This is *the* lesson of Karpathy's gist: every line of autograd
-- | is visible. The Reimagined demo (../jax/) outsources the same
-- | computation to JAX's grad and runs ~1000× faster, but here you
-- | can read what JAX is doing.
-- |
-- | Performance: O(N²) per backward pass in the number of ops in the
-- | computation graph. Fine for a 1-layer, 8-context, 32-vocab demo.
-- | Don't even think about scaling.
module Value
  ( Value
  , mk
  , add
  , sub
  , mul
  , neg
  , divv
  , exp
  , log
  , pow
  , backward
  , value
  , gradV
  , setValue
  , zeroGrad
  ) where

import Prelude hiding (add, sub, mul)

import Data.Array as Array
import Data.Maybe (Maybe(..))
import Data.Number as N
import Data.Set (Set)
import Data.Set as Set
import Data.Tuple (Tuple(..))
import Effect (Effect)
import Effect.Ref (Ref)
import Effect.Ref as Ref

-- | A scalar tracked through the computation graph.
-- |
-- | Identity is by Ref pointer (PS Refs aren't Eq directly, so we
-- | tag each Value with a unique Int and compare on that).
newtype Value = Value
  { uid        :: Int
  , dataR      :: Ref Number
  , gradR      :: Ref Number
  , children   :: Array Value
  , localGrads :: Array Number
  }

-- Module-private uid generator. A Ref shared across a single program
-- run. Must be initialized once at startup; we cheat with `unsafe`
-- via lazy init pattern — actually no, we make it a parameter via
-- the Ref API: each `mk` reads-and-bumps the Ref. The Ref lives in
-- the Effect that ultimately wraps the whole program; we initialize
-- it in `Main.main`.
--
-- Trade-off: this state could be threaded through every op as a
-- ReaderT, but that's a lot of plumbing for a debugging convenience.
-- We accept a single module-level Effect.Ref because Karpathy's
-- Python uses `id(self)` which is the same kind of cheating.
foreign import nextUidImpl :: Effect Int

-- | Allocate a fresh Value with no parents (a leaf).
mk :: Number -> Effect Value
mk x = do
  uid <- nextUidImpl
  dataR <- Ref.new x
  gradR <- Ref.new 0.0
  pure $ Value
    { uid, dataR, gradR
    , children: []
    , localGrads: []
    }

-- | Read the current value.
value :: Value -> Effect Number
value (Value v) = Ref.read v.dataR

-- | Read the accumulated gradient.
gradV :: Value -> Effect Number
gradV (Value v) = Ref.read v.gradR

-- | Reset this Value's gradient to zero. Caller is responsible for
-- | walking the parameter list; we don't traverse the graph here
-- | because the optimizer only needs the leaves (parameters).
zeroGrad :: Value -> Effect Unit
zeroGrad (Value v) = Ref.write 0.0 v.gradR

-- | Overwrite this Value's data. Intended only for optimizer
-- | parameter updates; calling this on an interior Value of a
-- | computation graph corrupts the autograd's assumption that
-- | data is fixed during forward + backward.
setValue :: Value -> Number -> Effect Unit
setValue (Value v) x = Ref.write x v.dataR

-- =============================================================================
-- Arithmetic ops — each constructs a new Value tied to its inputs.
-- =============================================================================

-- | Build a new Value from two parents and the chain-rule coefficients.
mkOp :: Number -> Array Value -> Array Number -> Effect Value
mkOp x children localGrads = do
  uid <- nextUidImpl
  dataR <- Ref.new x
  gradR <- Ref.new 0.0
  pure $ Value { uid, dataR, gradR, children, localGrads }

add :: Value -> Value -> Effect Value
add a b = do
  va <- value a
  vb <- value b
  -- ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
  mkOp (va + vb) [ a, b ] [ 1.0, 1.0 ]

sub :: Value -> Value -> Effect Value
sub a b = do
  va <- value a
  vb <- value b
  mkOp (va - vb) [ a, b ] [ 1.0, -1.0 ]

mul :: Value -> Value -> Effect Value
mul a b = do
  va <- value a
  vb <- value b
  -- ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
  mkOp (va * vb) [ a, b ] [ vb, va ]

divv :: Value -> Value -> Effect Value
divv a b = do
  va <- value a
  vb <- value b
  -- ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
  mkOp (va / vb) [ a, b ] [ 1.0 / vb, -va / (vb * vb) ]

neg :: Value -> Effect Value
neg a = do
  va <- value a
  mkOp (-va) [ a ] [ -1.0 ]

exp :: Value -> Effect Value
exp a = do
  va <- value a
  let ea = N.exp va
  mkOp ea [ a ] [ ea ]

log :: Value -> Effect Value
log a = do
  va <- value a
  -- ∂log(a)/∂a = 1/a
  mkOp (N.log va) [ a ] [ 1.0 / va ]

-- | Power with a fixed numeric exponent (n is not a Value). Same as
-- | Karpathy's `__pow__`: enough for `x**0.5`, `x**2`, etc.
pow :: Value -> Number -> Effect Value
pow a n = do
  va <- value a
  -- ∂(a^n)/∂a = n * a^(n-1)
  mkOp (N.pow va n) [ a ] [ n * N.pow va (n - 1.0) ]

-- =============================================================================
-- backward — topological sort + reverse chain rule.
-- =============================================================================

-- | Set this Value's gradient to 1.0 and propagate gradients backward
-- | through every parent. Mirrors PyTorch's `loss.backward()`.
-- |
-- | Algorithm:
-- |
-- |   1. Walk children depth-first to produce a topological ordering
-- |      of every Value reachable from `root`.
-- |   2. Set root.grad = 1.0.
-- |   3. Iterate the topo order in REVERSE. For each node v, for each
-- |      (child, localGrad) pair, accumulate child.grad += v.grad *
-- |      localGrad.
-- |
-- | Idempotent up to a `zeroGrad` of the parameters; calling
-- | `backward` twice without zeroing in between accumulates 2× the
-- | gradient on every leaf.
backward :: Value -> Effect Unit
backward root@(Value rootR) = do
  -- Phase 1: topological sort starting at the root.
  visited <- Ref.new (Set.empty :: Set Int)
  topo    <- Ref.new ([] :: Array Value)
  topoBuild visited topo root
  -- Phase 2: seed root gradient.
  Ref.write 1.0 rootR.gradR
  -- Phase 3: walk in reverse, distributing gradients.
  ts <- Ref.read topo
  -- Reverse and propagate.
  Array.reverse ts # foreachE \(Value v) -> do
    g <- Ref.read v.gradR
    let pairs = Array.zip v.children v.localGrads
    pairs # foreachE \(Tuple (Value cR) local) -> do
      cg <- Ref.read cR.gradR
      Ref.write (cg + g * local) cR.gradR

-- | DFS post-order: append a Value to `topo` only after all children
-- | have been visited. Same shape as the Python `build_topo`.
topoBuild :: Ref (Set Int) -> Ref (Array Value) -> Value -> Effect Unit
topoBuild visitedR topoR v@(Value vR) = do
  visited <- Ref.read visitedR
  case Set.member vR.uid visited of
    true  -> pure unit
    false -> do
      Ref.write (Set.insert vR.uid visited) visitedR
      vR.children # foreachE (topoBuild visitedR topoR)
      Ref.modify_ (\acc -> Array.snoc acc v) topoR

-- Local helpers that don't depend on Prelude's Tuple. We use simple
-- pair-projecting helpers to keep `backward` tight.
foreachE :: forall a. (a -> Effect Unit) -> Array a -> Effect Unit
foreachE f xs = case Array.uncons xs of
  Nothing -> pure unit
  Just { head, tail } -> do
    f head
    foreachE f tail

