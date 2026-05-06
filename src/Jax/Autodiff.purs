module Jax.Autodiff
  ( grad
  , valueAndGrad
  , jit
  , vmap
  , ValueAndGrad
  , sumSquareLoss
  , sumSquareTreeLoss
  , gradT
  , valueAndGradT
  ) where

import Effect (Effect)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Jax.Core (NDArray)

-- | Result of `valueAndGrad`: the function's output (typically a scalar
-- | loss of rank `e`) and its gradient with respect to the input
-- | (matching the input's rank `d`).
type ValueAndGrad d e = { value :: NDArray e, grad :: NDArray d }

foreign import gradImpl
  :: forall d e
   . EffectFn1
       (EffectFn1 (NDArray d) (NDArray e))
       (EffectFn1 (NDArray d) (NDArray d))

foreign import valueAndGradImpl
  :: forall d e
   . EffectFn1
       (EffectFn1 (NDArray d) (NDArray e))
       (EffectFn1 (NDArray d) (ValueAndGrad d e))

foreign import jitImpl
  :: forall d e
   . EffectFn1
       (EffectFn1 (NDArray d) (NDArray e))
       (EffectFn1 (NDArray d) (NDArray e))

foreign import vmapImpl
  :: forall d e
   . EffectFn1
       (EffectFn1 (NDArray d) (NDArray e))
       (EffectFn1 (NDArray d) (NDArray e))

-- | Differentiate a 1-arg effectful tensor function. Output is the
-- | gradient at the input point, shaped like the input.
grad
  :: forall d e
   . EffectFn1 (NDArray d) (NDArray e)
  -> Effect (EffectFn1 (NDArray d) (NDArray d))
grad = runEffectFn1 gradImpl

-- | Like `grad`, but returns both the function value and the gradient.
valueAndGrad
  :: forall d e
   . EffectFn1 (NDArray d) (NDArray e)
  -> Effect (EffectFn1 (NDArray d) (ValueAndGrad d e))
valueAndGrad = runEffectFn1 valueAndGradImpl

-- | JIT-compile a 1-arg effectful tensor function for kernel fusion.
jit
  :: forall d e
   . EffectFn1 (NDArray d) (NDArray e)
  -> Effect (EffectFn1 (NDArray d) (NDArray e))
jit = runEffectFn1 jitImpl

-- | Auto-vectorize a 1-arg effectful tensor function over a leading axis.
vmap
  :: forall d e
   . EffectFn1 (NDArray d) (NDArray e)
  -> Effect (EffectFn1 (NDArray d) (NDArray e))
vmap = runEffectFn1 vmapImpl

-- | Pure-JS sum-of-squares loss (`Σ x²`). Exposed as an `EffectFn1` so it
-- | can be passed to `grad`/`valueAndGrad` without being wrapped through
-- | PureScript's `mkEffectFn1`.
foreign import sumSquareLossImpl :: forall d e. EffectFn1 (NDArray d) (NDArray e)

sumSquareLoss :: forall d e. EffectFn1 (NDArray d) (NDArray e)
sumSquareLoss = sumSquareLossImpl

-- | Pytree-shaped sum-of-squares loss:
-- |   `loss({ a, b }) = Σ a² + Σ b²`
-- | Used by the pytree training test/demo.
foreign import sumSquareTreeLossImpl
  :: forall d e f
   . EffectFn1
       { a :: NDArray d, b :: NDArray e }
       (NDArray f)

sumSquareTreeLoss
  :: forall d e f
   . EffectFn1
       { a :: NDArray d, b :: NDArray e }
       (NDArray f)
sumSquareTreeLoss = sumSquareTreeLossImpl

-- =============================================================================
-- Pytree-aware grad / value-and-grad
-- =============================================================================
--
-- jax-js's grad/valueAndGrad accept any `JsTree<Array>` (nested
-- records/arrays of tensors) for input and output. The PS bindings below
-- are typed over arbitrary `a` and `b` so callers can plug in records
-- like `ModelWeights`. The JS shim is the *same* as the rank-typed
-- variants — only the PS type signatures broaden.

foreign import gradImpl_ :: forall a b. EffectFn1 (EffectFn1 a b) (EffectFn1 a a)

foreign import valueAndGradImpl_
  :: forall a b
   . EffectFn1 (EffectFn1 a b) (EffectFn1 a { value :: b, grad :: a })

-- | Differentiate a function with respect to its (pytree-shaped) input.
-- | The output gradient mirrors the input's structure.
gradT :: forall a b. EffectFn1 a b -> Effect (EffectFn1 a a)
gradT = runEffectFn1 gradImpl_

-- | Like `gradT` but returns both the value and the gradient.
valueAndGradT
  :: forall a b
   . EffectFn1 a b
  -> Effect (EffectFn1 a { value :: b, grad :: a })
valueAndGradT = runEffectFn1 valueAndGradImpl_
