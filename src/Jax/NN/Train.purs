module Jax.NN.Train
  ( makeCrossEntropyLoss
  ) where

import Prelude hiding (mul)

import Data.Array (head) as Array
import Data.Maybe (fromMaybe)
import Effect (Effect)
import Effect.Uncurried (EffectFn1, mkEffectFn1)
import Jax.Core
  ( D1
  , D2
  , NDArray
  , logSoftmax
  , matmul
  , mean
  , mul
  , mulScalar
  , oneHot
  , ref
  , shape
  , sumAxis
  , transpose
  )
import Jax.NN.Block
  ( ModelConfig
  , ModelWeights
  , transformerBlocksAndNorm
  )
import Jax.NN.RoPE (RoPETables)
import Jax.Shape.Tensor (withRank)

-- | Build a one-arg cross-entropy loss closing over `cfg`, `rope`, the
-- | input prompt, and target token IDs (one per position). Plugs into
-- | `valueAndGradT` for end-to-end transformer training.
-- |
-- | **Autodiff-friendly forward path.** jax-js's `gather` (the primitive
-- | underlying `take`) does not yet have a transpose rule, so the
-- | standard `embed` (= `take`) is not differentiable. We compute the
-- | embedding via `one_hot(ids) @ embedding_table` and the LM head via
-- | `hidden @ embedding_table.T` — both pure matmul, both differentiable.
-- | Mathematically equivalent to the production forward path.
makeCrossEntropyLoss
  :: forall e
   . ModelConfig
  -> RoPETables
  -> NDArray D1
  -> NDArray D1
  -> EffectFn1 ModelWeights (NDArray e)
makeCrossEntropyLoss cfg rope prompt targets = mkEffectFn1 \weights -> do
  promptR <- ref prompt
  promptOh <- oneHot promptR cfg.vocabSize
  embR1 <- ref (withRank weights.embedding :: NDArray D2)
  hidden0 <- matmul promptOh embR1
  hidden0Shape <- shape hidden0
  let seqLen = fromMaybe 0 (Array.head hidden0Shape)
  hiddenN <- transformerBlocksAndNorm cfg weights rope seqLen hidden0
  embR2 <- ref (withRank weights.embedding :: NDArray D2)
  embT <- transpose embR2
  logits <- matmul hiddenN embT
  logProbs <- logSoftmax logits (-1)
  targetsR <- ref targets
  targetsOh <- oneHot targetsR cfg.vocabSize
  masked <- mul logProbs targetsOh
  perPos <- sumAxis masked (-1)
  meanPos <- mean perPos
  mulScalar meanPos (-1.0)
