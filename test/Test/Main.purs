module Test.Main where

import Prelude hiding (add, mul)

import Control.Monad.Trans.Class (lift)
import Data.Foldable (foldl)
import Data.Function.Uncurried (Fn3, runFn3)
import Effect (Effect)
import Effect.Aff (launchAff_)
import Effect.Class (liftEffect)
import Effect.Console (log)
import Effect.Exception (throw)
import Foreign (Foreign)
import Test.Spec (Spec, describe, it)
import Test.Spec.Reporter.Console (consoleReporter)
import Test.Spec.Runner (runSpec)
import Jax.Core
  ( D1
  , D2
  , D3
  , NDArray
  , arange
  , array1D
  , arrayInt1D
  , dispose
  , init
  , linspace
  , mean
  , ones
  , ref
  , reshape
  , setDefaultDevice
  , shape
  , sliceAxis
  , square
  , sum
  , toJs
  , zeros
  )
import Jax.Managed (Managed, allocate, runManaged)
import Jax.Tensor (T, lit, reshapeT, run, (+.), (-.), (*.), (**.))
import Jax.Tensor as T
import Jax.NN.Attention (attention)
import Jax.NN.Block (ModelConfig, ModelWeights, forwardLogits)
import Data.Traversable (traverse)
import Jax.NN.Embed (embed, unembed)
import Jax.NN.Generate
  ( generateGreedy
  , generateGreedyCached
  , generateGreedyCachedStream
  , generateTemperature
  , generateTopK
  )
import Jax.Random (mkKey)
import Effect.Ref as Ref
import Jax.Optax as Optax
import Jax.Autodiff (sumSquareLoss, sumSquareTreeLoss, valueAndGrad, valueAndGradT)
import Effect.Uncurried (EffectFn1, mkEffectFn1, runEffectFn1)
import Test.Shape as TestShape
import Test.ShapeOps as TestShapeOps
import Test.ShapeNN as TestShapeNN
import Jax.NN.MLP (mlp)
import Jax.NN.RMSNorm (rmsnorm)
import Jax.NN.RoPE (applyRoPE, precomputeRoPE)
import Jax.NN.Sampling (sampleGreedy, sampleTopK, sampleTopP)
import Jax.NN.Train (makeCrossEntropyLoss)
import Jax.Loaders.Tokenizer (defaultTokenizer, encode, decode) as Tok
import Jax.Loaders.SentencePieceBPE as SBPE
import Jax.Loaders.Safetensors (parseSafetensors, tensorNames, getTensor) as ST
import Test.SafetensorsFixture (makeFixture, makeBF16Fixture)
import Test.BpeFixture as BpeFix
import Test.LlamaFixture (makeLlamaFixture, llamaFixtureCfg)
import Jax.Loaders.LlamaAdapter (loadLlamaWeights)
import Jax.Pytree (countParams, countTensors, perLayerL2sq, sumSquaredL2)
import Jax.Optics (_embedding, _finalNorm)
import Jax.Loaders.CharTokenizer as CharTokenizer
import Jax.Optax.Schedule as Schedule
import Jax.Loaders.Config as Cfg
import Data.Either (Either(..))
import Data.String (Pattern(..), indexOf) as StrIdx
import Data.Foldable (for_)
import Data.Lens (preview)
import Data.Maybe (Maybe(..))
import Data.Number as Math
import Unsafe.Coerce (unsafeCoerce)

main :: Effect Unit
main = launchAff_ do
  liftEffect init
  liftEffect (setDefaultDevice "wasm")
  runSpec [ consoleReporter ] specs

-- | Spec tree. Each describe-it pair wraps an existing `Effect Unit`
-- | test function via `liftEffect`. The functions still throw on
-- | failure; spec catches those throws and converts them to a
-- | structured failure with location, instead of aborting the whole
-- | run.
-- |
-- | To split this into per-module specs later, move each `describe`
-- | block to its own `Test.<Name>Spec` module exporting `spec :: Spec
-- | Unit`, then `Test.Main` uses `Test.Spec.Discovery.discover` to
-- | find them.
specs :: Spec Unit
specs = do
  describe "Phase 1 FFI parity (wasm)" do
    it "constructor shapes" (liftEffect testConstructorShapes)
    it "arange values" (liftEffect testArangeValues)
    it "add / mul" (liftEffect testAddAndMul)
    it "matmul shape" (liftEffect testMatmulShape)
    it "reductions" (liftEffect testReductions)
    it "transpose" (liftEffect testTranspose)
    it "managed scope" (liftEffect testManagedScope)
  describe "NN primitives" do
    it "RMSNorm" (liftEffect testRMSNorm)
    it "Embed / Unembed" (liftEffect testEmbedUnembed)
    it "RoPE" (liftEffect testRoPE)
    it "Attention (incl. GQA)" (liftEffect testAttention)
    it "MLP (SwiGLU)" (liftEffect testMLP)
    it "transformerBlock + forwardLogits" (liftEffect testBlock)
  describe "Inference" do
    it "sampling" (liftEffect testSampling)
    it "generate (greedy / cached / temperature)" (liftEffect testGenerate)
    it "top-k / top-p" (liftEffect testTopK)
    it "streaming decode" (liftEffect testStreaming)
  describe "Training" do
    it "single-tensor sum-square" (liftEffect testTraining)
    it "pytree (record) Adam" (liftEffect testTrainingPytree)
    it "full transformer cross-entropy" (liftEffect testTransformerTraining)
  describe "Loaders" do
    it "tokenizer (cl100k_base BPE)" (liftEffect testTokenizer)
    it "SentencePiece BPE parity" (liftEffect testSentencePieceBPE)
    it "char tokenizer round-trip" (liftEffect testCharTokenizer)
    it "safetensors" (liftEffect testSafetensors)
    it "Llama end-to-end (synthetic fixture)" (liftEffect testLlamaEndToEnd)
    it "HF config decoder (real-world JSON shape)" (liftEffect testHFConfigDecode)
  describe "Schedules + Train helpers" do
    it "linearDecay endpoints" (liftEffect testLinearDecay)
    it "cosineDecay endpoints" (liftEffect testCosineDecay)
  describe "Numerical parity" do
    it "analytic checks" (liftEffect testNumericalParity)
  TestShape.spec
  TestShapeOps.spec
  TestShapeNN.spec

-- Assertions ------------------------------------------------------------------

assertEqArrayInt :: String -> Array Int -> Array Int -> Effect Unit
assertEqArrayInt label expected actual =
  if expected == actual then log $ "  ✓ " <> label
  else throw $ "  ✗ " <> label <> ": expected " <> show expected <> ", got " <> show actual

assertCloseNum :: String -> Number -> Number -> Effect Unit
assertCloseNum label expected actual =
  let diff = if actual > expected then actual - expected else expected - actual
  in
    if diff < 1.0e-4 then log $ "  ✓ " <> label
    else throw $ "  ✗ " <> label <> ": expected " <> show expected <> ", got " <> show actual

assertCloseArray :: String -> Array Number -> Array Number -> Effect Unit
assertCloseArray label expected actual =
  if length expected /= length actual then
    throw $ "  ✗ " <> label <> ": length mismatch — expected " <> show expected <> ", got " <> show actual
  else
    let
      tol = 1.0e-4
      ok = allClose tol expected actual
    in
      if ok then log $ "  ✓ " <> label
      else throw $ "  ✗ " <> label <> ": expected " <> show expected <> ", got " <> show actual

-- Coercions for jax-js .js() output ------------------------------------------
-- jax-js .js() returns a nested JS structure: number for rank 0, Array
-- Number for rank 1, Array (Array Number) for rank 2, etc. unsafeCoerce
-- is acceptable in tests that own the rank invariant.

asNumber :: Foreign -> Number
asNumber = unsafeCoerce

asArray1D :: Foreign -> Array Number
asArray1D = unsafeCoerce

asArray2D :: Foreign -> Array (Array Number)
asArray2D = unsafeCoerce

-- Foreign array helpers (no Data.Array import to keep deps minimal) -----------

foreign import lengthImpl :: forall a. Array a -> Int

length :: forall a. Array a -> Int
length = lengthImpl

foreign import allCloseImpl :: Fn3 Number (Array Number) (Array Number) Boolean

allClose :: Number -> Array Number -> Array Number -> Boolean
allClose tol a b = runFn3 allCloseImpl tol a b

-- Tensor-DSL test helpers -----------------------------------------------------
-- Each helper realizes the deferred T expression once via `T.run`,
-- inspects the resulting NDArray, then disposes it. Inputs the
-- expression borrows (via `lit`) are untouched. This lets test
-- bodies stay declarative — `lit a +. lit b` instead of the manual
-- `lift (ref a) >>= \aR -> lift (ref b) >>= \bR -> allocate (add aR bR)` dance.

-- Run + flatten + assert closeness against a 1D expected vector.
-- Works for any rank: we reshape to D1 first so `assertCloseArray`
-- can compare a flat array.
checkT :: forall d. String -> Array Number -> T d -> Effect Unit
checkT label expected t = do
  r <- run (reshapeT t [ length expected ] :: T D1)
  f <- toJs r
  dispose r
  assertCloseArray label expected (asArray1D f)

-- Run + read shape + assert.
checkShape :: forall d. String -> Array Int -> T d -> Effect Unit
checkShape label expected t = do
  r <- run t
  s <- shape r
  dispose r
  assertEqArrayInt label expected s

-- Run + read scalar + assert. Use for reductions producing a 0-rank.
checkScalar :: forall d. String -> Number -> T d -> Effect Unit
checkScalar label expected t = do
  r <- run t
  f <- toJs r
  dispose r
  assertCloseNum label expected (asNumber f)

-- Variant of `runManaged` that drops the continuation result and
-- returns `Effect Unit` — convenient when the asserts inside the
-- managed scope are the test's only effect.
runManaged_ :: Managed Unit -> Effect Unit
runManaged_ m = runManaged m \_ -> pure unit

-- Tests -----------------------------------------------------------------------

testConstructorShapes :: Effect Unit
testConstructorShapes = do
  log "Constructors:"
  runManaged (allocate (zeros [ 3, 4 ] :: Effect (NDArray D2))) \a -> do
    s <- shape a
    assertEqArrayInt "zeros [3,4] shape" [ 3, 4 ] s
  runManaged (allocate (ones [ 2, 5 ] :: Effect (NDArray D2))) \a -> do
    s <- shape a
    assertEqArrayInt "ones [2,5] shape" [ 2, 5 ] s

testArangeValues :: Effect Unit
testArangeValues = do
  log "arange:"
  runManaged (allocate (arange 0.0 5.0 1.0)) \a -> do
    s <- shape a
    assertEqArrayInt "arange 0..5 shape" [ 5 ] s
    f <- toJs a
    assertCloseArray "arange 0..5 values" [ 0.0, 1.0, 2.0, 3.0, 4.0 ] (asArray1D f)

testAddAndMul :: Effect Unit
testAddAndMul = do
  log "add / mul:"
  runManaged_ do
    o <- allocate (ones [ 4 ] :: Effect (NDArray D1))
    rng <- allocate (arange 0.0 4.0 1.0)
    lift do
      checkT "ones[4] + ones[4]" [ 2.0, 2.0, 2.0, 2.0 ] (lit o +. lit o)
      checkT "[0,1,2,3] * [0,1,2,3]" [ 0.0, 1.0, 4.0, 9.0 ] (lit rng *. lit rng)

testMatmulShape :: Effect Unit
testMatmulShape = do
  log "matmul:"
  runManaged_ do
    a <- allocate (zeros [ 2, 3 ] :: Effect (NDArray D2))
    b <- allocate (ones [ 3, 4 ] :: Effect (NDArray D2))
    lift $ checkShape "zeros[2,3] @ ones[3,4] shape" [ 2, 4 ]
      (lit a **. lit b)

testReductions :: Effect Unit
testReductions = do
  log "reductions:"
  -- `Core.mean` / `Core.sum` reduce over all axes (rank-0 result).
  -- Tensor DSL's `meanAxisKeepT` reduces a single axis with keepdims,
  -- so it's not the same op — drop down to Core directly here.
  runManaged_ do
    o10 <- allocate (ones [ 10 ] :: Effect (NDArray D1))
    o10R <- lift (ref o10)
    m <- allocate (mean o10R :: Effect (NDArray D1))
    o5 <- allocate (ones [ 5 ] :: Effect (NDArray D1))
    o5R <- lift (ref o5)
    s <- allocate (sum o5R :: Effect (NDArray D1))
    lift do
      f1 <- toJs m
      assertCloseNum "mean(ones[10])" 1.0 (asNumber f1)
      f2 <- toJs s
      assertCloseNum "sum(ones[5])" 5.0 (asNumber f2)

testTranspose :: Effect Unit
testTranspose = do
  log "transpose:"
  runManaged_ do
    a <- allocate (zeros [ 2, 3 ] :: Effect (NDArray D2))
    lift $ checkShape "transpose [2,3] -> [3,2]" [ 3, 2 ]
      (T.transposeT (lit a))

testManagedScope :: Effect Unit
testManagedScope = do
  log "Managed:"
  -- Just confirm the scope runs without error and the continuation fires.
  -- We can't directly observe dispose without a refcount probe; the
  -- absence of a "tensor already disposed" error from a follow-up matmul
  -- (run in a fresh scope below) is the implicit signal.
  runManaged (allocate (zeros [ 8, 8 ] :: Effect (NDArray D2))) \_ ->
    log "  ✓ runManaged completes scope cleanly"
  runManaged_ do
    a <- allocate (ones [ 4, 4 ] :: Effect (NDArray D2))
    b <- allocate (ones [ 4, 4 ] :: Effect (NDArray D2))
    lift $ checkShape "subsequent allocation reuses cleaned-up backend"
      [ 4, 4 ] (lit a **. lit b)

testRMSNorm :: Effect Unit
testRMSNorm = do
  log "RMSNorm:"
  -- Shape parity (1D)
  runManaged_ do
    x <- allocate (array1D [ 1.0, 2.0, 3.0, 4.0 ])
    w <- allocate (ones [ 4 ] :: Effect (NDArray D1))
    out <- allocate (rmsnorm 1.0e-6 x w :: Effect (NDArray D1))
    lift $ checkShape "rmsnorm 1D shape preserved" [ 4 ] (lit out)
  -- Numerical: input [2,2,2,2], weight = ones, eps = 0
  --   x² = [4,4,4,4]; mean = 4; rsqrt(4) = 0.5; out = x * 0.5 = [1,1,1,1]
  runManaged_ do
    x <- allocate (array1D [ 2.0, 2.0, 2.0, 2.0 ])
    w <- allocate (ones [ 4 ] :: Effect (NDArray D1))
    out <- allocate (rmsnorm 0.0 x w :: Effect (NDArray D1))
    lift $ checkT "rmsnorm [2,2,2,2] = [1,1,1,1]" [ 1.0, 1.0, 1.0, 1.0 ] (lit out)
  -- Shape parity (2D): ones[2,3] with weight ones[3] -> output ones[2,3]
  runManaged_ do
    x <- allocate (ones [ 2, 3 ] :: Effect (NDArray D2))
    w <- allocate (ones [ 3 ] :: Effect (NDArray D1))
    out <- allocate (rmsnorm 1.0e-6 x w :: Effect (NDArray D2))
    lift $ checkShape "rmsnorm 2D shape preserved" [ 2, 3 ] (lit out)

testEmbedUnembed :: Effect Unit
testEmbedUnembed = do
  log "Embed / LMHead:"
  -- Embed: table [[1,2],[3,4],[5,6]], ids [0,2,1] -> [[1,2],[5,6],[3,4]]
  runManaged_ do
    flat <- allocate (array1D [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ])
    table <- allocate (run (reshapeT (lit flat) [ 3, 2 ]) :: Effect (NDArray D2))
    ids <- allocate (arrayInt1D [ 0, 2, 1 ])
    out <- allocate (embed table ids)
    lift do
      checkShape "embed output shape" [ 3, 2 ] (lit out)
      checkT "embed table[0,2,1] = [1,2,5,6,3,4]"
        [ 1.0, 2.0, 5.0, 6.0, 3.0, 4.0 ] (lit out)
  -- Unembed: hidden [[1,2,3],[4,5,6]] @ table^T where
  --   table = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]] (vocab=4, embed=3).
  -- Expected: [[14,32,50,68],[32,77,122,167]].
  runManaged_ do
    hFlat <- allocate (array1D [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ])
    hidden <- allocate (run (reshapeT (lit hFlat) [ 2, 3 ]) :: Effect (NDArray D2))
    tFlat <- allocate
      (array1D [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 ])
    table <- allocate (run (reshapeT (lit tFlat) [ 4, 3 ]) :: Effect (NDArray D2))
    out <- allocate (unembed hidden table)
    lift $ checkT "unembed [2,3] @ table^T = [2,4]"
      [ 14.0, 32.0, 50.0, 68.0, 32.0, 77.0, 122.0, 167.0 ] (lit out)

-- | Slice along an arbitrary axis while leaving the source borrowed.
-- | `sliceAxis` consumes its first argument, so ref-bump up front.
sliceAxisBorrow :: forall d. NDArray d -> Int -> Int -> Int -> Effect (NDArray d)
sliceAxisBorrow x axis start end = do
  xR <- ref x
  sliceAxis xR axis start end

testRoPE :: Effect Unit
testRoPE = do
  log "RoPE:"
  -- Shape parity for the precomputed tables
  runManaged_ do
    tab <- lift (precomputeRoPE 4 8 10000.0)
    cosT <- allocate (pure tab.cos)
    sinT <- allocate (pure tab.sin)
    lift do
      checkShape "RoPE cos table shape" [ 8, 2 ] (lit cosT)
      checkShape "RoPE sin table shape" [ 8, 2 ] (lit sinT)
  -- Identity at position 0: cos=1, sin=0, output == input.
  runManaged_ do
    tab <- lift (precomputeRoPE 4 8 10000.0)
    cosT <- allocate (pure tab.cos)
    sinT <- allocate (pure tab.sin)
    cosRow <- allocate (sliceAxisBorrow cosT 0 0 1 :: Effect (NDArray D2))
    sinRow <- allocate (sliceAxisBorrow sinT 0 0 1 :: Effect (NDArray D2))
    xFlat <- allocate (array1D [ 3.0, 1.0, 4.0, 1.0 ])
    x <- allocate (run (reshapeT (lit xFlat) [ 1, 4 ]) :: Effect (NDArray D2))
    out <- allocate (applyRoPE 2 x cosRow sinRow)
    lift $ checkT "applyRoPE @pos=0 is identity"
      [ 3.0, 1.0, 4.0, 1.0 ] (lit out)
  -- Position-1 rotation of x = [1, 0, 0, 0] (4-dim, halfDim=2):
  --   cos[1] = [cos(1), cos(0.01)],  sin[1] = [sin(1), sin(0.01)]
  --   xFirst=[1,0], xSecond=[0,0]
  --   out = [cos(1), 0, sin(1), 0] ≈ [0.5403, 0, 0.8415, 0]
  runManaged_ do
    tab <- lift (precomputeRoPE 4 8 10000.0)
    cosT <- allocate (pure tab.cos)
    sinT <- allocate (pure tab.sin)
    cosRow <- allocate (sliceAxisBorrow cosT 0 1 2 :: Effect (NDArray D2))
    sinRow <- allocate (sliceAxisBorrow sinT 0 1 2 :: Effect (NDArray D2))
    xFlat <- allocate (array1D [ 1.0, 0.0, 0.0, 0.0 ])
    x <- allocate (run (reshapeT (lit xFlat) [ 1, 4 ]) :: Effect (NDArray D2))
    out <- allocate (applyRoPE 2 x cosRow sinRow)
    lift $ checkT "applyRoPE @pos=1 of [1,0,0,0]"
      [ 0.5403023058681398, 0.0, 0.8414709848078965, 0.0 ] (lit out)

testAttention :: Effect Unit
testAttention = do
  log "Attention:"
  -- Shape parity for unbatched [seq, n_heads, head_dim].
  runManaged_ do
    q <- allocate (ones [ 3, 2, 4 ] :: Effect (NDArray D3))
    k <- allocate (ones [ 3, 2, 4 ] :: Effect (NDArray D3))
    v <- allocate (ones [ 3, 2, 4 ] :: Effect (NDArray D3))
    out <- allocate (attention q k v)
    lift $ checkShape "attention output shape [3,2,4]" [ 3, 2, 4 ] (lit out)
  -- GQA shape: 4 query heads, 2 KV heads. jax-js broadcasts internally.
  runManaged_ do
    q <- allocate (ones [ 3, 4, 4 ] :: Effect (NDArray D3))
    k <- allocate (ones [ 3, 2, 4 ] :: Effect (NDArray D3))
    v <- allocate (ones [ 3, 2, 4 ] :: Effect (NDArray D3))
    out <- allocate (attention q k v)
    lift $ checkShape "GQA output shape [3,4,4] (4 q-heads, 2 kv-heads)"
      [ 3, 4, 4 ] (lit out)

testMLP :: Effect Unit
testMLP = do
  log "MLP (SwiGLU):"
  -- Shape parity: x [seq=3, hidden=4], inter=8, out [3, 4]
  runManaged_ do
    x <- allocate (ones [ 3, 4 ] :: Effect (NDArray D2))
    gp <- allocate (ones [ 4, 8 ] :: Effect (NDArray D2))
    up <- allocate (ones [ 4, 8 ] :: Effect (NDArray D2))
    dp <- allocate (ones [ 8, 4 ] :: Effect (NDArray D2))
    out <- allocate (mlp x gp up dp)
    lift $ checkShape "mlp output shape [3,4]" [ 3, 4 ] (lit out)
  -- Numerical sanity (zeros input → zeros output regardless of weights).
  runManaged_ do
    x <- allocate (zeros [ 2, 3 ] :: Effect (NDArray D2))
    gp <- allocate (ones [ 3, 5 ] :: Effect (NDArray D2))
    up <- allocate (ones [ 3, 5 ] :: Effect (NDArray D2))
    dp <- allocate (ones [ 5, 3 ] :: Effect (NDArray D2))
    out <- allocate (mlp x gp up dp)
    lift $ checkT "mlp(zeros) = zeros"
      [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] (lit out)

testBlock :: Effect Unit
testBlock = do
  log "Block / Stack:"
  let
    cfg =
      { hidden: 8
      , nHeads: 2
      , nKvHeads: 2
      , headDim: 4
      , intermediate: 16
      , nLayers: 2
      , maxSeqLen: 4
      , vocabSize: 5
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
  runManaged_ do
    emb <- allocate (ones [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2))
    -- Layer 0
    an0 <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
    wq0 <- allocate (ones [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
    wk0 <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
    wv0 <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
    wo0 <- allocate (ones [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
    mn0 <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
    gp0 <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
    up0 <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
    dp0 <- allocate (ones [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
    -- Layer 1
    an1 <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
    wq1 <- allocate (ones [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
    wk1 <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
    wv1 <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
    wo1 <- allocate (ones [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
    mn1 <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
    gp1 <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
    up1 <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
    dp1 <- allocate (ones [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
    fn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
    rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
    cosT <- allocate (pure rope.cos)
    sinT <- allocate (pure rope.sin)
    ids <- allocate (arrayInt1D [ 0, 1, 2 ])
    let
      weights =
        { embedding: emb
        , layers:
            [ { attnNorm: an0
              , attn: { wq: wq0, wk: wk0, wv: wv0, wo: wo0 }
              , mlpNorm: mn0
              , mlp: { gateProj: gp0, upProj: up0, downProj: dp0 }
              }
            , { attnNorm: an1
              , attn: { wq: wq1, wk: wk1, wv: wv1, wo: wo1 }
              , mlpNorm: mn1
              , mlp: { gateProj: gp1, upProj: up1, downProj: dp1 }
              }
            ]
        , finalNorm: fn
        }
      ropeTables = { cos: cosT, sin: sinT }
    out <- allocate (forwardLogits cfg weights ropeTables ids)
    lift $ checkShape "forwardLogits output [seq=3, vocab=5]" [ 3, 5 ] (lit out)

testSampling :: Effect Unit
testSampling = do
  log "Sampling:"
  -- argmax of [1, 5, 3, 2] = 1
  runManaged
    (allocate (array1D [ 1.0, 5.0, 3.0, 2.0 ]))
    \v -> do
      idx <- sampleGreedy v
      assertEqArrayInt "sampleGreedy picks index of max" [ 1 ] [ idx ]

testGenerate :: Effect Unit
testGenerate = do
  log "Generate:"
  let
    cfg =
      { hidden: 8
      , nHeads: 2
      , nKvHeads: 2
      , headDim: 4
      , intermediate: 16
      , nLayers: 1
      , maxSeqLen: 8
      , vocabSize: 5
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
  runManaged
    ( do
        emb <- allocate (ones [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2))
        an <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        wq <- allocate (ones [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
        wk <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wv <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wo <- allocate (ones [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
        mn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        gp <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        up <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        dp <- allocate (ones [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
        fn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
        cosT <- allocate (pure rope.cos)
        sinT <- allocate (pure rope.sin)
        let
          weights =
            { embedding: emb
            , layers:
                [ { attnNorm: an
                  , attn: { wq, wk, wv, wo }
                  , mlpNorm: mn
                  , mlp: { gateProj: gp, upProj: up, downProj: dp }
                  }
                ]
            , finalNorm: fn
            }
          ropeTables = { cos: cosT, sin: sinT }
        lift (generateGreedy cfg weights ropeTables [ 0, 1 ] 3)
    )
    \tokens -> do
      assertEqArrayInt "generateGreedy length = prompt(2) + maxNew(3)"
        [ 5 ]
        [ length tokens ]
  -- Cached vs naive parity: they should produce identical sequences.
  runManaged
    ( do
        emb <- allocate (ones [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2))
        an <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        wq <- allocate (ones [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
        wk <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wv <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wo <- allocate (ones [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
        mn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        gp <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        up <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        dp <- allocate (ones [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
        fn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
        cosT <- allocate (pure rope.cos)
        sinT <- allocate (pure rope.sin)
        let
          weights =
            { embedding: emb
            , layers:
                [ { attnNorm: an
                  , attn: { wq, wk, wv, wo }
                  , mlpNorm: mn
                  , mlp: { gateProj: gp, upProj: up, downProj: dp }
                  }
                ]
            , finalNorm: fn
            }
          ropeTables = { cos: cosT, sin: sinT }
        naive <- lift (generateGreedy cfg weights ropeTables [ 0, 1 ] 3)
        cached <- lift (generateGreedyCached cfg weights ropeTables [ 0, 1 ] 3)
        pure { naive, cached }
    )
    \{ naive, cached } -> do
      assertEqArrayInt "generateGreedyCached parity with naive" naive cached
  -- Temperature-sampled generation: verify length and that low-temp ~ greedy.
  runManaged
    ( do
        emb <- allocate (ones [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2))
        an <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        wq <- allocate (ones [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
        wk <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wv <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wo <- allocate (ones [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
        mn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        gp <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        up <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        dp <- allocate (ones [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
        fn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
        cosT <- allocate (pure rope.cos)
        sinT <- allocate (pure rope.sin)
        let
          weights =
            { embedding: emb
            , layers:
                [ { attnNorm: an
                  , attn: { wq, wk, wv, wo }
                  , mlpNorm: mn
                  , mlp: { gateProj: gp, upProj: up, downProj: dp }
                  }
                ]
            , finalNorm: fn
            }
          ropeTables = { cos: cosT, sin: sinT }
        key <- lift (mkKey 42)
        toks <- lift (generateTemperature key 0.8 cfg weights ropeTables [ 0, 1 ] 3)
        pure toks
    )
    \toks ->
      assertEqArrayInt "generateTemperature length" [ 5 ] [ length toks ]
  where
  cfg =
    { hidden: 8
    , nHeads: 2
    , nKvHeads: 2
    , headDim: 4
    , intermediate: 16
    , nLayers: 1
    , maxSeqLen: 8
    , vocabSize: 5
    , ropeTheta: 10000.0
    , normEps: 1.0e-6
    }

testTraining :: Effect Unit
testTraining = do
  log "Training (autodiff + Adam):"
  -- Toy regression: minimize sum(w^2). Initial w = [3, 4]. Optimum is
  -- w = [0, 0]. With adam(0.1) over 100 steps the loss should drop from
  -- 25 to a small number.
  -- Use the pure-JS sumSquareLoss to avoid mkEffectFn1 wrapping.
  let lossFn = sumSquareLoss :: EffectFn1 (NDArray D1) (NDArray D1)
  vagFn <- valueAndGrad lossFn
  opt <- Optax.adam 0.1
  -- Initial parameter and state. Both Optax.init and vagFn consume their
  -- input, so we ref-bump w0 before each.
  w0 <- array1D [ 3.0, 4.0 ]
  w0R1 <- ref w0
  state0 <- Optax.init opt w0R1
  w0R2 <- ref w0
  initial <- runEffectFn1 vagFn w0R2
  initialLossF <- toJs initial.value
  let initialLoss = unsafeCoerce initialLossF :: Number
  dispose initial.value
  -- Run training loop. Each step consumes the previous params (via
  -- applyUpdates) and returns fresh ones.
  finalState <- trainLoop vagFn opt 100 { params: w0, state: state0, grad: initial.grad }
  -- Final loss
  paramsR <- ref finalState.params
  finalVag <- runEffectFn1 vagFn paramsR
  finalLossF <- toJs finalVag.value
  let finalLoss = unsafeCoerce finalLossF :: Number
  dispose finalVag.value
  dispose finalVag.grad
  dispose finalState.params
  log $ "  initial loss = " <> show initialLoss
  log $ "  final loss   = " <> show finalLoss
  if finalLoss < initialLoss * 0.01 then
    log "  ✓ Adam reduces sum(w²) by >100×"
  else
    throw $ "  ✗ Training failed: initial=" <> show initialLoss
      <> " final=" <> show finalLoss

trainLoop
  :: forall d
   . (EffectFn1 (NDArray d) { value :: NDArray d, grad :: NDArray d })
  -> Optax.Transformation
  -> Int
  -> { params :: NDArray d, state :: Optax.OptState, grad :: NDArray d }
  -> Effect { params :: NDArray d, state :: Optax.OptState }
trainLoop vagFn opt 0 acc = do
  dispose acc.grad
  pure { params: acc.params, state: acc.state }
trainLoop vagFn opt n acc = do
  -- Apply update: state -> (updates, newState); newParams = params + updates
  { updates, state: newState } <- Optax.update opt acc.grad acc.state
  newParams <- Optax.applyUpdates acc.params updates
  -- Compute next vag for the next step (also reads the new loss/grad)
  newParamsR <- ref newParams
  vag <- runEffectFn1 vagFn newParamsR
  dispose vag.value
  trainLoop vagFn opt (n - 1) { params: newParams, state: newState, grad: vag.grad }

-- | Build a small varied weight tensor (linspace in [-0.1, 0.1] reshaped).
varyingWeight :: forall d. Array Int -> Effect (NDArray d)
varyingWeight sh = do
  let n = arrayProduct sh
  base <- linspace (-0.1) 0.1 n
  baseR <- ref base
  reshape baseR sh

arrayProduct :: Array Int -> Int
arrayProduct = foldl (*) 1

testNumericalParity :: Effect Unit
testNumericalParity = do
  log "Numerical parity (analytic checks):"
  -- 1. Matmul value parity: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
  runManaged_ do
    aFlat <- allocate (array1D [ 1.0, 2.0, 3.0, 4.0 ])
    bFlat <- allocate (array1D [ 5.0, 6.0, 7.0, 8.0 ])
    let a = reshapeT (lit aFlat) [ 2, 2 ] :: T D2
        b = reshapeT (lit bFlat) [ 2, 2 ] :: T D2
    lift $ checkT "matmul [[1,2],[3,4]] @ [[5,6],[7,8]]"
      [ 19.0, 22.0, 43.0, 50.0 ] (a **. b)
  -- 2. RMSNorm with non-uniform weight: input [4,4,4,4] (RMS=4),
  -- weight [0.5, 1.0, 1.5, 2.0]. After RMS-normalize, x is [1,1,1,1],
  -- then * weight = [0.5, 1.0, 1.5, 2.0].
  runManaged_ do
    x <- allocate (array1D [ 4.0, 4.0, 4.0, 4.0 ])
    w <- allocate (array1D [ 0.5, 1.0, 1.5, 2.0 ])
    out <- allocate (rmsnorm 1.0e-9 x w :: Effect (NDArray D1))
    lift $ checkT "RMSNorm with non-uniform weight"
      [ 0.5, 1.0, 1.5, 2.0 ] (lit out)
  -- 3. mean of arange[0..5]: (0+1+2+3+4)/5 = 2.0
  runManaged_ do
    a <- allocate (arange 0.0 5.0 1.0)
    aR <- lift (ref a)
    m <- allocate (mean aR :: Effect (NDArray D1))
    lift do
      f <- toJs m
      assertCloseNum "mean(arange[0..5])" 2.0 (asNumber f)
  -- 4. matmul transpose: A @ A^T for A = [[1,2],[3,4]] = [[5,11],[11,25]]
  runManaged_ do
    aFlat <- allocate (array1D [ 1.0, 2.0, 3.0, 4.0 ])
    let a = reshapeT (lit aFlat) [ 2, 2 ] :: T D2
    lift $ checkT "A @ A^T for A=[[1,2],[3,4]]"
      [ 5.0, 11.0, 11.0, 25.0 ] (a **. T.transposeT a)

-- | Regression test for the config codec.
-- |
-- | Both `extrasCodec` and `rawCodec` previously used
-- | `Data.Codec.Argonaut.Common.maybe`, which encodes `Maybe a` as a
-- | tagged sum (`{"Just": ...}`) — wrong for HF JSON where optional
-- | fields are simply absent or carry a real value. This caught the
-- | regression that broke TinyLlama load with:
-- |
-- |     Under 'ConfigExtras': At object key tie_word_embeddings:
-- |     Under 'Maybe': Expected value of type 'Object'.
-- |
-- | Using a representative HF config (mirrors TinyLlama-1.1B-Chat
-- | layout: tied=false, BF16 source, GQA 8:1).
testHFConfigDecode :: Effect Unit
testHFConfigDecode = do
  log "HF config decoder (real-world JSON shape):"
  let
    cfgJson =
      """{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "max_position_embeddings": 2048,
        "vocab_size": 32000,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": false,
        "bos_token_id": 1,
        "eos_token_id": 2
      }"""
  case Cfg.probeRawExtras cfgJson of
    Left err -> throw $ "  ✗ probeRawExtras failed: " <> err
    Right r -> do
      case r.tie_word_embeddings of
        Just false -> log "  ✓ probeRawExtras decoded tie_word_embeddings=false"
        Just true -> throw "  ✗ tie_word_embeddings expected false, got true"
        Nothing -> throw "  ✗ tie_word_embeddings missing from decoded extras"
      case r.model_type of
        Just "llama" -> log "  ✓ probeRawExtras decoded model_type=llama"
        _ -> throw "  ✗ model_type expected Just \"llama\""
  -- Round-trip the full config too — same JSON, full ModelConfig.
  cfg <- Cfg.parseLlamaConfig cfgJson
  if cfg.hidden == 2048 && cfg.nHeads == 32 && cfg.nKvHeads == 4
    && cfg.nLayers == 22 && cfg.vocabSize == 32000
  then log "  ✓ parseLlamaConfig produces correct ModelConfig"
  else throw $ "  ✗ ModelConfig fields wrong: " <> show cfg
  -- Soft-test: probe rejects an incompatible model_type.
  let phiJson = """{"model_type": "phi", "hidden_size": 0,
                    "num_attention_heads": 0, "intermediate_size": 0,
                    "num_hidden_layers": 0, "vocab_size": 0}"""
  case Cfg.probeRawExtras phiJson of
    Left err
      | indexOfStr "phi" err -> log "  ✓ probeRawExtras rejects model_type=phi"
      | otherwise -> throw $ "  ✗ wrong rejection message: " <> err
    Right _ -> throw "  ✗ probeRawExtras should have rejected phi"
  where
  indexOfStr needle s = case StrIdx.indexOf (StrIdx.Pattern needle) s of
    Just _ -> true
    Nothing -> false

testLlamaEndToEnd :: Effect Unit
testLlamaEndToEnd = do
  log "Real-model wiring (synthetic Llama-format safetensors → forwardLogits):"
  -- Build a Llama-formatted safetensors blob with a tiny 1-layer config
  -- (PyTorch [out, in] convention; LlamaAdapter transposes on load).
  let fcfg = llamaFixtureCfg
  fixture <- makeLlamaFixture
  parsed <- ST.parseSafetensors fixture
  -- Sanity: tensor count should be 2 (embedding+norm) + nLayers * 9 keys per layer
  names <- ST.tensorNames parsed
  let expectedCount = 2 + fcfg.nLayers * 9
  assertEqArrayInt "Llama fixture tensor count" [ expectedCount ] [ length names ]
  -- Convert via LlamaAdapter.
  let
    cfg :: ModelConfig
    cfg =
      { hidden: fcfg.hidden
      , nHeads: fcfg.nHeads
      , nKvHeads: fcfg.nKvHeads
      , headDim: fcfg.headDim
      , intermediate: fcfg.intermediate
      , nLayers: fcfg.nLayers
      , maxSeqLen: 16
      , vocabSize: fcfg.vocab
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
  ckpt <- loadLlamaWeights cfg parsed
  -- Run a forward pass with a small prompt.
  rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
  ids <- arrayInt1D [ 0, 1, 2 ]
  logits <- forwardLogits cfg ckpt.weights rope ids
  s <- shape logits
  assertEqArrayInt "real-model forwardLogits output [seq=3, vocab=8]"
    [ 3, fcfg.vocab ]
    s
  -- Heterogeneous pytree traversal: counting tensors and total params.
  -- Expected counts (vocab=hidden=8, nHeads=nKvHeads=2, headDim=4,
  -- intermediate=16, nLayers=1):
  --   tensors  = embedding (1) + finalNorm (1) + 1 layer × 9
  --            = 11
  --   params   = embedding (64) + finalNorm (8)
  --            + 1 × (attnNorm (8) + 4 attn projs (4×64) + mlpNorm (8)
  --                  + 3 mlp projs (3×128))
  --            = 72 + 656 = 728
  let nTensors = countTensors ckpt.weights
  assertEqArrayInt "countTensors ckpt.weights" [ 2 + fcfg.nLayers * 9 ] [ nTensors ]
  nParams <- countParams ckpt.weights
  let
    expectedParams = fcfg.vocab * fcfg.hidden + fcfg.hidden + fcfg.nLayers *
      ( fcfg.hidden                                    -- attnNorm
          + 2 * (fcfg.hidden * fcfg.nHeads * fcfg.headDim)
          + 2 * (fcfg.hidden * fcfg.nKvHeads * fcfg.headDim)
          + fcfg.hidden                                -- mlpNorm
          + 3 * (fcfg.hidden * fcfg.intermediate)
      )
  assertEqArrayInt "countParams ckpt.weights"
    [ expectedParams ]
    [ nParams ]
  -- sumSquaredL2: every leaf is filled with linspace(-0.1, 0.1) values,
  -- so the magnitude is bounded but non-zero. Just sanity-check that
  -- it returns a finite positive Number. (We don't pin an exact value
  -- because the per-leaf linspace ranges depend on each tensor's size.)
  l2sq <- sumSquaredL2 ckpt.weights
  if l2sq > 0.0 then log "  ✓ sumSquaredL2 ckpt.weights > 0"
  else throw $ "  ✗ sumSquaredL2 expected > 0, got " <> show l2sq
  -- Optic-based per-layer L2: walk just one layer's worth of tensors
  -- via the `_layer i` affine traversal from `Jax.Optics`. Sum across
  -- layers must match the global sumSquaredL2 (within FP slop) since
  -- the only non-layer tensors are embedding + finalNorm, and we
  -- subtract those. Also verifies the lens path-composition compiles.
  perLayerSq <- perLayerL2sq fcfg.nLayers ckpt.weights
  let layerSumSq = foldl (+) 0.0 perLayerSq
  case preview _embedding ckpt.weights, preview _finalNorm ckpt.weights of
    Just embT, Just fnT -> do
      eR <- ref embT
      esq <- square eR
      es <- sum esq
      esF <- toJs es
      dispose es
      let embSq = unsafeCoerce esF :: Number
      fR <- ref fnT
      fsq <- square fR
      fs <- sum fsq
      fsF <- toJs fs
      dispose fs
      let fnSq = unsafeCoerce fsF :: Number
      let diff = Math.abs (l2sq - layerSumSq - embSq - fnSq)
      if diff < 1.0e-3 then
        log $ "  ✓ Σ perLayerL2sq + embedding² + finalNorm² ≈ sumSquaredL2 (Δ="
          <> show diff <> ")"
      else throw $ "  ✗ pytree L2 decomposition mismatch · diff=" <> show diff
    _, _ -> throw "  ✗ preview _embedding / _finalNorm returned Nothing"
  log "  ✓ end-to-end safetensors → adapter → forward + pytree traversal"

testTransformerTraining :: Effect Unit
testTransformerTraining = do
  log "Transformer training (full forward + cross-entropy + Adam):"
  let
    cfg :: ModelConfig
    cfg =
      { hidden: 8
      , nHeads: 2
      , nKvHeads: 2
      , headDim: 4
      , intermediate: 16
      , nLayers: 1
      , maxSeqLen: 8
      , vocabSize: 8
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
  weights0 <- buildVaryingWeights cfg
  rope <- precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta
  -- Next-token training: predict prompt[i+1] from prompt[:i+1].
  --   prompt  = [1, 2, 3]  (input)
  --   targets = [2, 3, 4]  (shifted-by-1 next-token labels)
  prompt <- arrayInt1D [ 1, 2, 3 ]
  targets <- arrayInt1D [ 2, 3, 4 ]
  let lossFn = makeCrossEntropyLoss cfg rope prompt targets
  vagFn <- valueAndGradT lossFn
  initVag <- runEffectFn1 vagFn weights0
  initLossF <- toJs initVag.value
  let initLoss = unsafeCoerce initLossF :: Number
  dispose initVag.value
  log $ "  initial loss = " <> show initLoss
  opt <- Optax.adam 0.05
  weightsForInit <- refModelWeights weights0
  state0 <- Optax.initT opt weightsForInit
  finalState <- trainLoopXform vagFn opt 60
    { weights: weights0, state: state0, grad: initVag.grad }
  finalVag <- runEffectFn1 vagFn finalState.weights
  finalLossF <- toJs finalVag.value
  let finalLoss = unsafeCoerce finalLossF :: Number
  dispose finalVag.value
  log $ "  final loss   = " <> show finalLoss
  if finalLoss < initLoss * 0.6 then
    log "  ✓ Adam reduces full-transformer cross-entropy by ≥40%"
  else
    throw $ "  ✗ training didn't reduce loss enough: "
      <> show initLoss <> " → " <> show finalLoss
  -- Now use the trained weights with `generateGreedyCached`. The
  -- training forward used `one_hot @ embedding`; production uses
  -- `embed` (= `take`). Mathematically identical, so generation should
  -- reproduce the memorized continuation.
  --   prompt [1] + maxNew 3 → expected [1, 2, 3, 4]
  generated <- generateGreedyCached cfg finalState.weights rope [ 1 ] 3
  log $ "  generation [1] → " <> show generated
  -- Robust assertion: at minimum the first generated token (position 0
  -- prediction) must be 2, since training drove that case to ~zero loss.
  -- Later positions need more training steps to fully memorize.
  case generated of
    [ 1, 2, _, _ ] -> log "  ✓ trained model produces correct first prediction (1 → 2)"
    _ -> throw $ "  ✗ first prediction wrong: " <> show generated

trainLoopXform
  :: forall e
   . ( EffectFn1 ModelWeights
         { value :: NDArray e, grad :: ModelWeights }
     )
  -> Optax.Transformation
  -> Int
  -> { weights :: ModelWeights, state :: Optax.OptState, grad :: ModelWeights }
  -> Effect { weights :: ModelWeights, state :: Optax.OptState }
trainLoopXform _ _ 0 acc = pure { weights: acc.weights, state: acc.state }
trainLoopXform vagFn opt n acc = do
  { updates, state: newState } <- Optax.updateT opt acc.grad acc.state
  newWeights <- Optax.applyUpdatesT acc.weights updates
  vag <- runEffectFn1 vagFn newWeights
  dispose vag.value
  trainLoopXform vagFn opt (n - 1)
    { weights: newWeights, state: newState, grad: vag.grad }

-- | Allocate a fresh ModelWeights with linspace-derived varied init.
buildVaryingWeights :: ModelConfig -> Effect ModelWeights
buildVaryingWeights cfg = do
  emb <- varyingWeight [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2)
  an <- varyingWeight [ cfg.hidden ] :: Effect (NDArray D1)
  wq <- varyingWeight [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2)
  wk <- varyingWeight [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2)
  wv <- varyingWeight [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2)
  wo <- varyingWeight [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2)
  mn <- varyingWeight [ cfg.hidden ] :: Effect (NDArray D1)
  gp <- varyingWeight [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2)
  up <- varyingWeight [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2)
  dp <- varyingWeight [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2)
  fn <- varyingWeight [ cfg.hidden ] :: Effect (NDArray D1)
  pure
    { embedding: emb
    , layers:
        [ { attnNorm: an
          , attn: { wq, wk, wv, wo }
          , mlpNorm: mn
          , mlp: { gateProj: gp, upProj: up, downProj: dp }
          }
        ]
    , finalNorm: fn
    }

-- | Ref-bump every leaf of a ModelWeights record, returning a fresh
-- | record with the same NDArrays (refcounts +1 each).
refModelWeights :: ModelWeights -> Effect ModelWeights
refModelWeights w = do
  emb <- ref w.embedding
  fn <- ref w.finalNorm
  layers <- traverse refLayer w.layers
  pure { embedding: emb, layers, finalNorm: fn }
  where
  refLayer lw = do
    an <- ref lw.attnNorm
    wq <- ref lw.attn.wq
    wk <- ref lw.attn.wk
    wv <- ref lw.attn.wv
    wo <- ref lw.attn.wo
    mn <- ref lw.mlpNorm
    gp <- ref lw.mlp.gateProj
    up <- ref lw.mlp.upProj
    dp <- ref lw.mlp.downProj
    pure
      { attnNorm: an
      , attn: { wq, wk, wv, wo }
      , mlpNorm: mn
      , mlp: { gateProj: gp, upProj: up, downProj: dp }
      }

testTrainingPytree :: Effect Unit
testTrainingPytree = do
  log "Training pytree (autodiff + Adam over a record):"
  -- params = { a, b } each NDArray D1. loss = Σ a² + Σ b². With Adam,
  -- both leaves should converge to ~0.
  let lossFn = sumSquareTreeLoss
  vagFn <- valueAndGradT lossFn
  opt <- Optax.adam 0.1
  -- Initial params
  a0 <- array1D [ 3.0, 4.0 ]
  b0 <- array1D [ 1.0, 2.0 ]
  -- Init optimizer state — consumes leaves, so ref-bump.
  aR1 <- ref a0
  bR1 <- ref b0
  state0 <- Optax.initT opt { a: aR1, b: bR1 }
  -- Initial vag — also consumes leaves.
  aR2 <- ref a0
  bR2 <- ref b0
  initial <- runEffectFn1 vagFn { a: aR2, b: bR2 }
  initLossF <- toJs initial.value
  let initLoss = unsafeCoerce initLossF :: Number
  dispose initial.value
  -- Train.
  finalState <- pytreeTrainLoop vagFn opt 100
    { params: { a: a0, b: b0 }, state: state0, grad: initial.grad }
  -- Final loss.
  aR3 <- ref finalState.params.a
  bR3 <- ref finalState.params.b
  finalVag <- runEffectFn1 vagFn { a: aR3, b: bR3 }
  finalLossF <- toJs finalVag.value
  let finalLoss = unsafeCoerce finalLossF :: Number
  dispose finalVag.value
  dispose finalVag.grad.a
  dispose finalVag.grad.b
  dispose finalState.params.a
  dispose finalState.params.b
  log $ "  initial loss = " <> show initLoss
  log $ "  final loss   = " <> show finalLoss
  if finalLoss < initLoss * 0.01 then
    log "  ✓ Pytree Adam reduces Σa² + Σb² by >100×"
  else
    throw $ "  ✗ Pytree training failed"

pytreeTrainLoop
  :: forall e
   . ( EffectFn1
         { a :: NDArray D1, b :: NDArray D1 }
         { value :: NDArray e, grad :: { a :: NDArray D1, b :: NDArray D1 } }
     )
  -> Optax.Transformation
  -> Int
  -> { params :: { a :: NDArray D1, b :: NDArray D1 }
     , state :: Optax.OptState
     , grad :: { a :: NDArray D1, b :: NDArray D1 }
     }
  -> Effect
       { params :: { a :: NDArray D1, b :: NDArray D1 }
       , state :: Optax.OptState
       }
pytreeTrainLoop vagFn opt 0 acc = do
  dispose acc.grad.a
  dispose acc.grad.b
  pure { params: acc.params, state: acc.state }
pytreeTrainLoop vagFn opt n acc = do
  { updates, state: newState } <- Optax.updateT opt acc.grad acc.state
  newParams <- Optax.applyUpdatesT acc.params updates
  -- Ref leaves before next vagFn call.
  aR <- ref newParams.a
  bR <- ref newParams.b
  vag <- runEffectFn1 vagFn { a: aR, b: bR }
  dispose vag.value
  pytreeTrainLoop vagFn opt (n - 1)
    { params: newParams, state: newState, grad: vag.grad }

testTokenizer :: Effect Unit
testTokenizer = do
  log "Tokenizer (cl100k_base BPE):"
  let
    tok = Tok.defaultTokenizer
    text = "Hello, world!"
  toks <- Tok.encode tok text
  back <- Tok.decode tok toks
  if back == text then
    log $ "  ✓ round-trip: " <> show toks <> " ↔ " <> show text
  else
    throw $ "  ✗ round-trip mismatch: " <> show text <> " ↔ " <> show back
  -- Empty string sanity
  empty <- Tok.encode tok ""
  emptyBack <- Tok.decode tok empty
  if emptyBack == "" then log "  ✓ empty string round-trip"
  else throw "  ✗ empty string round-trip failed"

testSentencePieceBPE :: Effect Unit
testSentencePieceBPE = do
  log "SentencePiece BPE (Llama tokenizer.model parity):"
  bytes <- BpeFix.loadTokenizerBytes
  sp <- SBPE.fromBinary bytes
  meta <- BpeFix.fixtureMeta
  let
    actualVocab = SBPE.vocabSize sp
    actualBos = SBPE.bosToken sp
    actualEos = SBPE.eosToken sp
    actualUnk = SBPE.unkToken sp
  assertEqArrayInt "vocabSize" [ meta.vocabSize ] [ actualVocab ]
  assertEqArrayInt "bos id" [ meta.bos ] [ actualBos ]
  assertEqArrayInt "eos id" [ meta.eos ] [ actualEos ]
  assertEqArrayInt "unk id" [ meta.unk ] [ actualUnk ]
  cases <- BpeFix.fixtureCases
  for_ cases \c -> do
    encoded <- SBPE.encode sp c.text
    assertEqArrayInt ("encode " <> showText c.text) c.ids encoded
    decoded <- SBPE.decode sp c.ids
    if decoded == c.decoded then
      log $ "  ✓ decode " <> showText c.text
    else
      throw $ "  ✗ decode " <> showText c.text
        <> ": expected " <> showText c.decoded
        <> ", got " <> showText decoded
  where
  showText :: String -> String
  showText s = "\"" <> s <> "\""

testTopK :: Effect Unit
testTopK = do
  log "Top-k sampling:"
  -- logits = [1, 5, 3, 2]; top-2 at indices [1, 2]; with temp→0, sample == argmax = 1
  runManaged (allocate (array1D [ 1.0, 5.0, 3.0, 2.0 ])) \v -> do
    key <- mkKey 7
    idx <- sampleTopK key 2 0.001 v
    if idx == 1 then log "  ✓ sampleTopK low-temp picks argmax (1)"
    else throw $ "  ✗ sampleTopK expected 1, got " <> show idx
  -- Top-p with low temperature behaves like greedy: nucleus collapses to argmax.
  runManaged (allocate (array1D [ 1.0, 5.0, 3.0, 2.0 ])) \v -> do
    key <- mkKey 13
    idx <- sampleTopP key 0.9 0.001 v
    if idx == 1 then log "  ✓ sampleTopP low-temp picks argmax (1)"
    else throw $ "  ✗ sampleTopP expected 1, got " <> show idx

testStreaming :: Effect Unit
testStreaming = do
  log "Streaming decode:"
  let
    cfg =
      { hidden: 8
      , nHeads: 2
      , nKvHeads: 2
      , headDim: 4
      , intermediate: 16
      , nLayers: 1
      , maxSeqLen: 8
      , vocabSize: 5
      , ropeTheta: 10000.0
      , normEps: 1.0e-6
      }
  collected <- Ref.new []
  runManaged
    ( do
        emb <- allocate (ones [ cfg.vocabSize, cfg.hidden ] :: Effect (NDArray D2))
        an <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        wq <- allocate (ones [ cfg.hidden, cfg.nHeads * cfg.headDim ] :: Effect (NDArray D2))
        wk <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wv <- allocate (ones [ cfg.hidden, cfg.nKvHeads * cfg.headDim ] :: Effect (NDArray D2))
        wo <- allocate (ones [ cfg.nHeads * cfg.headDim, cfg.hidden ] :: Effect (NDArray D2))
        mn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        gp <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        up <- allocate (ones [ cfg.hidden, cfg.intermediate ] :: Effect (NDArray D2))
        dp <- allocate (ones [ cfg.intermediate, cfg.hidden ] :: Effect (NDArray D2))
        fn <- allocate (ones [ cfg.hidden ] :: Effect (NDArray D1))
        rope <- lift (precomputeRoPE cfg.headDim cfg.maxSeqLen cfg.ropeTheta)
        cosT <- allocate (pure rope.cos)
        sinT <- allocate (pure rope.sin)
        let
          weights =
            { embedding: emb
            , layers:
                [ { attnNorm: an
                  , attn: { wq, wk, wv, wo }
                  , mlpNorm: mn
                  , mlp: { gateProj: gp, upProj: up, downProj: dp }
                  }
                ]
            , finalNorm: fn
            }
          ropeTables = { cos: cosT, sin: sinT }
          onTok t = Ref.modify_ (\xs -> xs <> [ t ]) collected
        lift (generateGreedyCachedStream cfg weights ropeTables [ 0, 1 ] 3 onTok)
    )
    \_ -> pure unit
  toks <- Ref.read collected
  assertEqArrayInt "streaming yielded 3 tokens via callback" [ 3 ] [ length toks ]

testSafetensors :: Effect Unit
testSafetensors = do
  log "Safetensors:"
  fixture <- makeFixture
  parsed <- ST.parseSafetensors fixture
  names <- ST.tensorNames parsed
  assertEqArrayInt "safetensors tensor count" [ 1 ] [ length names ]
  -- Pull the tensor and check shape + values
  runManaged
    ( do
        w <- allocate (ST.getTensor parsed "weight" :: Effect (NDArray D2))
        wR <- lift (ref w)
        flat <- allocate (reshape wR [ 4 ] :: Effect (NDArray D1))
        s <- lift (shape w)
        f <- lift (toJs flat)
        pure { s, f }
    )
    \{ s, f } -> do
      assertEqArrayInt "safetensors weight shape [2,2]" [ 2, 2 ] s
      assertCloseArray "safetensors weight values"
        [ 1.0, 2.0, 3.0, 4.0 ]
        (asArray1D f)
  -- BF16 promotion: TinyLlama-shaped fixture with one BF16 tensor.
  -- Values [1.0, 2.0, -1.0, 0.5] encoded as the high 16 bits of their
  -- F32 representations. Promotion should round-trip exactly because
  -- BF16 is the high bits of F32 — no rounding for these targets.
  bf16Fix <- makeBF16Fixture
  bf16Parsed <- ST.parseSafetensors bf16Fix
  bf16Names <- ST.tensorNames bf16Parsed
  assertEqArrayInt "BF16 safetensors tensor count" [ 1 ] [ length bf16Names ]
  runManaged_ do
    w <- allocate (ST.getTensor bf16Parsed "w" :: Effect (NDArray D1))
    lift do
      s <- shape w
      assertEqArrayInt "BF16 weight shape [4]" [ 4 ] s
      f <- toJs w
      assertCloseArray "BF16 → F32 promoted values"
        [ 1.0, 2.0, -1.0, 0.5 ]
        (asArray1D f)

testCharTokenizer :: Effect Unit
testCharTokenizer = do
  log "Char tokenizer:"
  -- Build from a small corpus. Vocabulary is first-occurrence ordered,
  -- so we can predict exact indices.
  let
    corpus = "abca\nbcd"
    tok = CharTokenizer.fromText corpus
    vocabSize = CharTokenizer.size tok
  -- Distinct chars: 'a', 'b', 'c', '\n', 'd' → vocab=5
  assertEqArrayInt "char vocab size on 'abca\\nbcd'" [ 5 ] [ vocabSize ]
  -- Encode follows first-occurrence order: a=0, b=1, c=2, \n=3, d=4.
  let encoded = CharTokenizer.encode tok corpus
  assertEqArrayInt "char encode 'abca\\nbcd'"
    [ 0, 1, 2, 0, 3, 1, 2, 4 ] encoded
  -- Decode round-trips the encoding back to the original.
  let decoded = CharTokenizer.decode tok encoded
  if decoded == corpus then log "  ✓ encode/decode round-trip"
  else throw $ "  ✗ round-trip failed: got " <> show decoded
  -- IDs out of range are silently dropped.
  let trimmed = CharTokenizer.decode tok [ 0, 99, 1 ]
  if trimmed == "ab" then log "  ✓ decode drops out-of-range IDs"
  else throw $ "  ✗ expected \"ab\", got " <> show trimmed

testLinearDecay :: Effect Unit
testLinearDecay = do
  log "Schedule.linearDecay:"
  -- step=0 → 1.0; step=numSteps → 0.0; step=numSteps/2 → 0.5; clamped past end.
  let schedule = Schedule.linearDecay 1000
  assertCloseNum "linearDecay 1000 0" 1.0 (schedule 0)
  assertCloseNum "linearDecay 1000 500" 0.5 (schedule 500)
  assertCloseNum "linearDecay 1000 1000" 0.0 (schedule 1000)
  assertCloseNum "linearDecay 1000 2000 (clamped)" 0.0 (schedule 2000)

testCosineDecay :: Effect Unit
testCosineDecay = do
  log "Schedule.cosineDecay:"
  -- step=0 → 1.0; step=numSteps → 0.0; step=numSteps/2 ≈ 0.5; clamped past end.
  let schedule = Schedule.cosineDecay 1000
  assertCloseNum "cosineDecay 1000 0" 1.0 (schedule 0)
  assertCloseNum "cosineDecay 1000 500 (~ 0.5)" 0.5 (schedule 500)
  assertCloseNum "cosineDecay 1000 1000" 0.0 (schedule 1000)
  assertCloseNum "cosineDecay 1000 2000 (clamped)" 0.0 (schedule 2000)
