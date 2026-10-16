-- | Custom WGSL kernels on jax-js arrays, without copies (WebGPU backend).
-- |
-- | A kernel reads its inputs from the arrays' own GPU buffers and writes
-- | fresh jax-js arrays, which later jax-js ops (or kernels) read in place.
-- | Bindings: inputs 0 .. n-1 (`var<storage, read>`), outputs n .. n+m-1
-- | (`var<storage, read_write>`), then, if any, a uniform of i32s.
module Jax.WebGPU
  ( GPUDevice
  , GPUBuffer
  , KernelSpec
  , OutputSpec
  , device
  , gpuBuffer
  , runKernel
  ) where

import Control.Promise (Promise, toAffE)
import Effect (Effect)
import Effect.Aff (Aff)
import Effect.Uncurried (EffectFn1, runEffectFn1)
import Jax.Core (NDArray)

foreign import data GPUDevice :: Type
foreign import data GPUBuffer :: Type

type OutputSpec = { shape :: Array Int, dtype :: String }

type KernelSpec d =
  { code :: String
  , entryPoint :: String
  , workgroups :: Array Int
  , inputs :: Array (NDArray d)
  , outputs :: Array OutputSpec
  , uniforms :: Array Int
  }

foreign import deviceImpl :: Effect GPUDevice
foreign import gpuBufferImpl :: forall d. EffectFn1 (NDArray d) (Promise GPUBuffer)
foreign import runKernelImpl :: forall d e. EffectFn1 (KernelSpec d) (Promise (Array (NDArray e)))

-- | jax-js's WebGPU device (the WebGPU backend must be initialised).
device :: Effect GPUDevice
device = deviceImpl

-- | The array's GPU buffer, after its pending kernels; read-only, owned by
-- | jax-js while the array lives. Non-consuming.
gpuBuffer :: forall d. NDArray d -> Aff GPUBuffer
gpuBuffer a = toAffE (runEffectFn1 gpuBufferImpl a)

-- | Run a WGSL kernel; resolves to its outputs as jax-js arrays.
-- | Non-consuming on the inputs.
runKernel :: forall d e. KernelSpec d -> Aff (Array (NDArray e))
runKernel spec = toAffE (runEffectFn1 runKernelImpl spec)
