// Custom WGSL kernels on jax-js arrays, without copies.
//
// Inputs: each array's own GPU buffer (`Array.gpuBuffer`, read-only).
// Outputs: fresh jax-js arrays (np.zeros of the requested shape and dtype),
// realised, whose buffers the kernel writes; the caller gets jax-js arrays
// back that later jax-js ops (or kernels) read in place. This relies on a
// freshly realised array owning its buffer exclusively, which holds for
// np.zeros in jax-js 0.1.x; the contract of `gpuBuffer` says not to mutate
// buffers, so this is the one place that does, on arrays nobody else holds.
import { numpy as np, getWebGPUDevice } from "@jax-js/jax";

export const deviceImpl = () => getWebGPUDevice();
// Non-consuming: `gpuBuffer` disposes its receiver.
export const gpuBufferImpl = (a) => a.ref.gpuBuffer();

const pipelines = new Map();
const pipelineFor = (device, code, entryPoint) => {
  const key = entryPoint + "\u0000" + code;
  if (!pipelines.has(key)) {
    const module = device.createShaderModule({ code });
    pipelines.set(key, device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } }));
  }
  return pipelines.get(key);
};

// spec: { code, entryPoint, workgroups: [x, y, z], inputs: [Array],
//         outputs: [{ shape, dtype }], uniforms: [Int] }
// Bindings: inputs 0 .. n-1 (read), outputs n .. n+m-1 (read_write), then a
// uniform buffer of i32s if `uniforms` is not empty. Resolves to the outputs.
export const runKernelImpl = async (spec) => {
  const device = getWebGPUDevice();
  const pipeline = pipelineFor(device, spec.code, spec.entryPoint);
  const inBufs = await Promise.all(spec.inputs.map((a) => a.ref.gpuBuffer()));
  const outs = spec.outputs.map((o) => np.zeros(o.shape, { dtype: o.dtype }));
  const outBufs = await Promise.all(outs.map((a) => a.ref.gpuBuffer()));
  const entries = [...inBufs, ...outBufs].map((buffer, binding) => ({ binding, resource: { buffer } }));
  let uniform = null;
  if (spec.uniforms.length) {
    const data = new Int32Array(Math.ceil(spec.uniforms.length / 4) * 4);
    data.set(spec.uniforms);
    uniform = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniform, 0, data);
    entries.push({ binding: entries.length, resource: { buffer: uniform } });
  }
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(...spec.workgroups);
  pass.end();
  device.queue.submit([enc.finish()]);
  if (uniform) device.queue.onSubmittedWorkDone?.().then(() => uniform.destroy());
  return outs;
};
