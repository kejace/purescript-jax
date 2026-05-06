import { safetensors } from "@jax-js/loaders";
import { numpy as np } from "@jax-js/jax";

const dtypeMap = {
  F16: "float16",
  F32: "float32",
  F64: "float64",
  I8: "int8",
  I16: "int16",
  I32: "int32",
  I64: "int64",
  U8: "uint8",
  U16: "uint16",
  U32: "uint32",
  BOOL: "bool",
};

// BF16 is F32 with the lower 16 mantissa bits truncated. To upcast we
// place each BF16 value into the upper 16 bits of a fresh F32, leaving
// the lower 16 zero. Lossless. (jax-js doesn't expose a bfloat16 dtype,
// so we promote on load.)
function bf16ToFloat32(bf16Bytes) {
  // bf16Bytes is the safetensors `data` view (might be Uint16Array or
  // some other typed view depending on the parser). Normalise to a
  // Uint16Array reading from the underlying buffer at the right offset.
  let u16;
  if (bf16Bytes instanceof Uint16Array) {
    u16 = bf16Bytes;
  } else {
    const buf = bf16Bytes.buffer;
    const off = bf16Bytes.byteOffset;
    const len = bf16Bytes.byteLength / 2;
    u16 = new Uint16Array(buf, off, len);
  }
  const out = new Float32Array(u16.length);
  const outU32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) outU32[i] = u16[i] << 16;
  return out;
}

// Parse safetensors bytes → JS object mapping tensor name → NDArray.
// Data is loaded into NDArrays on the default device using the dtype
// declared in the safetensors header. BF16 is auto-promoted to F32.
export const parseSafetensorsImpl = (bytes) => {
  const parsed = safetensors.parse(bytes);
  const out = {};
  for (const name of Object.keys(parsed.tensors)) {
    const t = parsed.tensors[name];
    if (t.dtype === "BF16") {
      const f32 = bf16ToFloat32(t.data);
      out[name] = np.array(f32, { dtype: "float32", shape: t.shape });
      continue;
    }
    const dt = dtypeMap[t.dtype];
    if (!dt) throw new Error(`Unsupported safetensors dtype: ${t.dtype}`);
    out[name] = np.array(t.data, { dtype: dt, shape: t.shape });
  }
  return out;
};

// Return the list of tensor names in a parsed map.
export const tensorNamesImpl = (parsedMap) => Object.keys(parsedMap);

// Fetch a name from the parsed map. Returns the NDArray (a *borrowed*
// handle owned by the parsed map). Bump with `.ref` if you want a
// separate lifetime.
export const getTensorImpl = (parsedMap, name) => {
  const t = parsedMap[name];
  if (!t) throw new Error(`Tensor not found in safetensors: ${name}`);
  return t;
};
