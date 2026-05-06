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

// BF16 → F32 promotion: place each BF16 value into the upper 16 bits of
// a fresh F32, leaving the lower 16 zero. Lossless (BF16 IS the high 16
// bits of an F32). jax-js doesn't expose a bfloat16 dtype, so we have
// to promote on load — the working set roughly doubles.
function bf16ToFloat32(buffer, byteOffset, byteLength) {
  const u16 = new Uint16Array(buffer, byteOffset, byteLength / 2);
  const out = new Float32Array(u16.length);
  const outU32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) outU32[i] = u16[i] << 16;
  return out;
}

// Parse a safetensors blob ourselves rather than via @jax-js/loaders.
// Why: upstream's `safetensors.parse` throws "Unsupported dtype: BF16"
// before we ever see the parsed tensors — so its switch is the gating
// failure, not our promotion code that runs after. The format is
// simple enough that re-implementing the header parse is one page.
//
// Layout: <8 LE bytes: header size N> <N bytes: JSON header> <data...>
// Header JSON: { "tensor_name": { dtype, shape, data_offsets:[lo,hi] }, ... }
// All offsets are relative to the *start of the data section* (after
// the 8-byte size + the N-byte header), not the start of the file.
export const parseSafetensorsImpl = (bytes) => {
  // bytes is a Uint8Array view; resolve the underlying ArrayBuffer + base offset.
  const buffer = bytes.buffer;
  const base = bytes.byteOffset || 0;
  const view = new DataView(buffer, base, bytes.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));
  if (headerLen <= 0 || headerLen > bytes.byteLength - 8) {
    throw new Error(`Invalid safetensors header size: ${headerLen}`);
  }
  let header;
  try {
    header = JSON.parse(new TextDecoder().decode(
      new Uint8Array(buffer, base + 8, headerLen)
    ));
  } catch (e) {
    throw new Error(`Failed to parse safetensors header JSON: ${e}`);
  }
  const dataStart = base + 8 + headerLen;
  const out = {};
  for (const name of Object.keys(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = header[name];
    const byteOffset = dataStart + data_offsets[0];
    const byteLength = data_offsets[1] - data_offsets[0];
    if (dtype === "BF16") {
      const f32 = bf16ToFloat32(buffer, byteOffset, byteLength);
      out[name] = np.array(f32, { dtype: "float32", shape });
      continue;
    }
    const dt = dtypeMap[dtype];
    if (!dt) throw new Error(`Unsupported safetensors dtype: ${dtype}`);
    let data;
    switch (dtype) {
      case "F16":  data = new Float16Array(buffer, byteOffset, byteLength / 2); break;
      case "F32":  data = new Float32Array(buffer, byteOffset, byteLength / 4); break;
      case "F64":  data = new Float64Array(buffer, byteOffset, byteLength / 8); break;
      case "I8":   data = new Int8Array(buffer, byteOffset, byteLength);        break;
      case "I16":  data = new Int16Array(buffer, byteOffset, byteLength / 2);   break;
      case "I32":  data = new Int32Array(buffer, byteOffset, byteLength / 4);   break;
      case "I64":  data = new BigInt64Array(buffer, byteOffset, byteLength / 8);break;
      case "U8":
      case "BOOL": data = new Uint8Array(buffer, byteOffset, byteLength);       break;
      case "U16":  data = new Uint16Array(buffer, byteOffset, byteLength / 2);  break;
      case "U32":  data = new Uint32Array(buffer, byteOffset, byteLength / 4);  break;
      // U64/BigUint64Array would also fit here but jax-js's np.array
      // doesn't accept BigUint64Array; add when needed.
      default: throw new Error(`Unsupported safetensors dtype branch: ${dtype}`);
    }
    out[name] = np.array(data, { dtype: dt, shape });
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
