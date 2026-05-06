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

  // Pre-flight memory estimate. After BF16 → F32 promotion, the
  // working set is up to 2× the on-disk safetensors size, plus the
  // original buffer must stay alive for the parse loop. wasm32 caps
  // total linear memory at 4 GiB; in practice browsers limit further.
  // Surface a clear error before the parse loop hits an opaque wasm
  // trap. The threshold is conservative — leaves headroom for jax-js
  // intermediate buffers + the fetched safetensors blob.
  let bf16Bytes = 0, f32Bytes = 0;
  for (const name of Object.keys(header)) {
    if (name === "__metadata__") continue;
    const e = header[name];
    const sz = e.data_offsets[1] - e.data_offsets[0];
    if (e.dtype === "BF16") bf16Bytes += sz; else f32Bytes += sz;
  }
  const promoted = f32Bytes + bf16Bytes * 2; // BF16 → F32 doubles
  const sourceMb = (bf16Bytes + f32Bytes) / 1e6;
  const promotedMb = promoted / 1e6;
  console.log(
    `[safetensors] header=${headerLen} bytes · ${Object.keys(header).length} entries · `
      + `source=${sourceMb.toFixed(0)} MB · post-promotion=${promotedMb.toFixed(0)} MB`
  );
  // Diagnostic-only — can't refuse without knowing the backend, which
  // lives in the worker. Surface a heads-up so the user knows whether
  // a failure is likely capacity-bound.
  if (promoted > 3.0e9) {
    console.warn(
      `[safetensors] working set exceeds ~3 GB after promotion. wasm32 backends `
        + `cap at 4 GB total linear memory; if you're not on WebGPU this load `
        + `will likely fail with "Out of bounds access" (= wasm OOM trap).`
    );
  }

  const out = {};
  let n = 0, total = Object.keys(header).filter(k => k !== "__metadata__").length;
  for (const name of Object.keys(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = header[name];
    const byteOffset = dataStart + data_offsets[0];
    const byteLength = data_offsets[1] - data_offsets[0];
    n++;
    try {
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
        default: throw new Error(`Unsupported safetensors dtype branch: ${dtype}`);
      }
      out[name] = np.array(data, { dtype: dt, shape });
    } catch (e) {
      // Re-throw with context: which tensor, how big, and how far in.
      // The user previously saw a bare "Out of bounds access" with no
      // way to tell which tensor or where in the file.
      throw new Error(
        `safetensors parse failed at tensor ${n}/${total} "${name}" `
          + `(dtype=${dtype}, shape=[${shape}], byteLength=${byteLength}, `
          + `offset=${byteOffset}): ${e?.message || e}`
      );
    }
  }
  console.log(`[safetensors] parsed ${total} tensors successfully`);
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
