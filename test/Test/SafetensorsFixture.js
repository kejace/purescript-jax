// Construct a minimal safetensors blob in memory for testing the parser.
// Format spec: 8-byte LE u64 header length, then header JSON, then tensor data.
//
// We embed a single tensor "weight" of shape [2,2] dtype F32 with values
// [[1, 2], [3, 4]].

export const makeFixtureImpl = () => {
  let header = JSON.stringify({
    weight: { dtype: "F32", shape: [ 2, 2 ], data_offsets: [ 0, 16 ] },
    __metadata__: {},
  });
  // Pad to 8-byte alignment so the data section is naturally aligned for
  // F32/I32 reads.
  while ((8 + header.length) % 8 !== 0) header += " ";
  const headerBytes = new TextEncoder().encode(header);
  const headerLen = headerBytes.length;
  const data = new Float32Array([ 1, 2, 3, 4 ]);
  const total = 8 + headerLen + 16;
  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  dv.setBigUint64(0, BigInt(headerLen), /* littleEndian = */ true);
  buf.set(headerBytes, 8);
  buf.set(new Uint8Array(data.buffer), 8 + headerLen);
  return buf;
};

// BF16 fixture: one tensor "w" of shape [4] with values [1.0, 2.0, -1.0, 0.5].
// BF16 encoding = upper 16 bits of the F32 representation:
//   1.0  = 0x3F800000 → BF16 0x3F80
//   2.0  = 0x40000000 → BF16 0x4000
//  -1.0  = 0xBF800000 → BF16 0xBF80
//   0.5  = 0x3F000000 → BF16 0x3F00
// 4 values × 2 bytes = 8 bytes. After our promotion to F32, the values
// must round-trip exactly because BF16 IS the high bits of F32 (no
// rounding for these specific simple values).
export const makeBF16FixtureImpl = () => {
  let header = JSON.stringify({
    w: { dtype: "BF16", shape: [ 4 ], data_offsets: [ 0, 8 ] },
    __metadata__: {},
  });
  while ((8 + header.length) % 8 !== 0) header += " ";
  const headerBytes = new TextEncoder().encode(header);
  const headerLen = headerBytes.length;
  // 4 BF16 values = 4 × 2 bytes = 8 bytes, little-endian (low byte first
  // within each u16). Upper 16 bits of F32 of each target value:
  const u16 = new Uint16Array([ 0x3F80, 0x4000, 0xBF80, 0x3F00 ]);
  const total = 8 + headerLen + 8;
  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  dv.setBigUint64(0, BigInt(headerLen), true);
  buf.set(headerBytes, 8);
  buf.set(new Uint8Array(u16.buffer), 8 + headerLen);
  return buf;
};
