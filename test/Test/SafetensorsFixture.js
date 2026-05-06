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
