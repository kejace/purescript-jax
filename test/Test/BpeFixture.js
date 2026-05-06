import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// Resolve relative to this source file so the lookup is independent of
// the CWD spago test happens to run from. After the spago build the
// emitted module sits under `output/Test.BpeFixture/`, so we walk up
// to the project root before joining `test/fixtures/...`.
const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, "..", "..");
const tokenizerPath = resolve(
  projectRoot,
  "test/fixtures/smol-llama-tokenizer.model",
);
const fixturePath = resolve(
  projectRoot,
  "test/fixtures/smol-llama-bpe-fixture.json",
);

export const loadTokenizerBytesImpl = () => {
  const buf = readFileSync(tokenizerPath);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
};

let _cached = null;
const fixture = () => {
  if (_cached) return _cached;
  _cached = JSON.parse(readFileSync(fixturePath, "utf8"));
  return _cached;
};

export const fixtureMetaImpl = () => {
  const f = fixture();
  return {
    vocabSize: f.vocabSize,
    bos: f.bos,
    eos: f.eos,
    unk: f.unk,
  };
};

// Each case becomes { text, ids, decoded }.
export const fixtureCasesImpl = () =>
  fixture().cases.map((c) => ({
    text: c.text,
    ids: c.ids,
    decoded: c.decoded,
  }));
