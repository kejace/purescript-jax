// Async fetch helpers with OPFS-backed caching, so model checkpoints
// don't re-download on every page reload. Storage lives in the
// origin-private filesystem (workers + main thread can both reach it).
//
// Cache key = SHA-256(url) hex. Files are flat in OPFS root; not many,
// so no directory structure needed. Writes are best-effort — if OPFS
// is unavailable or quota-full, we silently fall back to live fetch.

async function urlKey(url) {
  const enc = new TextEncoder().encode(url);
  const hash = await crypto.subtle.digest("SHA-256", enc);
  return [...new Uint8Array(hash)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function getOpfsRoot() {
  if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
    throw new Error("OPFS unavailable");
  }
  return navigator.storage.getDirectory();
}

async function getCached(url) {
  try {
    const root = await getOpfsRoot();
    const key = await urlKey(url);
    const fh = await root.getFileHandle(key);
    const file = await fh.getFile();
    return new Uint8Array(await file.arrayBuffer());
  } catch (_) {
    return null;
  }
}

// Default OPFS quota is ~1-2 GB per origin without `persist()`, and
// large writes can fail mid-stream with a transient OOM-style error
// even when the quota isn't strictly exceeded. Skip caching anything
// large up front so we don't generate an alarming warning that looks
// like the load failed (it didn't — the bytes are already in memory
// and being handed to the parser). Threshold chosen to fit Smol-Llama
// (~400 MB) in cache while skipping TinyLlama (~2.2 GB) and larger.
const CACHE_MAX_BYTES = 1_500_000_000;

async function putCached(url, bytes) {
  if (bytes.byteLength > CACHE_MAX_BYTES) {
    console.log(
      `[fetch-cache] skip ${url} (${(bytes.byteLength / 1e6).toFixed(0)} MB > `
        + `${(CACHE_MAX_BYTES / 1e6).toFixed(0)} MB threshold) — every reload will re-fetch`
    );
    return;
  }
  try {
    const root = await getOpfsRoot();
    const key = await urlKey(url);
    const fh = await root.getFileHandle(key, { create: true });
    const ws = await fh.createWritable();
    await ws.write(bytes);
    await ws.close();
  } catch (e) {
    // Best-effort cache; load already succeeded. Surface as info, not warn.
    console.log(
      `[fetch-cache] write failed for ${url} (${e?.message || e}); `
        + `load still succeeded — this URL will be re-fetched on next reload`
    );
  }
}

// Defensive: skip caching anything whose first byte is `<` (0x3C). That
// catches HTML error pages from misconfigured proxies / redirects /
// 4xx-5xx pages. None of the file types we cache (safetensors binary,
// JSON config, SentencePiece model) legitimately start with `<`.
function looksLikeHtmlError(bytes) {
  return bytes.length >= 1 && bytes[0] === 0x3c;
}

async function fetchOrCached(url) {
  const cached = await getCached(url);
  if (cached) {
    if (looksLikeHtmlError(cached)) {
      // Cache poisoned from an earlier bad fetch — drop it and refetch.
      console.warn(`[fetch-cache] poisoned cache for ${url}; evicting`);
      try {
        const root = await getOpfsRoot();
        const key = await urlKey(url);
        await root.removeEntry(key);
      } catch (_) { /* ignore */ }
    } else {
      console.log(`[fetch-cache] hit  ${url} (${cached.byteLength} B)`);
      return cached;
    }
  }
  console.log(`[fetch-cache] miss ${url} — fetching`);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${url}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  if (looksLikeHtmlError(bytes)) {
    // Don't poison the cache; caller will see the parse error and act.
    console.warn(`[fetch-cache] response for ${url} looks like HTML; not caching`);
  } else {
    await putCached(url, bytes);
  }
  return bytes;
}

// Aff-style: return the Promise directly. PS-side `toAffE` from
// `aff-promise` adapts it into Aff (with cancellation + error
// propagation through the Aff fiber).
export const fetchBytesAffImpl = (url) => fetchOrCached(url);

export const fetchTextAffImpl = (url) =>
  fetchOrCached(url).then((bytes) => new TextDecoder().decode(bytes));

// Legacy callback API (kept until callers migrate).
export const fetchBytesImpl = (url, onOk, onErr) => {
  fetchOrCached(url)
    .then((bytes) => onOk(bytes)())
    .catch((e) => onErr(String(e))());
};

export const fetchTextImpl = (url, onOk, onErr) => {
  fetchOrCached(url)
    .then((bytes) => onOk(new TextDecoder().decode(bytes))())
    .catch((e) => onErr(String(e))());
};
