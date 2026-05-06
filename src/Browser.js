// DOM + Worker FFI for the browser controller. Kept inline to avoid
// pulling in the full purescript-web-* dependency stack.

export const getElByIdImpl = (id) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`No element with id="${id}"`);
  return el;
};

export const setTextImpl = (el, text) => { el.textContent = text; };
export const appendTextImpl = (el, text) => { el.textContent += text; };

export const onClickImpl = (el, action) => {
  el.addEventListener("click", () => action());
};

export const getValueImpl = (el) => el.value;
export const setHtmlImpl = (el, html) => { el.innerHTML = html; };
export const setStyleDisplayImpl = (el, value) => { el.style.display = value; };
export const getInnerHtmlImpl = (el) => () => el.innerHTML;
export const setValueImpl = (el, v) => { el.value = v; };
export const onChangeImpl = (el, action) => {
  el.addEventListener("change", () => action());
};

// hasUrlParamImpl: returns true if the named query param appears in
// window.location.search, regardless of value (`?debug` and `?debug=1`
// both true). Used to gate dev-only diagnostics.
export const hasUrlParamImpl = (name) => () => {
  if (typeof window === "undefined") return false;
  const p = new URLSearchParams(window.location.search);
  return p.has(name);
};

// Clear the origin-private filesystem (the fetch cache). Returns count
// of files removed via the callback, or an error message via onErr.
export const clearOpfsImpl = (onOk, onErr) => {
  (async () => {
    try {
      if (!navigator?.storage?.getDirectory) throw new Error("OPFS unavailable");
      const root = await navigator.storage.getDirectory();
      let n = 0;
      for await (const name of root.keys()) {
        await root.removeEntry(name);
        n++;
      }
      onOk(n)();
    } catch (e) {
      onErr(String(e))();
    }
  })();
};

// Worker spawning + message passing.
export const mkWorkerImpl = (url) => new Worker(url, { type: "module" });
export const postMessageImpl = (w, msg) => { w.postMessage(msg); };
export const onMessageImpl = (w, cb) => { w.onmessage = (e) => cb(e.data)(); };

export const toNumberImpl = (n) => n;

export const toFixed1Impl = (x) => Number(x).toFixed(1);
export const toFixed4Impl = (x) => Number(x).toFixed(4);
