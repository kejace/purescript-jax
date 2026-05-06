import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { Readable } from "stream";
import { pipeline } from "stream/promises";

// jax-js's wasm/WebGPU backends need SharedArrayBuffer, only available
// in cross-origin-isolated contexts (= secure context + COOP/COEP). We
// always send COOP+COEP and proxy HuggingFace through this server so
// model downloads carry the `Cross-Origin-Resource-Policy` header that
// COEP=require-corp insists on.
//
// HTTPS toggle: set VITE_USE_HTTPS=1 for direct browser access via
// `https://localhost:5174` (basic-ssl, self-signed). When fronted by an
// ngrok tunnel, leave it off — ngrok terminates TLS at the public edge,
// and `http://localhost` is itself a secure context, so WebGPU + SAB
// still work both directly and via the tunnel.
//
// Why a custom middleware instead of vite's `server.proxy`: HF's
// download URLs return 307 redirects to `cdn-lfs.huggingface.co`. The
// vite proxy forwards the redirect verbatim, the browser then hits the
// CDN directly (bypassing our header injection), and require-corp
// blocks the CDN response. The middleware below follows redirects
// server-side and streams the body back with the right headers.
const useHttps = process.env.VITE_USE_HTTPS === "1";
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const hfMiddleware = {
  name: "hf-proxy",
  configureServer(server) {
    server.middlewares.use("/hf", async (req, res) => {
      const upstream = "https://huggingface.co" + req.url;
      try {
        const response = await fetch(upstream, { redirect: "follow" });
        res.statusCode = response.status;
        const ct = response.headers.get("content-type");
        if (ct) res.setHeader("content-type", ct);
        const cl = response.headers.get("content-length");
        if (cl) res.setHeader("content-length", cl);
        res.setHeader("cross-origin-resource-policy", "cross-origin");
        res.setHeader("cache-control", "public, max-age=31536000");
        if (response.body) {
          await pipeline(Readable.fromWeb(response.body), res);
        } else {
          res.end();
        }
      } catch (e) {
        res.statusCode = 502;
        res.setHeader("content-type", "text/plain");
        res.end(`hf-proxy error: ${e?.message ?? String(e)}`);
      }
    });
  },
};

export default defineConfig({
  root: ".",
  plugins: useHttps ? [basicSsl(), hfMiddleware] : [hfMiddleware],
  server: {
    port: 5173,
    headers: isolationHeaders,
    allowedHosts: ["localhost", "brunos", "inferentially-diadromous-starr.ngrok-free.dev"],
  },
  preview: {
    port: 5173,
    headers: isolationHeaders,
    allowedHosts: ["localhost", "brunos", "inferentially-diadromous-starr.ngrok-free.dev"],
  },
  build: {
    outDir: "dist",
    target: "esnext",
  },
});
