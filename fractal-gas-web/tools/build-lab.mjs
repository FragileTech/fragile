import { mkdir, copyFile } from "node:fs/promises";
import { syncAgentCatalog } from "./sync-agent-catalog.mjs";
const root = new URL("../", import.meta.url);
await syncAgentCatalog();
// Keep the laboratory identity synchronized with the documentation assets.
await mkdir(new URL("web/lab/branding/", root), { recursive: true });
for (const name of ["logo.png", "favicon.png"])
  await copyFile(
    new URL(`../docs/${name}`, root),
    new URL(`web/lab/branding/${name}`, root),
  );
await mkdir(new URL("web/lab/vendor/", root), { recursive: true });
for (const file of ["three.module.js", "three.core.js"]) {
  await copyFile(
    new URL(`node_modules/three/build/${file}`, root),
    new URL(`web/lab/vendor/${file}`, root),
  );
}
await copyFile(
  new URL("node_modules/three/LICENSE", root),
  new URL("web/lab/vendor/LICENSE-three.txt", root),
);
await import("./build-control-assets.mjs");
console.log(
  "Control lab renderer bundled. Build the C++ WebAssembly targets before serving web/.",
);
