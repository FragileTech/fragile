import { mkdir, copyFile, readFile, writeFile } from "node:fs/promises";
const root = new URL("../", import.meta.url),
  vendor = new URL("web/optimization/vendor/", root);
await mkdir(vendor, { recursive: true });
for (const name of ["three.module.js", "three.core.js"])
  await copyFile(
    new URL(`node_modules/three/build/${name}`, root),
    new URL(name, vendor),
  );
await mkdir(new URL("addons/controls/", vendor), { recursive: true });
const source = await readFile(
  new URL("node_modules/three/examples/jsm/controls/OrbitControls.js", root),
  "utf8",
);
await writeFile(
  new URL("addons/controls/OrbitControls.js", vendor),
  source.replaceAll("from 'three'", "from '../../three.module.js'"),
);
await copyFile(
  new URL("node_modules/three/LICENSE", root),
  new URL("LICENSE-three.txt", vendor),
);
console.log("Optimization Lab renderer bundled.");
