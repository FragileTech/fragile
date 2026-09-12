import { mkdir, copyFile, readFile, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const app = new URL("../", import.meta.url);
const workspace = new URL("../algorithmic-gas/", app);
const output = new URL("web/euclidean-gas/", app);
const bindgen = process.env.WASM_BINDGEN || "wasm-bindgen";
function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: fileURLToPath(workspace),
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw new Error(`${command} exited with ${result.status}`);
}
const version = spawnSync(bindgen, ["--version"], { encoding: "utf8" });
if (!version.stdout?.includes("0.2.114")) {
  throw new Error(
    "Install the pinned tool: cargo install wasm-bindgen-cli --version 0.2.114 --locked",
  );
}
const profiles = process.argv.includes("--cpu-only")
  ? ["cpu"]
  : ["cpu", "webgpu"];
for (const profile of profiles) {
  run("cargo", [
    "build",
    "--locked",
    "--release",
    "--target",
    "wasm32-unknown-unknown",
    "-p",
    "algorithmic-gas-wasm",
    ...(profile === "webgpu" ? ["--features", "webgpu"] : []),
  ]);
  const directory = new URL(`engine/${profile}/`, output);
  await mkdir(directory, { recursive: true });
  run(bindgen, [
    fileURLToPath(
      new URL(
        "target/wasm32-unknown-unknown/release/algorithmic_gas_wasm.wasm",
        workspace,
      ),
    ),
    "--target",
    "web",
    "--out-dir",
    fileURLToPath(directory),
    "--out-name",
    "gas",
  ]);
}
const vendor = new URL("vendor/", output);
await mkdir(vendor, { recursive: true });
for (const name of ["three.module.js", "three.core.js"]) {
  await copyFile(
    new URL(`node_modules/three/build/${name}`, app),
    new URL(name, vendor),
  );
}
await mkdir(new URL("addons/controls/", vendor), { recursive: true });
const controls = await readFile(
  new URL("node_modules/three/examples/jsm/controls/OrbitControls.js", app),
  "utf8",
);
await writeFile(
  new URL("addons/controls/OrbitControls.js", vendor),
  controls.replaceAll("from 'three'", "from '../../three.module.js'"),
);
await copyFile(
  new URL("node_modules/three/LICENSE", app),
  new URL("LICENSE-three.txt", vendor),
);
await writeFile(
  new URL("engine/build.json", output),
  JSON.stringify(
    { version: 1, burn: "0.21.0", rust: "1.95.0", profiles },
    null,
    2,
  ),
);
console.log(`Euclidean Gas Lab built: ${profiles.join(", ")}.`);
