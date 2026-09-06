import { mkdir, writeFile } from "node:fs/promises";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import {
    shipModel,
    kartModel,
    droneModel,
    rockModel,
    zoneModel,
    reactorModel,
    palette,
} from "../web/lab/models.js";
// GLTFExporter uses the browser FileReader API to finish its binary container.
globalThis.FileReader = class {
    async readAsArrayBuffer(blob) {
        this.result = await blob.arrayBuffer();
        this.onloadend?.();
    }
};
const destination = new URL("../web/lab/assets/", import.meta.url);
await mkdir(destination, { recursive: true });
const vertices = Array.from({ length: 7 }, (_, i) => [
    Math.cos((i * Math.PI * 2) / 7),
    Math.sin((i * Math.PI * 2) / 7),
]);
const assets = {
    "kestrel-tug": shipModel(),
    "mite-forager": kartModel(),
    "wisp-drone": droneModel(),
    "veined-ore": rockModel(vertices),
    "recovery-dock": zoneModel(3, palette.green, "base"),
    "flux-gate": zoneModel(3, palette.gold),
    "gravity-reactor": reactorModel(),
};
for (const [name, model] of Object.entries(assets)) {
    const buffer = await new GLTFExporter().parseAsync(model, { binary: true });
    await writeFile(
        new URL(`${name}.glb`, destination),
        new Uint8Array(buffer),
    );
}
console.log(`Exported ${Object.keys(assets).length} original GLB models.`);
