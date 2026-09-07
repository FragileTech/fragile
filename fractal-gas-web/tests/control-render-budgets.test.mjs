import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { glbJson } from "./helpers/asset-glb.mjs";

// Freeze the actual exports before the concept-fidelity pass. Absolute model
// limits alone would permit a visual refinement to increase rendering costs.
const budgets = JSON.parse(
  await readFile(
    new URL("./fixtures/lab-render-budgets.json", import.meta.url),
  ),
);
const worldBudgets = JSON.parse(
  await readFile(
    new URL("./fixtures/world-render-budgets.json", import.meta.url),
  ),
);
for (const [name, budget] of Object.entries(budgets)) {
  test(`${name}: concept refinement preserves rendering and loading budgets`, async () => {
    const bytes = await readFile(
      new URL(`../web/lab/assets/${name}.glb`, import.meta.url),
    );
    const json = glbJson(bytes);
    const primitives = json.meshes.flatMap((mesh) => mesh.primitives);
    const binaryOffset = 28 + bytes.readUInt32LE(12);
    const texturePixels = json.images.reduce((sum, image) => {
      assert(image.bufferView != null && !image.uri, "embedded image");
      const view = json.bufferViews[image.bufferView];
      const offset = binaryOffset + (view.byteOffset || 0);
      assert.equal(bytes.toString("ascii", offset + 1, offset + 4), "PNG");
      return (
        sum + bytes.readUInt32BE(offset + 16) * bytes.readUInt32BE(offset + 20)
      );
    }, 0);
    const actual = {
      triangles: primitives.reduce(
        (sum, primitive) => sum + json.accessors[primitive.indices].count / 3,
        0,
      ),
      primitives: primitives.length,
      materials: json.materials.length,
      images: json.images.length,
      bytes: bytes.length,
      texturePixels,
    };
    for (const [metric, ceiling] of Object.entries(budget)) {
      assert(
        actual[metric] <= ceiling,
        `${metric}: ${actual[metric]} exceeds preceding export ${ceiling}`,
      );
    }
    // Pack totals cannot hide a regression in a frequently instanced prop.
    if (name.includes("/world-")) {
      const [style, pack] = name.split("/");
      const lod = pack.slice("world-".length);
      const cost = (index) => {
        const node = json.nodes[index];
        const parts =
          node.mesh == null ? [] : json.meshes[node.mesh].primitives;
        const result = {
          triangles: parts.reduce(
            (n, p) => n + json.accessors[p.indices].count / 3,
            0,
          ),
          primitives: parts.length,
        };
        for (const child of node.children || []) {
          const childCost = cost(child);
          result.triangles += childCost.triangles;
          result.primitives += childCost.primitives;
        }
        return result;
      };
      json.nodes.forEach((node, index) => {
        if (!node.extras?.assetModel) return;
        const key = `${style}/${node.extras.assetModel}-${lod}`;
        const ceiling = worldBudgets[key];
        assert(ceiling, `Recorded world budget for ${key}`);
        for (const [metric, count] of Object.entries(cost(index))) {
          assert(
            count <= ceiling[metric],
            `${key} ${metric}: ${count} > ${ceiling[metric]}`,
          );
        }
      });
    }
  });
}
