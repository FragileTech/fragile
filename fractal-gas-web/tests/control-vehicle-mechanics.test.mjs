import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { glbJson } from "./helpers/asset-glb.mjs";

// Fixed source-space housing positions shared by high and crowd LODs. Keeping
// these under the normalized model root lets the runtime cache their world
// transforms without a mesh traversal or extra per-vehicle draw batches.
function expectedSockets(style, model) {
  if (model === "kart") return {
    brake: [[-1.505, -0.29, 0.41], [-1.505, 0.29, 0.41]],
    reverse: [[-1.532, 0, 0.30]],
    drive: [style === "steampunk" ? [-1.475, 0, 0.76] : [-1.44, 0, 0.54]],
  };
  const steam = style === "steampunk";
  return {
    brake: [[steam ? -1.894 : -2.185, -0.6, steam ? 1.13 : 0.86], [steam ? -1.894 : -2.185, 0.6, steam ? 1.13 : 0.86]],
    reverse: [steam ? [-1.89, 0, 1.3] : [-2.17, 0, 0.94]],
    drive: [steam ? [-1.65, 0.15, 2.12] : [-2.17, 0, 1.42]],
  };
}

for (const style of ["futuristic", "steampunk"]) {
  for (const model of ["kart", "harvester"]) {
    for (const lod of ["high", "low"]) {
      test(`${style}/${model}-${lod}: illumination sockets stay on authored housings`, async () => {
        const bytes = await readFile(new URL(`../web/lab/assets/${style}/${model}-${lod}.glb`, import.meta.url));
        const json = glbJson(bytes);
        const rootIndex = json.nodes.findIndex((node) => node.extras?.assetModel === model);
        assert(rootIndex >= 0);
        const sockets = json.nodes.map((node, index) => ({ node, index }))
          .filter(({ node }) => node.extras?.effectSocket);
        assert.equal(sockets.length, 4);
        for (const [effect, points] of Object.entries(expectedSockets(style, model))) {
          const matches = sockets.filter(({ node }) => node.extras.effectSocket === effect)
            .sort((a, b) => a.node.translation[1] - b.node.translation[1]);
          assert.equal(matches.length, points.length);
          matches.forEach(({ node, index }, socketIndex) => {
            assert.equal(node.mesh, undefined, "sockets add no mesh draw");
            assert.equal(node.extras.motion, undefined, "housing sockets remain static");
            assert.equal(node.extras.outwardAxis, "-X");
            assert(json.nodes[rootIndex].children.includes(index));
            node.translation.forEach((value, axis) => {
              assert(Number.isFinite(value));
              assert(Math.abs(value - points[socketIndex][axis]) < 1e-5);
            });
          });
        }
      });
    }
  }
}
