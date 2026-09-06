import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import * as T from "../web/lab/vendor/three.module.js";
import { glbJson, geometryOnly } from "./helpers/asset-glb.mjs";
import { worldCatalog } from "../web/lab/visuals/world-catalog.js";
import { assetManifest } from "../web/lab/visuals/asset-manifest.js";
import { animateWorld, WorldInstances } from "../web/lab/visuals/world.js";
import { disposeGroup } from "../web/lab/visuals/resources.js";

const decoded = {};
test("world instancing culls and compacts entries as the camera moves", () => {
  const parent = new T.Group();
  const batch = new WorldInstances("futuristic", parent);
  batch.add("drop-crystal", [0, 0, 0]);
  const distant = batch.add("drop-crystal", [100, 0, 0]);
  batch.build();
  const camera = new T.OrthographicCamera(-5, 5, 5, -5, 0.1, 30);
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);
  batch.updateLod(camera, 500);
  assert.equal(
    batch.batches
      .filter((b) => b.mesh.visible)
      .reduce((n, b) => n + b.mesh.count, 0),
    1,
  );
  camera.position.x = 100;
  camera.lookAt(100, 0, 0);
  batch.updateLod(camera, 500);
  const matrix = new T.Matrix4();
  const visible = batch.batches.find((b) => b.mesh.visible).mesh;
  visible.getMatrixAt(0, matrix);
  assert.equal(
    matrix.elements[12],
    100,
    "Visible instances are packed into the first slot",
  );
  distant.visible = false;
  batch.update();
  assert(batch.batches.every((b) => b.mesh.count === 0 && !b.mesh.visible));
  disposeGroup(parent);
});
for (const style of ["futuristic", "steampunk"]) {
  decoded[style] = {};
  for (const lod of ["high", "low"]) {
    const bytes = await readFile(
      new URL(`../web/lab/assets/${style}/world-${lod}.glb`, import.meta.url),
    );
    const json = glbJson(bytes);
    assert.equal(
      json.scenes.length,
      1,
      "Export only the authored active scene",
    );
    assert(json.images.every((i) => i.bufferView != null && !i.uri));
    assert(json.buffers.every((b) => !b.uri));
    decoded[style][lod] = await geometryOnly(bytes, json);
  }
}

for (const [kind, spec] of Object.entries(worldCatalog))
  test(`${kind}: paired shared envelopes, detailed geometry and replay`, () => {
    const signatures = [];
    for (const style of ["futuristic", "steampunk"]) {
      for (const lod of ["high", "low"]) {
        const pack = decoded[style][lod];
        let root;
        pack.traverse((p) => {
          if (p.userData.assetModel === kind) root = p;
        });
        assert(root, `${style}/${lod}/${kind}`);
        assert.deepEqual(root.userData.collisionEnvelope, spec.size);
        assert.equal(
          root.userData.physicsSource,
          "scene-defined; identical across styles",
        );
        assert(assetManifest[style][kind][lod].endsWith(`#${kind}`));
        pack.updateMatrixWorld(true);
        const bounds = new T.Box3().setFromObject(root);
        for (let axis = 0; axis < 3; axis++) {
          const key = ["x", "y", "z"][axis];
          const min = axis === 2 ? 0.024 : -spec.size[axis] / 2 - 0.001;
          const max =
            axis === 2 ? spec.size[axis] + 0.026 : spec.size[axis] / 2 + 0.001;
          assert(
            bounds.min[key] >= min - 0.003,
            `${style}/${kind} ${key} lower bound ${bounds.min[key]}`,
          );
          assert(
            bounds.max[key] <= max + 0.003,
            `${style}/${kind} ${key} upper bound ${bounds.max[key]}`,
          );
        }
        let triangles = 0,
          vertices = 0,
          coordinateSum = 0;
        root.traverse((p) => {
          if (p.isMesh) {
            const positions = p.geometry.attributes.position;
            if (
              kind.startsWith("ore-") &&
              p.name.includes("Veined natural slate")
            ) {
              p.geometry.computeBoundingBox();
              const center = p.geometry.boundingBox.getCenter(new T.Vector3());
              const normal = p.geometry.attributes.normal;
              let outward = 0;
              for (let i = 0; i < positions.count; i++) {
                const direction = new T.Vector3()
                  .fromBufferAttribute(positions, i)
                  .sub(center)
                  .normalize();
                outward += direction.dot(
                  new T.Vector3().fromBufferAttribute(normal, i),
                );
              }
              assert(
                outward / positions.count > 0.4,
                `${style}/${kind}: mineral normals face outward`,
              );
              const uv = p.geometry.attributes.uv;
              const index = p.geometry.index;
              for (let i = 0; i < (index?.count || positions.count); i += 3) {
                const u = [0, 1, 2].map((j) =>
                  uv.getX(index ? index.getX(i + j) : i + j),
                );
                assert(
                  Math.max(...u) - Math.min(...u) <= 0.51,
                  `${style}/${kind}: mineral UVs do not stretch across the longitude seam`,
                );
              }
            }
            vertices += positions.count;
            triangles += (p.geometry.index?.count || positions.count) / 3;
            for (let i = 0; i < positions.count; i++)
              coordinateSum +=
                (positions.getX(i) +
                  3 * positions.getY(i) +
                  7 * positions.getZ(i)) *
                ((i % 7) + 1);
          }
        });
        assert(
          triangles > 0 && triangles <= (lod === "low" ? 6000 : 50000),
          `${kind}/${lod}: ${triangles} triangles`,
        );
        if (lod === "high")
          signatures.push([vertices, +coordinateSum.toFixed(4)]);
        const pose = () => {
          pack.updateMatrixWorld(true);
          const result = [];
          root.traverse((p) => result.push([...p.matrixWorld.elements]));
          return result;
        };
        animateWorld(root, 3);
        const first = pose();
        animateWorld(root, 20);
        animateWorld(root, 3);
        assert.deepEqual(pose(), first);
        animateWorld(root, 0);
      }
    }
    assert.notDeepEqual(
      signatures[0],
      signatures[1],
      `${kind}: styles must differ in geometry, not just materials`,
    );
  });
