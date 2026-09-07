import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { glbJson, geometryOnly } from "./helpers/asset-glb.mjs";
import * as T from "../web/lab/vendor/three.module.js";
import { animatedParts, animateAgent } from "../web/lab/visuals/registry.js";
import { chooseLod } from "../web/lab/visuals/body-layer.js";
import { retainAsset, disposeGroup } from "../web/lab/visuals/resources.js";
import { StyleService, styleStorageKey } from "../web/lab/visual-style.js";

// Captured before the concept-surface refinement. Fidelity improvements must
// fit the existing GPU workload, not merely the looser absolute LOD ceilings.
const vehicleBudgets = JSON.parse(
  await readFile(
    new URL("./fixtures/vehicle-render-budgets.json", import.meta.url),
    "utf8",
  ),
);

for (const style of ["futuristic", "steampunk"]) {
  for (const model of [
    "rocket",
    "kart",
    "drone",
    "harvester",
    "dock",
    "gate",
    "reactor",
  ]) {
    test(`${style} ${model}: packed assets, budgets, footprint and replay`, async () => {
      const sizes = [];
      for (const lod of ["high", "low"]) {
        const bytes = await readFile(
          new URL(
            `../web/lab/assets/${style}/${model}-${lod}.glb`,
            import.meta.url,
          ),
        );
        const json = glbJson(bytes);
        assert(json.buffers.every((buffer) => !buffer.uri));
        assert(json.images.length > 0);
        assert(
          json.images.every((image) => image.bufferView != null && !image.uri),
        );
        const triangles = json.meshes
          .flatMap((mesh) => mesh.primitives)
          .reduce(
            (sum, primitive) =>
              sum + json.accessors[primitive.indices].count / 3,
            0,
          );
        const budget = vehicleBudgets[`${style}/${model}-${lod}`];
        if (budget) {
          assert(
            triangles <= budget.triangles,
            "no additional vehicle triangles",
          );
          assert(
            json.meshes.reduce(
              (sum, mesh) => sum + mesh.primitives.length,
              0,
            ) <= budget.primitives,
            "no additional material draw batches",
          );
          assert(
            json.materials.length <= budget.materials,
            "no additional materials",
          );
          assert(
            json.images.length <= budget.images,
            "no additional texture images",
          );
          const binaryOffset = 28 + bytes.readUInt32LE(12);
          const pixels = json.images.reduce((sum, image) => {
            const view = json.bufferViews[image.bufferView];
            const offset = binaryOffset + (view.byteOffset || 0);
            assert.equal(
              bytes.toString("ascii", offset + 1, offset + 4),
              "PNG",
            );
            return (
              sum +
              bytes.readUInt32BE(offset + 16) * bytes.readUInt32BE(offset + 20)
            );
          }, 0);
          assert(
            pixels <= budget.texturePixels,
            "no additional decoded texture memory",
          );
          assert(bytes.length <= budget.bytes, "no additional download bytes");
        }
        assert(
          triangles <=
            (lod === "low"
              ? model === "harvester"
                ? 6000
                : 3000
              : model === "harvester"
                ? 80000
                : 50000),
        );
        const root = await geometryOnly(bytes, json);
        if (model === "rocket") {
          const glazing = json.materials.find((material) =>
            material.name.includes("glazing"),
          );
          assert(glazing, "rocket glazing material is present");
          assert.equal(
            glazing.alphaMode || "OPAQUE",
            lod === "high" ? "BLEND" : "OPAQUE",
          );
          assert(
            !glazing.extensions?.KHR_materials_transmission,
            "glazing needs no refraction pass",
          );
        }
        root.updateMatrixWorld(true);
        const bounds = new T.Box3();
        root.traverse((part) => {
          if (!part.isMesh) return;
          for (let p = part; p; p = p.parent)
            if (p.userData.motion === "thrust") return;
          part.geometry.computeBoundingBox();
          bounds.union(
            part.geometry.boundingBox.clone().applyMatrix4(part.matrixWorld),
          );
        });
        if (["rocket", "kart", "drone", "harvester"].includes(model)) {
          assert(
            bounds.min.x >= -0.8 && bounds.max.x <= 0.8,
            `${model} x bounds`,
          );
          assert(
            bounds.min.y >= -0.8 && bounds.max.y <= 0.8,
            `${model} y bounds`,
          );
          assert(bounds.min.z >= 0, `${model} ground alignment`);
          sizes.push(bounds.getSize(new T.Vector3()));
        }
        const parts = animatedParts(root);
        const motions = parts.map(({ part }) => part.userData.motion);
        if (model === "rocket")
          assert.equal(motions.filter((m) => m === "thrust").length, 2);
        if (model === "kart" || model === "harvester") {
          assert.equal(motions.filter((m) => m === "steer").length, 2);
          assert.equal(
            motions.filter((m) => m === "wheel").length,
            model === "kart" ? 4 : 7,
          );
          // Mirroring a Y-axis wheel must not invert its outward tread normals.
          let outward = 0,
            treadSamples = 0;
          root.traverse((mesh) => {
            if (
              !mesh.isMesh ||
              mesh.parent?.userData.motion !== "wheel" ||
              !mesh.material.name.startsWith("Tire rubber")
            )
              return;
            const { position, normal } = mesh.geometry.attributes;
            const radius = mesh.parent.userData.wheelRadius;
            for (let i = 0; i < position.count; i++) {
              const x = position.getX(i),
                z = position.getZ(i);
              const nx = normal.getX(i),
                nz = normal.getZ(i);
              if (Math.hypot(x, z) < radius * 0.9 || Math.hypot(nx, nz) < 0.3)
                continue;
              treadSamples++;
              if (x * nx + z * nz > 0) outward++;
            }
          });
          assert(treadSamples > 0, "wheel tread geometry is present");
          assert(
            outward / treadSamples > 0.85,
            "wheel tread normals face outward",
          );
        }
        if (model === "drone")
          assert.equal(motions.filter((m) => m === "rotor").length, 4);
        const frame = { time: 3, speed: 1.2, thrust: 0.6, steer: 0.3 };
        const pose = () => {
          root.updateMatrixWorld(true);
          return parts.map(({ part }) => [
            ...part.matrixWorld.elements,
            part.visible,
          ]);
        };
        animateAgent(parts, frame);
        const first = pose();
        animateAgent(parts, { ...frame, time: 20, thrust: 0, steer: -0.6 });
        animateAgent(parts, frame);
        assert.deepEqual(pose(), first);
      }
      if (sizes.length)
        assert(sizes[0].distanceTo(sizes[1]) < 0.04, "LOD proportions match");
    });
  }
}

test("screen-size LOD hysteresis and crowd override", () => {
  assert.equal(chooseLod(121), "high");
  assert.equal(chooseLod(100, "high"), "high");
  assert.equal(chooseLod(100, "low"), "low");
  assert.equal(chooseLod(89, "high"), "low");
  assert.equal(chooseLod(400, "high", true), "low");
});

test("latest style request wins; stale errors cannot replace the view", async () => {
  const pending = [];
  const writes = [];
  const service = new StyleService({
    preload: () =>
      new Promise((resolve, reject) => pending.push({ resolve, reject })),
    storage: { setItem: (...args) => writes.push(args) },
  });
  const commits = [];
  service.subscribe((style) => ({ commit: () => commits.push(style) }));
  const old = service.change("steampunk"),
    current = service.change("futuristic");
  pending[1].resolve();
  assert.equal(await current, true);
  pending[0].reject(new Error("Old request failed"));
  assert.equal(await old, false);
  assert.deepEqual(commits, ["futuristic"]);
  assert.deepEqual(writes, [[styleStorageKey, "futuristic"]]);
});

test("failed preparation leaves all views and preference unchanged; retry succeeds", async () => {
  let fail = true,
    cancel = 0,
    commit = 0;
  const service = new StyleService({ preload: async () => {} });
  service.subscribe(() => ({ commit: () => commit++, cancel: () => cancel++ }));
  service.subscribe(() => {
    if (fail) throw new Error("Asset missing");
  });
  await assert.rejects(service.change("steampunk"), /Asset missing/);
  assert.equal(service.current, "futuristic");
  assert.equal(commit, 0);
  assert.equal(cancel, 1);
  fail = false;
  await service.change("steampunk");
  assert.equal(commit, 1);
  assert.equal(service.current, "steampunk");
});

test("blocked storage and unknown saved styles use the default", async () => {
  const service = new StyleService({
    preload: async () => {},
    storage: {
      getItem: () => "unknown",
      setItem: () => {
        throw new Error("Denied");
      },
    },
  });
  assert.equal(service.preferred(), "futuristic");
  await service.change("steampunk");
  assert.equal(service.current, "steampunk");
  await assert.rejects(service.change("unknown"), /Unknown visual style/);
});

test("disposing a viewport preserves geometry and textures shared with another", () => {
  const geometry = new T.BoxGeometry(),
    map = new T.Texture();
  const material = new T.MeshStandardMaterial({ map });
  const asset = new T.Group();
  asset.add(new T.Mesh(geometry, material));
  retainAsset(asset);
  let disposed = 0;
  for (const resource of [geometry, map, material])
    resource.addEventListener("dispose", () => disposed++);
  const left = asset.clone(true),
    right = asset.clone(true);
  disposeGroup(left);
  assert.equal(disposed, 0);
  assert.equal(right.children[0].material.map, map);
  disposeGroup(right);
  disposeGroup(asset, true);
  assert.equal(disposed, 3);
});
