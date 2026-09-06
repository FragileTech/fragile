import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { glbJson, geometryOnly } from "./helpers/asset-glb.mjs";
import { assetManifest } from "../web/lab/visuals/asset-manifest.js";
import * as T from "../web/lab/vendor/three.module.js";
for (const style of ["steampunk", "futuristic"])
  test(`${style} refinery: packed materials, both LOD budgets, open apron and source`, async () => {
    for (const lod of ["high", "low"]) {
      const bytes = await readFile(new URL(assetManifest[style].refinery[lod]));
      const json = glbJson(bytes);
      assert(json.buffers.every((b) => !b.uri));
      assert(json.images.every((i) => i.bufferView != null && !i.uri));
      const triangles = json.meshes
        .flatMap((m) => m.primitives)
        .reduce((n, p) => n + json.accessors[p.indices].count / 3, 0);
      assert(triangles <= (lod === "low" ? 6000 : 50000));
      const model = await geometryOnly(bytes, json);
      model.updateMatrixWorld(true);
      const bounds = new T.Box3().setFromObject(model);
      assert(bounds.min.y >= -6.01 && bounds.max.y <= 10.3);
      assert(bounds.min.x >= -6.01 && bounds.max.x <= 6.01);
      // No machinery above the drive-through interior of the apron.
      model.traverse((p) => {
        if (!p.isMesh) return;
        const pos = p.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) {
          const v = new T.Vector3()
            .fromBufferAttribute(pos, i)
            .applyMatrix4(p.matrixWorld);
          if (Math.abs(v.x) < 5.5 && v.y < 5.8) assert(v.z < 0.12);
        }
      });
    }
    const source = await readFile(
      new URL(
        `../web/lab/assets/sources/${style}/refinery.blend`,
        import.meta.url,
      ),
    );
    assert(source.length > 10000);
  });
