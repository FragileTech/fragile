// Real-browser smoke check for every world/refinery style and LOD. No RAF loop;
// each completed case is persisted before the next render or style load.
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";

const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/";
const directory = process.env.WORLD_REVIEW_DIR || "/tmp/world-polish-browser";
await mkdir(directory, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
const report = {
  cases: [],
  note: "All model variants are decoded and pose-checked. Rendering/culling smoke checks sample visible props with direct lighting; no environment prefiltering or frame-rate claim. Blender family previews provide complete visual review.",
};
try {
  const page = await browser.newPage({
    viewport: { width: 1120, height: 900 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/world-polish-qa.html", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: '<html><body style="margin:0;background:#131d29;color:#eef;font:16px system-ui"><p id="label" style="margin:12px"></p><canvas></canvas></body></html>',
    }),
  );
  await page.goto(new URL("world-polish-qa.html", base).href);
  await page.evaluate(async () => {
    const T = await import("./vendor/three.module.js");
    const assets = await import("./visuals/assets.js");
    const { worldCatalog } = await import("./visuals/world-catalog.js");
    const { animateWorld, WorldInstances } = await import("./visuals/world.js");
    const renderer = new T.WebGLRenderer({
      canvas: document.querySelector("canvas"),
      antialias: true,
    });
    renderer.setSize(1120, 840);
    renderer.setPixelRatio(0.75);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    const scene = new T.Scene();
    scene.background = new T.Color(0x263342);
    scene.add(new T.HemisphereLight(0xd7eaff, 0x4b3944, 2.5));
    const light = new T.DirectionalLight(0xffecd8, 4);
    light.position.set(4, -8, 16);
    scene.add(light);
    const camera = new T.OrthographicCamera(-12, 12, 9, -9, 0.1, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(11, -16, 26);
    camera.lookAt(0, 0, 0);
    window.polish = {
      T,
      assets,
      worldCatalog,
      animateWorld,
      WorldInstances,
      renderer,
      scene,
      camera,
    };
  });
  for (const style of ["futuristic", "steampunk"])
    for (const lod of ["high", "low"]) {
      const row = await page.evaluate(
        async ({ style, lod }) => {
          const {
            T,
            assets,
            worldCatalog,
            animateWorld,
            WorldInstances,
            renderer,
            scene,
            camera,
          } = window.polish;
          await assets.preloadStyle(style);
          const group = new T.Group();
          const names = [...Object.keys(worldCatalog), "refinery"];
          const checks = [];
          const check = (condition, label) => {
            if (!condition) throw new Error(style + "/" + lod + ": " + label);
            checks.push(label);
          };
          const images = new Set();
          names.forEach((kind, index) => {
            const model = assets.assetModel(style, kind, lod);
            check(!!model, kind + " loads");
            const parts = [];
            model.traverse((part) => {
              parts.push([part, part.rotation.toArray()]);
              for (const mat of Array.isArray(part.material)
                ? part.material
                : [part.material])
                for (const key of [
                  "map",
                  "normalMap",
                  "roughnessMap",
                  "emissiveMap",
                ])
                  if (mat?.[key]?.image) images.add(mat[key].image);
            });
            animateWorld(model, 3);
            const pose = parts.map(([part]) => part.rotation.toArray());
            animateWorld(model, 19);
            animateWorld(model, 3);
            check(
              parts.every(
                ([part], i) =>
                  JSON.stringify(part.rotation.toArray()) ===
                  JSON.stringify(pose[i]),
              ),
              kind + " seek pose",
            );
            animateWorld(model, 5, { enabled: false });
            check(
              parts.every(
                ([part, rest]) =>
                  JSON.stringify(part.rotation.toArray()) ===
                  JSON.stringify(rest),
              ),
              kind + " animation off rest",
            );
            model.updateMatrixWorld(true);
            const bounds = new T.Box3().setFromObject(model);
            const center = bounds.getCenter(new T.Vector3());
            const size = bounds.getSize(new T.Vector3());
            const scale = 1.8 / Math.max(size.x, size.y, size.z);
            model.scale.setScalar(scale);
            model.position.set(
              ((index % 7) - 3) * 3 - center.x * scale,
              (Math.floor(index / 7) - 3) * 3 - center.y * scale,
              -bounds.min.z * scale,
            );
            group.add(model);
          });
          check(
            [...images].every(
              (im) =>
                im.width > 0 &&
                im.height > 0 &&
                im.width <= 2048 &&
                im.height <= 2048,
            ),
            "all embedded textures decoded",
          );
          scene.add(group);
          renderer.render(scene, camera);
          renderer.render(scene, camera);
          check(
            renderer.info.render.calls > 0 &&
              renderer.info.render.triangles > 0,
            "models rendered",
          );
          check(renderer.getContext().getError() === 0, "no WebGL error");
          const measured = {
            draws: renderer.info.render.calls,
            triangles: renderer.info.render.triangles,
          };
          // Repeated props use the production batch/culling path, independent of
          // this contact sheet's one-model-per-cell presentation.
          const holder = new T.Group();
          const batch = new WorldInstances(style, holder);
          for (let i = 0; i < 128; i++) batch.add("rail", [i * 5, 0, 0]);
          batch.build();
          batch.updateLod(camera, 840);
          const visible = batch.batches
            .filter((b) => b.mesh.visible)
            .reduce((n, b) => n + b.mesh.count, 0);
          check(
            visible > 0 && visible < 128 * batch.batches.length,
            "128 repeated props cull and compact",
          );
          for (const { mesh } of batch.batches) mesh.dispose();
          holder.clear();
          document.querySelector("#label").textContent =
            style +
            " / " +
            lod +
            " — 42 world props and refinery · animations off";
          window.polish.group = group;
          return {
            style,
            lod,
            models: names.length,
            checks,
            decodedImages: images.size,
            ...measured,
            rasterPixelRatio: renderer.getPixelRatio(),
          };
        },
        { style, lod },
      );
      report.cases.push(row);
      await writeFile(
        directory + "/report.json",
        JSON.stringify(report, null, 2) + "\n",
      );
      if (process.env.WORLD_CAPTURE)
        await page.screenshot({
          path: directory + "/" + style + "-" + lod + ".png",
        });
      await page.evaluate(() => {
        polish.scene.remove(polish.group);
        polish.group = null;
      });
      console.log(
        style + "/" + lod + ": " + row.checks.length + " checks passed",
      );
    }
  assert.equal(errors.length, 0, errors.join("\n"));
  assert.equal(report.cases.length, 4);
} finally {
  await browser.close();
}
