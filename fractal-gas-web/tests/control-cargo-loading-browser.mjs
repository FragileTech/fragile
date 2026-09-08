// Bounded visual review of actual vehicle GLBs with runtime cargo. No environment
// prefiltering or context-restoration retry loop. Persist each completed case.
import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/";
const output = process.env.CARGO_REVIEW_DIR || "/tmp/cargo-loading-review";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
const report = { cases: [], status: "running" };
try {
  const page = await browser.newPage({
    viewport: { width: 1400, height: 900 },
  });
  await page.route("**/cargo-loading-review.html", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: '<html><head><meta charset="utf-8"></head><body style="margin:0;background:#172536"><div style="position:relative;width:1400px;height:900px"><canvas></canvas></div></body></html>',
    }),
  );
  await page.goto(new URL("cargo-loading-review.html", base).href);
  await page.evaluate(async () => {
    const T = await import("./vendor/three.module.js");
    const { GLTFLoader } = await import(
      "./vendor/addons/loaders/GLTFLoader.js"
    );
    const { CargoVisuals } = await import("./visuals/cargo.js");
    const { CargoReadout } = await import("./visuals/cargo-readout.js");
    const renderer = new T.WebGLRenderer({
      canvas: document.querySelector("canvas"),
      antialias: true,
    });
    renderer.setSize(1400, 900);
    renderer.setPixelRatio(1);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    const camera = new T.OrthographicCamera(-4.9, 4.9, 3.15, -3.15, 0.1, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(5, -9, 12);
    camera.lookAt(0, 0, 0);
    const readout = new CargoReadout(renderer.domElement);
    readout.resize(1400, 900);
    window.review = { T, GLTFLoader, CargoVisuals, renderer, camera, readout };
  });
  for (const style of process.env.CARGO_BUDGET_ONLY
    ? []
    : ["futuristic", "steampunk"])
    for (const lod of ["high", "low"]) {
      await page.evaluate(
        async ({ style, lod }) => {
          const { T, GLTFLoader, CargoVisuals, renderer, camera, readout } =
            window.review;
          const world = new T.Scene();
          world.background = new T.Color(
            style === "steampunk" ? 0x302a24 : 0x172536,
          );
          world.add(new T.HemisphereLight(0xe6efff, 0x313c50, 3));
          const light = new T.DirectionalLight(0xffeedb, 4);
          light.position.set(3, -7, 12);
          world.add(light);
          const floor = new T.Mesh(
            new T.PlaneGeometry(30, 30),
            new T.MeshStandardMaterial({
              color: style === "steampunk" ? 0x252019 : 0x11232e,
              roughness: 1,
            }),
          );
          floor.position.z = -0.06;
          world.add(floor);
          const kinds = ["rocket", "kart", "drone", "harvester"],
            models = [],
            presentations = [];
          for (let i = 0; i < kinds.length; i++) {
            const gltf = await new GLTFLoader().loadAsync(
              `./assets/${style}/${kinds[i]}-${lod}.glb`,
            );
            const model = new T.Group(),
              pose = new T.Group();
            pose.add(gltf.scene);
            model.add(pose);
            world.add(model);
            model.position.set(i % 2 ? 2 : -2, i < 2 ? 1.4 : -1.4, 0.1);
            models.push(model);
            presentations.push(pose);
          }
          const layer = {
            controlled: [0, 1, 2, 3],
            models,
            presentations,
            bodies: kinds.map((model) => ({ visual: { model } })),
            active: [true, true, true, true],
            inView: [true, true, true, true],
            presentationVersion: 0,
          };
          const info = new Uint32Array(16);
          info[1] = 4;
          info[15] = 32;
          const state = new Float32Array(48),
            bits = new Uint32Array(state.buffer);
          for (let i = 0; i < 4; i++) {
            state[8 + i] = models[i].position.x;
            state[12 + i] = models[i].position.y;
          }
          const config = {
            cargo: { capacity: 5 },
            physics: { dt: 1 / 60 },
            refineries: models.map((m) => ({
              position: [m.position.x, m.position.y],
              radius: 1.1,
            })),
          };
          const cargo = new CargoVisuals(config, info, layer, world, style);
          const set = (tick, amount, phase = 0, delivered = 0) => {
            bits[0] = tick;
            for (let i = 0; i < 4; i++) {
              state[32 + i * 4] = amount;
              state[33 + i * 4] = phase;
              state[34 + i * 4] = delivered;
            }
            cargo.update(state);
          };
          window.review.current = {
            world,
            layer,
            cargo,
            state,
            set,
            style,
            lod,
          };
          set(0, 0);
          if (cargo.fill.count !== 0) throw new Error("empty cargo geometry");
        },
        { style, lod },
      );
      for (const mode of ["loading", "full-off", "unloading"]) {
        const result = await page.evaluate((mode) => {
          const { renderer, camera, readout } = window.review;
          const { world, layer, cargo, state, set, style, lod } =
            window.review.current;
          if (mode === "loading") {
            set(1, 3);
            set(13, 3);
          }
          if (mode === "full-off") {
            set(80, 5);
            cargo.setAnimationsEnabled(false);
          }
          if (mode === "unloading") {
            cargo.setAnimationsEnabled(true);
            set(100, 4, 1, 1);
          }
          const original = Array.from(new Uint8Array(state.buffer));
          world.updateMatrixWorld(true);
          renderer.render(world, camera);
          if (renderer.getContext().isContextLost())
            throw new Error("WebGL context lost");
          if (renderer.info.render.triangles < 100)
            throw new Error("vehicle geometry not rendered");
          readout.update(cargo.entries, layer, camera, world, 3, null, style);
          if (!original.every((v, i) => v === new Uint8Array(state.buffer)[i]))
            throw new Error("state mutated");
          const transfer = cargo.transfer.count;
          if (mode !== "full-off" && !cargo.transfer.visible)
            throw new Error("transfer missing");
          if (mode === "full-off" && cargo.transfer.visible)
            throw new Error("disabled transfer visible");
          return {
            style,
            lod,
            mode,
            cargoDraws: [cargo.fill, cargo.meter, cargo.transfer].filter(
              (m) => m.visible,
            ).length,
            triangles: renderer.info.render.triangles,
            calls: renderer.info.render.calls,
            transferInstances: transfer,
            labels: readout.count,
          };
        }, mode);
        report.cases.push(result);
        await page.screenshot({
          path: `${output}/${style}-${lod}-${mode}.png`,
        });
        await writeFile(
          `${output}/results.json`,
          JSON.stringify(report, null, 2),
        );
      }
      await page.evaluate(async () => {
        const { disposeGroup } = await import("./visuals/resources.js");
        disposeGroup(window.review.current.world);
        window.review.renderer.renderLists.dispose();
      });
    }
  report.crowds = [];
  for (const count of [1, 16, 64, 128]) {
    const rows = await page.evaluate(async (count) => {
      const { T, CargoVisuals, renderer } = window.review;
      const { disposeGroup } = await import("./visuals/resources.js");
      const world = new T.Scene();
      const info = new Uint32Array(16);
      info[1] = count;
      info[15] = 8 + count * 6;
      const state = new Float32Array(info[15] + count * 4);
      const controlled = Array.from({ length: count }, (_, i) => i);
      const layer = {
        controlled,
        models: controlled.map(() => new T.Group()),
        bodies: controlled.map(() => ({ visual: { model: "drone" } })),
        presentations: [],
        active: controlled.map(() => true),
        inView: controlled.map(() => true),
        presentationVersion: 0,
      };
      for (const i of controlled) {
        state[8 + i] = (i % 16) - 8;
        state[8 + count + i] = Math.floor(i / 16) - 4;
        state[info[15] + 4 * i] = 6;
        state[info[15] + 4 * i + 1] = 1;
      }
      const cargo = new CargoVisuals(
        {
          cargo: { capacity: 6 },
          refineries: [{ position: [0, 0], radius: 20 }],
        },
        info,
        layer,
        world,
        "futuristic",
      );
      const camera = new T.OrthographicCamera(-20, 20, 15, -15, 0.1, 100);
      camera.position.z = 30;
      camera.lookAt(0, 0, 0);
      const rows = [];
      for (const animations of [true, false]) {
        cargo.setAnimationsEnabled(animations);
        cargo.update(state);
        renderer.render(world, camera);
        const calls = renderer.info.render.calls,
          triangles = renderer.info.render.triangles;
        if (
          calls !== (animations ? 3 : 2) ||
          triangles !== (animations ? 106 : 58) * count
        )
          throw new Error(
            `Cargo cost check: count=${count}, animations=${animations}, calls=${calls}, triangles=${triangles}, contextLost=${renderer.getContext().isContextLost()}`,
          );
        rows.push({
          count,
          animations,
          calls,
          triangles,
          geometryBudget: 106 * count,
          labelsDraws: 0,
        });
      }
      layer.inView.fill(false);
      layer.presentationVersion++;
      cargo.animate();
      renderer.render(world, camera);
      if (renderer.info.render.calls !== 0)
        throw new Error("Culled cargo still draws");
      disposeGroup(world);
      renderer.renderLists.dispose();
      return rows;
    }, count);
    report.crowds.push(...rows);
    await writeFile(`${output}/results.json`, JSON.stringify(report, null, 2));
  }
  report.status = "passed";
} catch (error) {
  report.status = "failed";
  report.error = String(error);
  throw error;
} finally {
  await writeFile(`${output}/results.json`, JSON.stringify(report, null, 2));
  await browser.close();
}
