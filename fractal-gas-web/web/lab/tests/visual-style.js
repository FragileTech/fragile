// Browser integration check: real textures, renderer, native state and two views.
import { LabRenderer } from "../renderer.js";
import { NativeEngine, loadNative } from "../native.js";
import { labStyle } from "../visual-style.js";
import {
  assetModel,
  preloadStyle,
  disposeAssetCache,
} from "../visuals/assets.js";
import { animatedParts, animateAgent } from "../visuals/registry.js";
import * as T from "../vendor/three.module.js";
import { worldCatalog } from "../visuals/world-catalog.js";

const output = document.getElementById("result");
const checks = [],
  performanceRows = [];
function check(condition, message) {
  if (!condition) throw new Error(message);
  checks.push(message);
  output.textContent = `${checks.length} checks passed\n${message}`;
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const frame = () => new Promise(requestAnimationFrame);
async function measure(renderer) {
  for (let i = 0; i < 3; i++) await frame();
  const times = [],
    cpuTimes = [];
  let last = performance.now();
  for (let i = 0; i < 15; i++) {
    await frame();
    const now = performance.now();
    times.push(now - last);
    cpuTimes.push(renderer.performance.cpuMs);
    last = now;
  }
  times.sort((a, b) => a - b);
  cpuTimes.sort((a, b) => a - b);
  const geometry = [];
  renderer.world.traverseVisible((object) => {
    if (object.isMesh)
      geometry.push({
        name: object.name,
        triangles:
          ((object.geometry.index?.count ||
            object.geometry.attributes.position.count) /
            3) *
          (object.count ?? 1),
      });
  });
  return {
    medianFrameMs: +times[7].toFixed(2),
    medianRenderCpuMs: +cpuTimes[7].toFixed(2),
    pageVisibility: document.visibilityState,
    draws: renderer.performance.calls,
    triangles: renderer.performance.triangles,
    largestGeometry: geometry
      .sort((a, b) => b.triangles - a.triangles)
      .slice(0, 5),
  };
}
let a, b, engine;
try {
  const native = await loadNative(false);
  a = new LabRenderer(document.getElementById("a"));
  b = new LabRenderer(document.getElementById("b"));
  await preloadStyle("futuristic");
  await preloadStyle("steampunk");
  for (const style of ["futuristic", "steampunk"])
    for (const [kind, spec] of Object.entries(worldCatalog)) {
      const model = assetModel(style, kind);
      check(!!model, `${style}/${kind}: world asset loaded`);
      let envelope;
      model.traverse((part) => {
        if (part.userData.collisionEnvelope)
          envelope = part.userData.collisionEnvelope;
      });
      check(same(envelope, spec.size), `${style}/${kind}: shared envelope`);
      let decoded = true;
      model.traverse((part) => {
        for (const key of ["map", "normalMap", "roughnessMap", "emissiveMap"])
          if (part.material?.[key]) {
            const image = part.material[key].image;
            decoded &&=
              image?.width > 0 &&
              image.width <= 2048 &&
              image?.height > 0 &&
              image.height <= 2048;
          }
      });
      check(
        decoded,
        `${style}/${kind}: embedded textures decoded within budget`,
      );
    }
  for (const style of ["futuristic", "steampunk"]) {
    for (const model of ["rocket", "kart", "drone", "harvester"]) {
      const asset = assetModel(style, model);
      let maps = 0;
      asset.traverse((mesh) => {
        if (mesh.material?.map) {
          const image = mesh.material.map.image;
          check(
            image.width > 0 && image.width <= 2048 && image.height <= 2048,
            `${style}/${model} PBR texture decoded`,
          );
          maps++;
        }
      });
      check(maps > 0, `${style}/${model} has baked textures`);
    }
  }
  const catalog = await (await fetch("../scenario-catalog.json")).json();
  for (const { id } of catalog) {
    const scene = await (await fetch(`../scenarios/${id}.json`)).json();
    engine = new NativeEngine(native, scene);
    const action = engine.neutralAction();
    if (action.length) action[0] = 0.4;
    engine.step(action, 8);
    const state = engine.states(),
      snapshot = [...engine.snapshot()];
    a.load(scene, engine.info, engine.channels);
    b.load(scene, engine.info, engine.channels);
    for (const renderer of [a, b]) {
      renderer.update(state, action);
      renderer.focus(renderer.controlled[0]);
      renderer.select([3, 4]);
      renderer.selectMany([
        [3, 4],
        [5, 6],
      ]);
    }
    const overlays = a.overlays;
    const camera = [...a.camera.position, a.zoom, ...a.viewCenter];
    for (const style of ["steampunk", "futuristic"]) {
      await labStyle.change(style);
      check(
        a.style === style && b.style === style,
        `${id}: both viewports switch to ${style}`,
      );
      check(
        same([...a.camera.position, a.zoom, ...a.viewCenter], camera),
        `${id}: camera preserved`,
      );
      check(
        a.overlays === overlays && a.multiSelection.children.length === 2,
        `${id}: diagnostics and editor selection preserved`,
      );
      check(
        a.state === state && same([...engine.snapshot()], snapshot),
        `${id}: native and displayed state preserved`,
      );
      check(
        a.models[a.controlled[0]].userData.assetStyle === style ||
          a.models[a.controlled[0]].children.some(
            (child) => child.userData.assetStyle === style,
          ),
        `${id}: authored asset active`,
      );
      performanceRows.push({ scene: id, style, ...(await measure(a)) });
      if (id === "harvest") {
        const ores = a.bodyLayer.lods.filter((entry) => entry.span != null);
        a.bodyLayer.updateLod({ top: 100, bottom: -100 }, 400);
        check(
          ores.length > 0 &&
            ores.every((entry) => entry.low.visible && !entry.high.visible),
          `${style}: distant ore uses simplified geometry`,
        );
        a.bodyLayer.updateLod({ top: 1, bottom: -1 }, 900);
        check(
          ores.every((entry) => entry.high.visible && !entry.low.visible),
          `${style}: close ore uses detailed geometry`,
        );
        a.bodyLayer.updateLod(a.camera, a.canvas.clientHeight);
      }
    }
    const pose = () => {
      const parts = a.bodyLayer.animations[a.controlled[0]];
      a.models[a.controlled[0]].updateMatrixWorld(true);
      return parts.map(({ part }) => [
        ...part.matrixWorld.elements,
        part.visible,
      ]);
    };
    const effectsPose = () => {
      a.worldDynamics.group.updateMatrixWorld(true);
      const values = [];
      a.worldDynamics.group.traverse((part) =>
        values.push([...part.matrixWorld.elements, part.visible]),
      );
      for (const p of a.worldDynamics.pickups)
        if (p) values.push([...p.position, p.visible]);
      return values;
    };
    const effectsBefore = effectsPose();
    const before = pose();
    engine.step(action, 20);
    a.update(engine.states(), action);
    await labStyle.change("steampunk");
    await labStyle.change("futuristic");
    a.update(state, action);
    check(same(before, pose()), `${id}: replay pose restored after switching`);
    check(
      same(effectsBefore, effectsPose()),
      `${id}: resource and effect poses restored after switching`,
    );
    engine.dispose();
    engine = null;
  }
  const agents = await (await fetch("../agent-catalog.json")).json();
  const crowd = {
    size: [40, 40],
    task: "forage",
    agent_types: agents,
    bodies: Array.from({ length: 64 }, (_, i) => ({
      agent_type: ["rocket", "kart", "drone", "harvester"][i % 4],
      position: [4 + (i % 8) * 4, 4 + Math.floor(i / 8) * 4],
    })),
  };
  engine = new NativeEngine(native, crowd);
  const state = engine.states(),
    action = engine.neutralAction();
  a.load(crowd, engine.info, engine.channels);
  a.update(state, action);
  for (const style of ["steampunk", "futuristic"]) {
    await labStyle.change(style);
    check(
      a.bodyLayer.instances.length > 0 && a.bodyLayer.lods.length === 0,
      `64 mixed agents: ${style} uses crowd instancing`,
    );
    performanceRows.push({
      scene: "64 mixed vehicles",
      style,
      ...(await measure(a)),
    });
    const overview = a.bodyLayer.instances.reduce(
      (n, { mesh }) => n + mesh.count,
      0,
    );
    const snapshot = [...engine.snapshot()];
    a.focus(0);
    a.bodyLayer.updateLod(a.camera, a.canvas.clientHeight);
    const focused = a.bodyLayer.instances.reduce(
      (n, { mesh }) => n + mesh.count,
      0,
    );
    check(
      focused > 0 && focused < overview,
      `${style}: off-screen crowd instances are omitted`,
    );
    check(
      same(snapshot, [...engine.snapshot()]),
      `${style}: crowd culling preserves native state`,
    );
    performanceRows.push({
      scene: "64 mixed vehicles / close view",
      style,
      ...(await measure(a)),
    });
    a.focus(null);
    a.bodyLayer.updateLod(a.camera, a.canvas.clientHeight);
  }
  b.dispose();
  b = null;
  await labStyle.change("steampunk");
  check(
    a.models.length === 64,
    "Disposing a comparison viewport preserves shared assets",
  );
  const restored = new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("Context restore timed out")),
      10000,
    );
    a.canvas.addEventListener(
      "webglcontextrestored",
      () => {
        clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
  });
  a.renderer.forceContextLoss();
  await frame();
  a.renderer.forceContextRestore();
  await restored;
  await frame();
  check(
    a.style === "steampunk" && a.world.environment,
    "WebGL context restores the selected style",
  );
  a.dispose();
  a = null;
  disposeAssetCache();
  const originalFetch = window.fetch;
  window.fetch = (request, options) => {
    const url = typeof request === "string" ? request : request.url;
    return url?.endsWith("steampunk/rocket-high.glb")
      ? Promise.resolve(new Response("Missing test asset", { status: 404 }))
      : originalFetch(request, options);
  };
  let rejected = false;
  try {
    await preloadStyle("steampunk");
  } catch {
    rejected = true;
  } finally {
    window.fetch = originalFetch;
  }
  check(rejected, "A missing GLB rejects collection loading");
  await preloadStyle("steampunk");
  check(
    !!assetModel("steampunk", "rocket"),
    "Retry loads the missing GLB after recovery",
  );
  disposeAssetCache();
  output.textContent = JSON.stringify(
    { status: "passed", checks: checks.length, performance: performanceRows },
    null,
    2,
  );
  output.dataset.status = "passed";
} catch (error) {
  output.textContent = JSON.stringify(
    {
      status: "failed",
      passed: checks.length,
      error: error.stack,
      performance: performanceRows,
    },
    null,
    2,
  );
  output.dataset.status = "failed";
  console.error(error);
} finally {
  engine?.dispose();
  a?.dispose();
  b?.dispose();
}
