import { DoubleSide } from "../vendor/three.module.js";
import { LabRenderer } from "../renderer.js";
import { NativeEngine, loadNative } from "../native.js";
import { labStyle } from "../visual-style.js";

const checks = [],
  rows = [];
const native = await loadNative(false);
const r = new LabRenderer(document.querySelector("#world"));
cancelAnimationFrame(r.frame); // Explicit renders keep software GPU cost bounded.
r.renderer.setPixelRatio(0.5);
let engine, action;
const kinds = ["rocket", "kart", "drone", "harvester"];
const check = (value, message) => {
  if (!value) throw new Error(message);
  checks.push(message);
};
function load(count, thrusters = false, centerJet = false) {
  engine?.dispose();
  const width = Math.ceil(Math.sqrt(count));
  const scene = {
    size: [Math.max(6, width * 3), Math.max(6, width * 3)],
    physics: { dt: 1 / 60 },
    bodies: Array.from({ length: count }, (_, i) => ({
      controlled: true,
      radius: 0.6,
      mass: 1,
      drag: 0.1,
      thrust: 3,
      visual: { model: kinds[i % 4] },
      position: [1.5 + (i % width) * 3, 1.5 + Math.floor(i / width) * 3],
      actuator: thrusters
        ? {
            kind: "thrusters",
            thrusters: Array.from({ length: centerJet ? 1 : 32 }, (_, j) => ({
              position: centerJet
                ? [0, 0]
                : [
                    Math.cos((j * Math.PI) / 16) * 0.5,
                    Math.sin((j * Math.PI) / 16) * 0.5,
                  ],
              direction: [
                Math.cos((j * Math.PI) / 16),
                Math.sin((j * Math.PI) / 16),
              ],
              force: 1,
              reversible: true,
            })),
          }
        : { kind: i % 4 === 0 ? "vector" : i % 4 === 2 ? "holonomic" : "kart" },
    })),
  };
  engine = new NativeEngine(native, scene);
  action = engine.neutralAction();
  const values = {
    thrust: 0.75,
    torque: -0.6,
    throttle: -0.7,
    steering: 0.8,
    brake: 0.5,
    force_x: 0.65,
    force_y: -0.8,
  };
  engine.channels.forEach((c, i) => {
    action[i] = c.name.startsWith("thruster_") ? 0.6 : values[c.name] || 0;
  });
  engine.step(action, 2);
  r.load(scene, engine.info, engine.channels);
  r.update(engine.states(), action);
  r.bodyLayer.updateLod(r.camera, r.canvas.clientHeight);
}
function passes(mesh) {
  return mesh.visible
    ? mesh.material.transparent &&
      mesh.material.side === DoubleSide &&
      !mesh.material.forceSinglePass
      ? 2
      : 1
    : 0;
}
function triangles(mesh) {
  return mesh.visible
    ? ((mesh.geometry.index?.count ?? mesh.geometry.attributes.position.count) /
        3) *
        (mesh.isInstancedMesh ? mesh.count : 1)
    : 0;
}
function render() {
  r.renderer.render(r.world, r.camera);
  return {
    draws: r.renderer.info.render.calls,
    triangles: r.renderer.info.render.triangles,
  };
}
function cpu(enabled, guides) {
  const iterations = 120;
  let start = performance.now();
  for (let i = 0; i < iterations; i++) {
    r.bodyLayer.animate(1 / 60, i / 60, { playing: true, speed: 1 });
    r.actionEffects.update();
  }
  const frameMs = (performance.now() - start) / iterations;
  start = performance.now();
  for (let i = 0; i < 20; i++) {
    r.bodyLayer.update(engine.states(), action);
    r.actionEffects.update();
    if (guides) r.actionEffects.updateGuides(r.actionGuideBody, true);
  }
  const commandUpdateMs = (performance.now() - start) / 20;
  return {
    cosmeticAndEffectsMs: +frameMs.toFixed(5),
    commandUpdateMs: +commandUpdateMs.toFixed(5),
    iterations,
  };
}
async function save(row) {
  rows.push(row);
  document.querySelector("#result").textContent = JSON.stringify(row, null, 2);
  await window.saveReadabilityRow?.(row);
}
async function run() {
  for (const style of ["futuristic", "steampunk"]) {
    await labStyle.change(style);
    for (const count of [1, 16, 64, 128]) {
      load(count);
      let authored = false;
      r.models[0].traverse((p) => {
        if (p.userData.assetStyle === style) authored = true;
      });
      check(authored, `${style}/${count}: authored assets ready`);
      const snapshot = [...engine.snapshot()],
        roots = r.models.map((m) => [
          ...m.position.toArray(),
          ...m.rotation.toArray(),
        ]);
      for (const enabled of [false, true])
        for (const guides of [false, true]) {
          r.setAnimationsEnabled(enabled);
          r.actionGuidesEnabled = guides;
          r.setActionGuideBody(count > 1 ? 1 : 999);
          r.bodyLayer.animate(1 / 30, 1, { playing: true, speed: 1 });
          r.actionEffects.update();
          r.refreshActionGuides();
          const effects = r.actionEffects;
          check(
            effects.guideBody === (count > 1 ? 1 : 0),
            `${style}/${count}: selected guide or fallback`,
          );
          check(
            effects.guides.visible === guides,
            `${style}/${count}: guide toggle independent of motion`,
          );
          const effectsDraws = passes(effects.jets) + passes(effects.lamps);
          const effectTriangles =
            triangles(effects.jets) * passes(effects.jets) +
            triangles(effects.lamps) * passes(effects.lamps);
          const guideDraws = passes(effects.guides),
            guideTriangles = guides
              ? effects.guideGeometry.drawRange.count / 3
              : 0;
          check(
            effectsDraws <= 2 && guideDraws <= 2,
            `${style}/${count}: shared draw budgets`,
          );
          check(
            effectTriangles <= 128 * count && guideTriangles <= 256,
            `${style}/${count}: shared triangle budgets`,
          );
          render(); // Warm existing buffers and materials once.
          effects.group.visible = false;
          const baseline = render();
          effects.group.visible = true;
          const measured = render();
          check(
            measured.draws - baseline.draws <= 4,
            `${style}/${count}: measured added draw budget`,
          );
          check(
            measured.triangles - baseline.triangles <= 128 * count + 256,
            `${style}/${count}: measured added triangle budget`,
          );
          const timing = cpu(enabled, guides);
          const nativeAfter = engine.snapshot();
          check(
            snapshot.every((v, i) => v === nativeAfter[i]),
            `${style}/${count}: native state unchanged`,
          );
          check(
            roots.every((p, i) =>
              p.every(
                (v, j) =>
                  v ===
                  [
                    ...r.models[i].position.toArray(),
                    ...r.models[i].rotation.toArray(),
                  ][j],
              ),
            ),
            `${style}/${count}: physical roots unchanged`,
          );
          if (!enabled) {
            const version = r.bodyLayer.presentationVersion;
            r.bodyLayer.animate(1 / 30, 3, { playing: true });
            check(
              r.bodyLayer.presentationVersion === version,
              `${style}/${count}: off skips cosmetic work`,
            );
          }
          await save({
            style,
            count,
            animations: enabled,
            guides,
            effectsDraws,
            guideDraws,
            effectTriangles,
            guideTriangles,
            baseline,
            measured,
            ...timing,
          });
        }
      r.setAnimationPlayback({ seek: true, playing: false });
      check(
        r.bodyLayer.animationInputs.every((i) => i.wheelTravel === 0),
        `${style}/${count}: seek resets wheel travel`,
      );
      r.setAnimationsEnabled(false);
      r.bodyLayer.update(engine.states(), action);
      const flame = r.bodyLayer.animations[0].find(
        (p) => p.part.userData.motion === "thrust",
      );
      check(
        flame?.part.visible,
        `${style}/${count}: main command remains visible with motion off`,
      );
      const zero = engine.neutralAction();
      r.update(engine.states(), zero);
      check(
        !flame.part.visible,
        `${style}/${count}: zero command clears static main flame`,
      );
    }
    load(1, true);
    r.setAnimationsEnabled(false);
    r.actionGuidesEnabled = true;
    r.setActionGuideBody(0);
    r.actionEffects.update();
    r.refreshActionGuides();
    check(
      r.bodyLayer.commands[0].kind === "thrusters",
      `${style}: independent controls loaded`,
    );
    check(
      r.actionEffects.jets.count === 32,
      `${style}: all 32 individual jets visible`,
    );
    check(
      triangles(r.actionEffects.jets) <= 128,
      `${style}: independent jet triangle cap`,
    );
    check(
      r.actionEffects.guideGeometry.drawRange.count / 3 <= 256,
      `${style}: independent guide triangle cap`,
    );
  }
  r.setAnimationsEnabled(false);
  r.actionGuidesEnabled = true;
  await labStyle.change("futuristic");
  check(
    !r.bodyLayer.animationsEnabled && r.actionGuidesEnabled,
    "style swap preserves motion and guide preferences",
  );
  const gl = r.renderer.getContext(),
    debug = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    checks,
    rows,
    rasterPixelRatio: r.renderer.getPixelRatio(),
    gpu: gl.getParameter(debug?.UNMASKED_RENDERER_WEBGL || gl.RENDERER),
    note: "CPU microbenchmarks exclude rasterization. GPU budget checks use two warm/baseline renders and one measured render; no frame-rate claim.",
  };
}
async function contextCheck() {
  await labStyle.change("futuristic");
  load(1);
  r.setAnimationsEnabled(false);
  r.actionGuidesEnabled = true;
  r.setActionGuideBody(0);
  r.actionEffects.update();
  r.refreshActionGuides();
  render();
  const gl = r.renderer.getContext(),
    loss = gl.getExtension("WEBGL_lose_context");
  if (!loss)
    return {
      outcome: "inconclusive",
      reason: "WEBGL_lose_context unavailable",
    };
  const flame = r.bodyLayer.animations[0].find(
    (p) => p.part.userData.motion === "thrust",
  );
  const before = {
    thrust: r.bodyLayer.commands[0].thrust,
    torque: r.bodyLayer.commands[0].torque,
    flame: flame.part.scale.x,
    jets: r.actionEffects.jets.count,
  };
  let timer;
  const restored = new Promise((resolve) =>
    r.canvas.addEventListener("webglcontextrestored", () => resolve(true), {
      once: true,
    }),
  );
  r.canvas.addEventListener(
    "webglcontextlost",
    () => setTimeout(() => loss.restoreContext(), 250),
    { once: true },
  );
  loss.loseContext();
  const ok = await Promise.race([
    restored,
    new Promise((resolve) => {
      timer = setTimeout(() => resolve(false), 5000);
    }),
  ]);
  clearTimeout(timer);
  if (!ok)
    return {
      outcome: "inconclusive",
      reason: "No context-restored event within five seconds",
    };
  const drainErrors = () => {
    const errors = [];
    for (let i = 0; i < 8; i++) {
      const error = gl.getError();
      if (error === gl.NO_ERROR) break;
      errors.push(error);
    }
    return errors;
  };
  const preRenderErrors = drainErrors();
  const measured = render();
  const diagnostics = {
    rendererMotionOff: !r.animationsEnabled,
    bodyMotionOff: !r.bodyLayer.animationsEnabled,
    guidesEnabled: r.actionGuidesEnabled,
    flameVisible: flame.part.visible,
    flameStable: flame.part.scale.x === before.flame,
    thrustStable: r.bodyLayer.commands[0].thrust === before.thrust,
    torqueStable: r.bodyLayer.commands[0].torque === before.torque,
    jetsStable: r.actionEffects.jets.count === before.jets,
    geometries: r.renderer.info.memory.geometries,
    preRenderErrors,
    postRenderErrors: drainErrors(),
  };
  const valid =
    diagnostics.rendererMotionOff &&
    diagnostics.bodyMotionOff &&
    diagnostics.guidesEnabled &&
    diagnostics.flameVisible &&
    diagnostics.flameStable &&
    diagnostics.thrustStable &&
    diagnostics.torqueStable &&
    diagnostics.jetsStable &&
    diagnostics.geometries > 0 &&
    measured.draws > 0 &&
    diagnostics.postRenderErrors.length === 0;
  return {
    outcome: valid ? "passed" : "failed",
    before,
    measured,
    diagnostics,
  };
}
window.readabilityFixture = {
  run,
  contextCheck,
  renderer: r,
  async showcaseCue({ style, kind, lod }) {
    await labStyle.change(style);
    load(
      kind === "center-jet" ? 1 : 4,
      kind === "center-jet",
      kind === "center-jet",
    );
    const body =
      kind === "drone" ? 2 : kind === "kart" ? 1 : kind === "harvester" ? 3 : 0;
    action.fill(0);
    engine.channels.forEach((c, i) => {
      if (
        c.body === body &&
        c.name ===
          (kind === "drone"
            ? "force_y"
            : kind === "center-jet"
              ? "thruster_0"
              : "torque")
      )
        action[i] = 0.9;
    });
    if (kind === "kart" || kind === "harvester")
      engine.channels.forEach((c, i) => {
        if (c.body === body)
          action[i] =
            c.name === "throttle" ? -0.9 : c.name === "brake" ? 0.7 : 0;
      });
    r.update(engine.states(), action);
    r.setAnimationsEnabled(false);
    r.actionGuidesEnabled = false;
    r.refreshActionGuides();
    r.focus(body);
    r.zoom = 2;
    r.resize();
    for (const level of r.bodyLayer.lods) {
      level.current = lod;
      level.high.visible = lod === "high";
      level.low.visible = lod === "low";
    }
    r.bodyLayer.presentationVersion++;
    r.actionEffects.update();
    document.querySelector("#title").textContent =
      `${style} / ${kind} / ${lod} — command only, no guides`;
    render();
    await new Promise(requestAnimationFrame);
    render();
  },
  async showcase(style) {
    await labStyle.change(style);
    load(4);
    r.setAnimationsEnabled(false);
    r.actionGuidesEnabled = true;
    r.setActionGuideBody(2);
    r.actionEffects.update();
    r.refreshActionGuides();
    document.querySelector("#title").textContent =
      `${style}: static command effects and signed guides`;
    render();
  },
};
