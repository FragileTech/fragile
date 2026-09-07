import { LabRenderer } from "../renderer.js";
import { NativeEngine, loadNative } from "../native.js";
import { labStyle } from "../visual-style.js";

const checks = [],
  rows = [];
const output = document.querySelector("#result");
const frame = () => new Promise(requestAnimationFrame);
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
function check(condition, message) {
  if (!condition) throw new Error(message);
  checks.push(message);
  output.textContent = `${checks.length} checks passed\n${message}`;
}
const kinds = ["rocket", "kart", "drone", "harvester"];
const native = await loadNative(false);
const renderer = new LabRenderer(document.querySelector("#world"));
renderer.renderer.setPixelRatio(0.5);
let engine;
function load(count) {
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
    })),
  };
  engine = new NativeEngine(native, scene);
  const action = engine.neutralAction();
  for (let i = 0; i < action.length; i++) action[i] = i % 2 ? 0.3 : 0.65;
  renderer.load(scene, engine.info, engine.channels);
  renderer.update(engine.states(), action);
  renderer.setAnimationPlayback({ playing: true });
}
async function measure(enabled) {
  renderer.setAnimationsEnabled(enabled);
  for (let i = 0; i < 2; i++) await frame();
  const cpu = [],
    animation = [],
    frames = [];
  let previous = performance.now();
  for (let i = 0; i < 5; i++) {
    await frame();
    const now = performance.now();
    frames.push(now - previous);
    previous = now;
    cpu.push(renderer.performance.cpuMs);
    animation.push(renderer.performance.animationCpuMs);
  }
  const median = (values) =>
    +values.sort((a, b) => a - b)[Math.floor(values.length / 2)].toFixed(4);
  return {
    enabled,
    animationCpuMs: median(animation),
    renderCpuMs: median(cpu),
    frameMs: median(frames),
    draws: renderer.performance.calls,
    triangles: renderer.performance.triangles,
    geometries: renderer.renderer.info.memory.geometries,
    textures: renderer.renderer.info.memory.textures,
  };
}
async function run({ contextLoss = false } = {}) {
  window.animationProgress = { checks, rows };
  for (const style of ["futuristic", "steampunk"]) {
    await labStyle.change(style);
    for (const count of [1, 16, 64, 128]) {
      load(count);
      let authored = false;
      renderer.bodyLayer.models[0].traverse((node) => {
        if (node.userData.assetStyle === style) authored = true;
      });
      check(
        authored,
        `${style}/${count}: authored GLB is ready before measurement`,
      );
      const nativeBefore = [...engine.snapshot()];
      const root = renderer.bodyLayer.models[0];
      const pose = [...root.position.toArray(), ...root.rotation.toArray()];
      // Warm existing effect meshes before comparing GPU allocations; Three
      // uploads their buffers lazily the first time an enabled plume is drawn.
      renderer.setAnimationsEnabled(true);
      for (let i = 0; i < 2; i++) await frame();
      const off = await measure(false),
        on = await measure(true);
      console.log(`Measured ${style}/${count}`);
      rows.push({ style, count, off, on });
      check(
        same(nativeBefore, [...engine.snapshot()]),
        `${style}/${count}: animation does not mutate native state`,
      );
      check(
        same(pose, [...root.position.toArray(), ...root.rotation.toArray()]),
        `${style}/${count}: physical root remains anchored`,
      );
      check(
        off.animationCpuMs === 0,
        `${style}/${count}: off skips cosmetic update block`,
      );
      check(
        off.geometries === on.geometries && off.textures === on.textures,
        `${style}/${count}: no animation geometry or texture allocations`,
      );
      check(
        Number.isFinite(on.draws) && Number.isFinite(on.triangles),
        `${style}/${count}: draw/triangle costs recorded`,
      );
      renderer.setAnimationPlayback({ playing: false, seek: true });
      check(
        renderer.bodyLayer.animationInputs.every((i) => i.wheelTravel === 0),
        `${style}/${count}: seeking resets cosmetic wheel distance`,
      );
      renderer.setAnimationsEnabled(false);
      check(
        renderer.bodyLayer.presentations.every(
          (p) => p.position.z === 0 && p.rotation.x === 0 && p.rotation.y === 0,
        ),
        `${style}/${count}: off restores cosmetic poses`,
      );
    }
  }
  load(4);
  renderer.setAnimationPlayback({ playing: false });
  renderer.setAnimationsEnabled(true);
  for (let i = 0; i < 2; i++) await frame();
  const hover = renderer.bodyLayer.presentations[2].position.z;
  for (let i = 0; i < 2; i++) await frame();
  check(
    renderer.bodyLayer.presentations[2].position.z !== hover,
    "paused simulation retains gentle idle motion",
  );
  const gl = renderer.renderer.getContext();
  const loss = gl.getExtension("WEBGL_lose_context");
  if (loss && contextLoss)
    for (const enabled of [false, true]) {
      renderer.setAnimationsEnabled(enabled);
      const lost = new Promise((resolve) =>
        renderer.canvas.addEventListener("webglcontextlost", resolve, {
          once: true,
        }),
      );
      loss.loseContext();
      await lost;
      const restored = new Promise((resolve) =>
        renderer.canvas.addEventListener("webglcontextrestored", resolve, {
          once: true,
        }),
      );
      // Let the loss event finish dispatching before requesting restoration.
      await new Promise((resolve) => setTimeout(resolve, 100));
      loss.restoreContext();
      await Promise.race([
        restored,
        new Promise((_, reject) =>
          setTimeout(
            () =>
              reject(
                new Error(
                  "WebGL restoration timed out; context QA inconclusive",
                ),
              ),
            5000,
          ),
        ),
      ]);
      for (let i = 0; i < 2; i++) await frame();
      check(
        renderer.animationsEnabled === enabled &&
          renderer.bodyLayer.animationsEnabled === enabled,
        `context restoration preserves ${enabled ? "enabled" : "disabled"} motion`,
      );
    }
  renderer.setAnimationsEnabled(false);
  await labStyle.change("futuristic");
  check(
    !renderer.bodyLayer.animationsEnabled,
    "style replacement preserves disabled preference",
  );
  window.animationResults = {
    checks,
    rows,
    userAgent: navigator.userAgent,
    visibility: document.visibilityState,
    viewport: [innerWidth, innerHeight],
    devicePixelRatio,
    rasterPixelRatio: renderer.renderer.getPixelRatio(),
    gpu: gl.getParameter(
      gl.getExtension("WEBGL_debug_renderer_info")?.UNMASKED_RENDERER_WEBGL ||
        gl.RENDERER,
    ),
    samplesPerMode: 5,
  };
  output.textContent = JSON.stringify(window.animationResults, null, 2);
  return window.animationResults;
}
window.animationFixture = {
  renderer,
  run,
  async showcase(style) {
    await labStyle.change(style);
    load(4);
    renderer.setAnimationsEnabled(true);
    renderer.setAnimationPlayback({ playing: true });
    document.querySelector("#title").textContent =
      `${style} — rocket · kart · drone · harvester`;
    for (let i = 0; i < 2; i++) await frame();
  },
};
