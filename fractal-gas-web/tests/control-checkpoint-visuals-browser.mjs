import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage({
    serviceWorkers: "block",
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  await page.addInitScript(() =>
    localStorage.setItem("lab.workspace.onboarded", "true"),
  );
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("installStyleControls();", "cancelAnimationFrame(renderer.frame); installStyleControls();")}
      window.checkpointTest = { renderer, replay, loadScene, get ready() { return ready; }, get frame() { return lastLiveFrame; }, get worker() { return worker; } };`,
    });
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8099/lab/");
  await page.waitForFunction(() => window.checkpointTest?.ready, null, {
    timeout: 60000,
  });
  await page.evaluate(() =>
    document
      .querySelectorAll("dialog[open]")
      .forEach((dialog) => dialog.close()),
  );
  const scene = {
    name: "Checkpoint feedback test",
    task: "tandem",
    size: [40, 40],
    physics: { dt: 0.1, substeps: 2 },
    environment: { flight: true, downward_gravity: 0 },
    bodies: [
      { controlled: true, position: [15, 20], radius: 0.1, drag: 0 },
      { position: [30, 30] },
      {
        controlled: true,
        position: [21, 20],
        velocity: [-10, 0],
        radius: 0.1,
        drag: 0,
      },
    ],
    gates: [
      { position: [15, 20], radius: 1 },
      { position: [24, 26], radius: 2 },
    ],
  };
  await page.evaluate((scene) => {
    for (const [id, value] of Object.entries({
      walkers: 8,
      horizon: 2,
      frames: 1,
      threads: 1,
    }))
      document.getElementById(id).value = value;
    document.getElementById("persistent-recording").checked = true;
    checkpointTest.renderer.setAnimationsEnabled(true);
    checkpointTest.loadScene(scene);
  }, scene);
  await page.waitForFunction(
    () =>
      checkpointTest.ready &&
      checkpointTest.renderer.config.name === "Checkpoint feedback test" &&
      checkpointTest.frame.tick === 0,
  );
  await page.evaluate(() => {
    const p = checkpointTest.renderer.checkpoints;
    window.checkpointResources = {
      component: p,
      geometry: p.ringGeometry,
      material: p.markers[0].ring.material,
    };
    p.now = () => 1000;
  });
  const inspect = () =>
    page.evaluate(() => {
      const r = checkpointTest.renderer,
        p = r.checkpoints;
      p.project(r.camera);
      r.renderer.render(r.world, r.camera);
      return {
        active: p.active,
        crossed: p.crossed,
        text: p.label?.textContent,
        colors: p.markers.map((m) => m.ring.material.color.getHex()),
        flashes: Array.from(p.flashes, (v) => Number.isFinite(v)),
        hidden: p.label?.hidden,
      };
    });
  assert.equal((await inspect()).text, "Checkpoint 1 · 0/2 crossed");
  const advance = async (frames, tick) => {
    await page.evaluate(
      (frames) =>
        checkpointTest.worker.postMessage({
          type: "manual",
          action: new Float32Array(checkpointTest.frame.action.length),
          frames,
        }),
      frames,
    );
    await page.waitForFunction(
      (tick) => checkpointTest.frame.tick === tick,
      tick,
    );
  };
  await advance(1, 1);
  let current = await inspect();
  assert.equal(current.text, "Checkpoint 1 · 1/2 crossed");
  assert.equal(current.colors[0], 0x7bffc1);
  await advance(5, 6);
  current = await inspect();
  assert.equal(current.text, "Checkpoint 2 · 0/2 crossed");
  assert.equal(current.colors[0], 0x7bffc1);
  assert.equal(current.colors[1], 0xffce70);
  await page.waitForFunction(() => checkpointTest.replay.recording.length >= 7);
  await page.evaluate(() => checkpointTest.replay.playback.seek(0));
  assert.deepEqual((await inspect()).flashes, [false, false]);
  assert.equal((await inspect()).active, 0);
  await page.evaluate(() => checkpointTest.replay.playback.play());
  await page.waitForFunction(
    () =>
      checkpointTest.replay.playback.cursor === 6 &&
      !checkpointTest.replay.playback.playing,
  );
  await page.waitForFunction(
    () => checkpointTest.renderer.checkpoints.active === 1,
  );
  assert.deepEqual((await inspect()).flashes, [true, false]);
  await page.evaluate(() => checkpointTest.replay.playback.seek(6));
  assert.deepEqual((await inspect()).flashes, [false, false]);
  assert.equal((await inspect()).active, 1);
  await page.evaluate(() => document.getElementById("motion-live").click());
  assert.deepEqual((await inspect()).flashes, [false, false]);
  for (const style of ["futuristic", "steampunk"]) {
    await page.evaluate(
      (style) => checkpointTest.renderer.setStyle(style),
      style,
    );
    for (const top of [false, true]) {
      await page.evaluate(
        (top) => checkpointTest.renderer.setViewPreset(top),
        top,
      );
      const state = await inspect();
      assert.equal(state.colors[1], 0xffce70);
      assert.equal(state.hidden, false);
      await page.screenshot({
        path: `/tmp/checkpoints-${style}-${top ? "overhead" : "side"}.png`,
      });
    }
  }
  assert.equal(
    await page.evaluate(() => {
      const p = checkpointTest.renderer.checkpoints,
        s = checkpointResources;
      return (
        p === s.component &&
        p.ringGeometry === s.geometry &&
        p.markers[0].ring.material === s.material
      );
    }),
    true,
  );
  await page.evaluate(() =>
    checkpointTest.renderer.setLayers({
      tethers: false,
      tree: false,
      cloud: false,
      geometry: false,
    }),
  );
  assert.equal((await inspect()).hidden, false);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.waitForFunction(
    () => !checkpointTest.renderer.checkpoints.enabled,
  );
  await page.evaluate(() => {
    const r = checkpointTest.renderer,
      state = r.state.slice(),
      bits = new Uint32Array(state.buffer);
    bits[0] += 1;
    bits[r.info[6]] += 1;
    r.update(state, r.action);
  });
  assert.deepEqual((await inspect()).flashes, [false, false]);
  assert.equal((await inspect()).text, "Checkpoint 2 · 1/2 crossed");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.evaluate(() =>
    checkpointTest.renderer.setAnimationsEnabled(false),
  );
  assert.deepEqual((await inspect()).flashes, [false, false]);
  await page.evaluate((scene) => {
    window.checkpointDisposals = 0;
    checkpointResources.geometry.addEventListener(
      "dispose",
      () => ++checkpointDisposals,
    );
    checkpointResources.material.addEventListener(
      "dispose",
      () => ++checkpointDisposals,
    );
    checkpointTest.loadScene({
      ...scene,
      task: "navigation",
      name: "Other task",
    });
  }, scene);
  await page.waitForFunction(
    () =>
      checkpointTest.ready &&
      checkpointTest.renderer.config.name === "Other task",
  );
  assert.equal(await page.evaluate(() => checkpointDisposals), 2);
  assert.equal(await page.locator(".checkpoint-readout").count(), 0);
  assert.deepEqual(errors, []);
  console.log(
    "Checkpoint highlights, flashes, seeking, styles/views, motion preferences and resource reuse passed",
  );
} finally {
  await browser.close();
}
