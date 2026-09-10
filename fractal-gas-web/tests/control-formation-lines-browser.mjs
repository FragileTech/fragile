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
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem("lab.workspace.onboarded", "true");
    localStorage.setItem("lab.workspace.panel", "view");
  });
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("installStyleControls();", "cancelAnimationFrame(renderer.frame); installStyleControls();")}
        window.formationTest = { renderer, editor, replay, loadScene,
          get ready() { return ready; }, get frame() { return lastLiveFrame; },
          get worker() { return worker; } };`,
    });
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8099/lab/");
  await page.waitForFunction(() => window.formationTest?.ready, null, {
    timeout: 60000,
  });
  await page.evaluate(() => {
    for (const [id, value] of Object.entries({
      walkers: 8,
      horizon: 2,
      frames: 1,
      threads: 1,
    }))
      document.getElementById(id).value = value;
    document.getElementById("persistent-recording").checked = true;
    formationTest.renderer.setLayers({
      tethers: true,
      tree: false,
      cloud: false,
    });
    document.getElementById("layer-tethers").checked = true;
    document.getElementById("tab-view").click();
  });
  const makeScene = (count = 4, task = "tandem") => ({
    name: `${task} / ${count} agents`,
    task,
    size: [40, 40],
    environment: { flight: true, downward_gravity: 0 },
    physics: { dt: 0.1, substeps: 2 },
    formation_distance: 8,
    formation_pairs: [
      { a: 1, b: 0, distance: 8 },
      { a: 0, b: 2, distance: 4 },
      { a: 0, b: 3, distance: 1 },
    ].filter(({ a, b }) => a < count && b < count),
    rewards: { formation: 0 },
    bodies: [
      [16, 16],
      [24, 16],
      [16, 24],
      [24, 24],
    ]
      .slice(0, count)
      .map((position, i) => ({
        controlled: true,
        position,
        velocity: i === 2 ? [1, 0] : [0, 0],
        drag: 0,
      })),
  });
  const load = async (scene) => {
    await page.evaluate((scene) => formationTest.loadScene(scene), scene);
    await page.waitForFunction(
      (name) =>
        formationTest.ready &&
        formationTest.renderer.config.name === name &&
        formationTest.frame?.tick === 0,
      scene.name,
      { timeout: 60000 },
    );
  };
  const inspect = () =>
    page.evaluate(() => {
      const { renderer: r } = formationTest;
      const overlay = r.formationOverlay;
      const geometry = overlay.geometry;
      r.renderer.render(r.world, r.camera);
      return {
        pairs: overlay.pairs,
        positions: geometry
          ? Array.from(geometry.attributes.position.array)
          : [],
        colors: geometry ? Array.from(geometry.attributes.color.array) : [],
        scores: Array.from(overlay.scores),
        visible: overlay.group.visible,
        legendHidden: document.getElementById("formation-legend").hidden,
        legendBackground: document.querySelector(".formation-quality-ramp")
          .style.background,
        state: Array.from(r.state),
        bodyCount: r.info[1],
      };
    });
  function assertEndpoints(result) {
    for (const [i, { a, b }] of result.pairs.entries()) {
      assert.deepEqual(result.positions.slice(i * 6, i * 6 + 2), [
        result.state[8 + a],
        result.state[8 + result.bodyCount + a],
      ]);
      assert.deepEqual(result.positions.slice(i * 6 + 3, i * 6 + 5), [
        result.state[8 + b],
        result.state[8 + result.bodyCount + b],
      ]);
    }
  }
  for (const count of [1, 2, 3, 4]) {
    await load(makeScene(count));
    const result = await inspect();
    assert.equal(result.pairs.length, (count * (count - 1)) / 2);
    assert.equal(result.visible, count > 1);
    assert.equal(result.legendHidden, count < 2);
    assertEndpoints(result);
  }
  const initial = await inspect();
  assert.equal(initial.scores[0], 1);
  assert.equal(initial.scores[1], 0.5);
  assert.ok(initial.scores[2] < 0.1);
  assert.match(initial.legendBackground, /linear-gradient/);
  await page.evaluate(() => {
    const o = formationTest.renderer.formationOverlay;
    window.savedFormationResources = {
      geometry: o.geometry,
      material: o.material,
      position: o.geometry.attributes.position,
      color: o.geometry.attributes.color,
      distance: o.geometry.attributes.lineDistance,
    };
    formationTest.worker.postMessage({
      type: "manual",
      action: new Float32Array(formationTest.frame.action.length),
      frames: 5,
    });
  });
  await page.waitForFunction(() => formationTest.frame.tick === 5);
  const moved = await inspect();
  assertEndpoints(moved);
  assert.notDeepEqual(moved.positions, initial.positions);
  assert.notDeepEqual(moved.colors, initial.colors);
  await page.waitForFunction(() => formationTest.replay.recording.length >= 6);
  await page.evaluate(() => formationTest.replay.playback.seek(0));
  const historical = await inspect();
  assertEndpoints(historical);
  assert.deepEqual(historical.positions, initial.positions);
  assert.deepEqual(historical.colors, initial.colors);
  await page.evaluate(() => document.getElementById("motion-live").click());
  assert.deepEqual((await inspect()).positions, moved.positions);

  // View and style changes preserve the score mapping and the allocated buffers.
  for (const style of ["futuristic", "steampunk"]) {
    await page.evaluate(
      (style) => formationTest.renderer.setStyle(style),
      style,
    );
    for (const top of [false, true]) {
      await page.evaluate((top) => {
        const r = formationTest.renderer;
        if (r.setViewPreset) r.setViewPreset(top);
        else {
          r.top = top;
          r.resize();
        }
        r.renderer.render(r.world, r.camera);
      }, top);
      const current = await inspect();
      assertEndpoints(current);
      assert.deepEqual(current.colors, moved.colors);
      const projected = await page.evaluate(() => {
        const r = formationTest.renderer;
        r.coordinateFrame.updateMatrixWorld(true);
        return (
          r.formationOverlay.lines.parent === r.formationOverlay.group &&
          r.formationOverlay.group.parent === r.overlays
        );
      });
      assert.equal(projected, true);
      await page.screenshot({
        path: `/tmp/formation-lines-${style}-${top ? "overhead" : "side"}.png`,
      });
    }
  }
  assert.equal(
    await page.evaluate(() => {
      const o = formationTest.renderer.formationOverlay,
        s = savedFormationResources;
      return (
        o.geometry === s.geometry &&
        o.material === s.material &&
        o.geometry.attributes.position === s.position &&
        o.geometry.attributes.color === s.color &&
        o.geometry.attributes.lineDistance === s.distance
      );
    }),
    true,
  );
  await page.evaluate(() => {
    const toggle = document.getElementById("layer-tethers");
    toggle.checked = false;
    toggle.dispatchEvent(new Event("change"));
  });
  assert.equal((await inspect()).visible, false);
  assert.equal((await inspect()).legendHidden, true);
  await page.evaluate(() => {
    const toggle = document.getElementById("layer-tethers");
    toggle.checked = true;
    toggle.dispatchEvent(new Event("change"));
    window.formationDisposals = 0;
    savedFormationResources.geometry.addEventListener(
      "dispose",
      () => ++formationDisposals,
    );
    savedFormationResources.material.addEventListener(
      "dispose",
      () => ++formationDisposals,
    );
    const draft = structuredClone(formationTest.renderer.config);
    draft.formation_pairs[0].distance = 3;
    formationTest.editor.setDraftScene(draft);
  });
  assert.deepEqual(
    (await inspect()).colors,
    moved.colors,
    "draft pair edits must not change the active overlay",
  );
  await load(makeScene(2));
  assert.equal(await page.evaluate(() => formationDisposals), 2);
  assert.equal((await inspect()).pairs.length, 1);
  const edited = makeScene(2);
  edited.name = "Applied pair target";
  edited.formation_pairs[0].distance = 4;
  await load(edited);
  assert.equal((await inspect()).scores[0], 0.5);
  await load(makeScene(2, "navigation"));
  assert.equal((await inspect()).visible, false);
  assert.equal((await inspect()).legendHidden, true);
  assert.deepEqual(errors, []);
  console.log(
    "Formation lines: pair counts, colors, live/replay, styles/views, legend, resource reuse and scene changes passed",
  );
} finally {
  await browser.close();
}
