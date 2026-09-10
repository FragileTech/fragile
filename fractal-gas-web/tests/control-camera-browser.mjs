// Run against a served web/ directory with CONTROL_TEST_URL=http://127.0.0.1:8088/lab/.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    serviceWorkers: "block",
    viewport: { width: 1536, height: 2400 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("installStyleControls();", "cancelAnimationFrame(renderer.frame); installStyleControls();")}\nwindow.cameraTest = { renderer, editor, replay, workspace, transitions, ready: () => ready };`,
    });
  });
  await page.addInitScript(() =>
    localStorage.setItem("lab.workspace.onboarded", "true"),
  );
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/");
  await page
    .waitForFunction(() => window.cameraTest?.renderer.state, null, {
      timeout: 20000,
    })
    .catch(async (e) => {
      console.error(
        await page.evaluate(() => ({
          test: !!window.cameraTest,
          status: document.querySelector("#status")?.textContent,
          backend: document.querySelector("#backend")?.textContent,
        })),
      );
      throw e;
    });
  console.log("Lab ready");
  // Harvest currently starts in flight mode. Exercise the ordinary ground
  // presentation explicitly before switching to the rocket's flight presets.
  if (await page.evaluate(() => cameraTest.renderer.flightMode)) {
    await page.locator("#world-physics").evaluate((e) => (e.open = true));
    await page.locator("#flight-mode").click();
    await page.locator("#apply-configuration").click();
    await page.waitForFunction(
      () =>
        cameraTest.ready() &&
        !cameraTest.transitions.busy &&
        cameraTest.renderer.state &&
        !cameraTest.renderer.flightMode,
    );
  }
  const state = () =>
    page.evaluate(() => {
      const r = cameraTest.renderer;
      return {
        center: [...r.viewCenter],
        zoom: r.zoom,
        follow: r.followBody,
        dragging: !!r.cameraGesture,
        orbit: { ...r.orbit },
      };
    });
  const assertArenaVisible = async () => {
    const frame = await page.evaluate(async () => {
      const { Vector3 } = await import("./vendor/three.module.js");
      const r = cameraTest.renderer;
      r.resize();
      r.camera.updateMatrixWorld(true);
      const side = r.flightMode && !r.top;
      const points = [
        [r.arena.minX, r.arena.minY],
        [r.arena.minX, r.arena.maxY],
        [r.arena.maxX, r.arena.minY],
        [r.arena.maxX, r.arena.maxY],
      ];
      const projected = points.map(([x, y]) =>
        new Vector3(x, side ? 0 : y, side ? y : 0).project(r.camera),
      );
      return {
        zoom: r.zoom,
        center: [...r.viewCenter],
        arenaCenter: [...r.arena.center],
        maxAbs: Math.max(
          ...projected.flatMap(({ x, y }) => [Math.abs(x), Math.abs(y)]),
        ),
      };
    });
    assert.equal(frame.zoom, 1);
    assert.deepEqual(frame.center, frame.arenaCenter);
    assert(
      frame.maxAbs <= 1 / 1.05 + 1e-6,
      `Arena boundary is outside the default frame: ${frame.maxAbs}`,
    );
  };
  await assertArenaVisible();
  const box = await page.locator("#world").boundingBox();
  const x = box.x + box.width / 2,
    y = Math.min(box.y + box.height / 2, page.viewportSize().height - 80);
  await page.evaluate(() => {
    window.cameraClicks = 0;
    document
      .querySelector("#world")
      .addEventListener("worldclick", () => cameraClicks++);
  });
  for (const top of [false, true]) {
    console.log("Testing orientation", top);
    await page.evaluate((top) => {
      cameraTest.renderer.setViewPreset(top);
    }, top);
    await page.mouse.move(x, y);
    await page.mouse.wheel(0, -900);
    await page.waitForTimeout(100);
    const before = await state();
    await page.mouse.down();
    await page.mouse.move(x + 90, y + 40, { steps: 8 });
    await page.mouse.up();
    const after = await state();
    assert.notDeepEqual(after.center, before.center);
    assert.equal(after.zoom, before.zoom);
    assert.equal(await page.evaluate(() => cameraClicks), 0);
    // Reapplying simulation state must not re-center a manually panned camera.
    await page.evaluate(() => {
      const r = cameraTest.renderer;
      r.update(r.state, r.action);
    });
    assert.deepEqual((await state()).center, after.center);
  }
  await page.mouse.click(x, y);
  assert.equal(await page.evaluate(() => cameraClicks), 1);
  const rightDrag = async (dx = 70, dy = -35) => {
    const rect = await page.locator("#world").boundingBox();
    const startX = rect.x + rect.width / 2,
      startY = rect.y + rect.height / 2;
    await page.mouse.move(startX, startY);
    await page.mouse.down({ button: "right" });
    await page.mouse.move(startX + dx, startY + dy, { steps: 5 });
    await page.mouse.up({ button: "right" });
  };
  const world = () =>
    page.evaluate(() => ({
      bytes: Array.from(new Uint8Array(cameraTest.renderer.state.buffer)),
      scene: JSON.stringify(cameraTest.renderer.config),
      draft: JSON.stringify(cameraTest.workspace.draft),
      clicks: cameraClicks,
    }));
  const captureStyles = async (name) => {
    const before = await state(),
      beforeWorld = await world();
    await page.setViewportSize({ width: 1440, height: 1000 });
    for (const style of ["steampunk", "futuristic"]) {
      await page.evaluate(async (style) => {
        const { labStyle } = await import("./visual-style.js");
        await labStyle.change(style);
        const r = cameraTest.renderer;
        r.resize();
        r.animate(performance.now());
        cancelAnimationFrame(r.frame);
      }, style);
      assert.deepEqual((await state()).orbit, before.orbit);
      assert.deepEqual(await world(), beforeWorld);
      await page
        .locator("#world")
        .screenshot({ path: `/tmp/lab-orbit-${name}-${style}.png` });
    }
    await page.setViewportSize({ width: 1536, height: 2400 });
    await page.evaluate(() => cameraTest.renderer.resize());
  };
  for (const top of [false, true]) {
    await page.evaluate((top) => cameraTest.renderer.setViewPreset(top), top);
    const before = await state(),
      beforeWorld = await world();
    await rightDrag(1, 1);
    assert.deepEqual(await state(), before, "Tiny movements must not rotate");
    await rightDrag();
    const rotated = await state();
    assert.notDeepEqual(rotated.orbit, before.orbit);
    assert.deepEqual(rotated.center, before.center);
    assert.equal(rotated.zoom, before.zoom);
    assert.deepEqual(await world(), beforeWorld);
    const rect = await page.locator("#world").boundingBox();
    await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height / 2);
    await page.mouse.wheel(0, 100);
    await page.waitForFunction(
      (zoom) => cameraTest.renderer.zoom !== zoom,
      rotated.zoom,
    );
    assert.deepEqual((await state()).orbit, rotated.orbit);
    await page.evaluate(() => {
      const { renderer: r, replay } = cameraTest;
      r.update(r.state, r.action);
      replay.playback.seek(0);
    });
    assert.deepEqual((await state()).orbit, rotated.orbit);
    await page.setViewportSize({ width: 1280, height: 1700 });
    await page.evaluate(() => cameraTest.renderer.resize());
    assert.deepEqual((await state()).orbit, rotated.orbit);
    await page.setViewportSize({ width: 1536, height: 2400 });
    await page.evaluate(() => cameraTest.renderer.resize());
    assert.deepEqual((await state()).orbit, rotated.orbit);
  }
  // Check projection correctness for several moves without rendering a frame.
  await page.mouse.move(x, y);
  await page.mouse.down();
  const error = await page.evaluate(
    ({ x, y }) => {
      const r = cameraTest.renderer,
        c = r.canvas,
        id = r.cameraGesture.id;
      const anchor = r.worldPoint({ clientX: x, clientY: y });
      for (let i = 1; i <= 5; i++)
        c.dispatchEvent(
          new PointerEvent("pointermove", {
            pointerId: id,
            clientX: x + i * 10,
            clientY: y + i * 5,
          }),
        );
      const final = r.worldPoint({ clientX: x + 50, clientY: y + 25 });
      return Math.hypot(...final.map((v, i) => v - anchor[i]));
    },
    { x, y },
  );
  assert(error < 1e-8, `Pan anchor drift: ${error}`);
  await page.mouse.up();
  await page.locator("#focus").click();
  assert.notEqual((await state()).follow, null);
  await rightDrag();
  const followed = await state();
  await page.evaluate(() => {
    const r = cameraTest.renderer;
    const original = r.state,
      moved = original.slice();
    moved[8 + r.followBody] += 1;
    r.update(moved, r.action);
    window.followedCenter = [...r.viewCenter];
    r.update(original, r.action);
  });
  assert(
    Math.abs(
      (await page.evaluate(() => followedCenter))[0] - followed.center[0] - 1,
    ) < 1e-5,
  );
  assert.deepEqual((await state()).orbit, followed.orbit);
  assert.equal((await state()).follow, followed.follow);
  await page.locator("#focus").click();
  await assertArenaVisible();
  assert.equal(await page.evaluate(() => cameraTest.renderer.top), false);
  await page.locator("#focus").click();
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 20, y + 20);
  await page.mouse.up();
  assert.equal((await state()).follow, null);
  assert.equal(await page.locator("#focus").textContent(), "Follow agent");
  await page.locator("#reset-view").click();
  assert.equal((await state()).zoom, 1);
  assert.deepEqual(
    (await state()).center,
    await page.evaluate(() => cameraTest.renderer.arena.center),
  );
  await assertArenaVisible();
  for (const button of ["middle", "alt"]) {
    const before = await state();
    await page.mouse.move(x, y);
    if (button === "alt") await page.keyboard.down("Alt");
    await page.mouse.down({ button: button === "alt" ? "left" : button });
    await page.mouse.move(x + 30, y + 20);
    await page.mouse.up({ button: button === "alt" ? "left" : button });
    if (button === "alt") await page.keyboard.up("Alt");
    assert.notDeepEqual((await state()).center, before.center);
  }
  for (const type of ["pointercancel", "lostpointercapture"]) {
    for (const button of ["left", "right"]) {
      await page.mouse.move(x, y);
      await page.mouse.down({ button });
      await page.evaluate((type) => {
        const r = cameraTest.renderer;
        r.canvas.dispatchEvent(
          new PointerEvent(type, { pointerId: r.cameraGesture.id }),
        );
      }, type);
      assert.equal((await state()).dragging, false);
      await page.mouse.up({ button });
    }
  }
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(box.x - 15, y);
  await page.mouse.up();
  assert.equal((await state()).dragging, false);
  await page.locator("#reset-view").click();
  await page.locator("#mode-edit").click();
  const beforeEdit = await state();
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 25, y + 20);
  await page.mouse.up();
  assert.deepEqual((await state()).center, beforeEdit.center);
  assert.equal((await state()).dragging, false);
  assert.equal(await page.locator("#run").isDisabled(), true);
  const editWorld = await world();
  await rightDrag(60, -20);
  assert.deepEqual(
    await world(),
    editWorld,
    "Rotation must not edit the draft",
  );
  await page.locator("#world").scrollIntoViewIfNeeded();
  // Drag an actual body through the editor and check the committed scene.
  const body = await page.evaluate(async () => {
    const { Vector3 } = await import("./vendor/three.module.js");
    const r = cameraTest.renderer,
      i = r.controlled[0];
    const position = { x: r.state[8 + i], y: r.state[8 + r.info[1] + i] };
    r.resize();
    r.camera.updateMatrixWorld(true);
    const side = r.flightMode && !r.top;
    const p = new Vector3(
      position.x,
      side ? 0 : position.y,
      side ? position.y : 0,
    ).project(r.camera);
    const rect = r.canvas.getBoundingClientRect();
    return {
      i,
      position: [...r.config.bodies[i].position],
      x: rect.left + ((p.x + 1) * rect.width) / 2,
      y: rect.top + ((1 - p.y) * rect.height) / 2,
    };
  });

  await page.mouse.move(body.x, body.y);
  await page.mouse.down();

  await page.mouse.move(body.x + 20, body.y + 10);
  await page.mouse.up();
  await page.locator("#apply-editor").click();
  await page.waitForFunction(
    ({ i, position }) =>
      JSON.stringify(cameraTest.renderer.config.bodies[i].position) !==
      JSON.stringify(position),
    body,
  );
  assert.equal((await state()).zoom, 1);
  // Render one frame for visual inspection without a continuous software-GPU loop.
  await page.evaluate(() => {
    const r = cameraTest.renderer;
    r.renderer.render(r.world, r.camera);
  });
  await page.screenshot({ path: "/tmp/lab-camera.png" });
  await rightDrag();
  await captureStyles("ground");
  await page.locator("#scenario").selectOption("rocket");
  await page.locator("#apply-configuration").click();
  await page.waitForFunction(
    () =>
      cameraTest.ready() &&
      !cameraTest.transitions.busy &&
      cameraTest.renderer.state &&
      cameraTest.renderer.flightMode &&
      cameraTest.renderer.config.name === "Mining rocket · thinking graphs",
  );
  await assertArenaVisible();
  const flightView = await page.evaluate(() => {
    const r = cameraTest.renderer;
    r.resize();
    const center = r.worldPoint({
      clientX: r.canvas.getBoundingClientRect().left + r.canvas.clientWidth / 2,
      clientY: r.canvas.getBoundingClientRect().top + r.canvas.clientHeight / 2,
    });
    return {
      top: r.top,
      frameRotation: r.coordinateFrame.rotation.x,
      camera: [r.camera.position.x, r.camera.position.y, r.camera.position.z],
      center,
      arenaCenter: [...r.arena.center],
      label: document.querySelector("#view").textContent.trim(),
    };
  });
  assert.equal(flightView.top, false);
  assert(Math.abs(flightView.frameRotation - Math.PI / 2) < 1e-8);
  assert(flightView.camera[1] < -80, JSON.stringify(flightView));
  flightView.center.forEach((value, i) =>
    assert(Math.abs(value - flightView.arenaCenter[i]) < 1e-8),
  );
  assert.equal(flightView.label, "Side / overhead");
  const flightControl = await page.evaluate(() => ({
    checked: document.querySelector("#flight-mode").checked,
    indeterminate: document.querySelector("#flight-mode").indeterminate,
    state: document.querySelector("#flight-mode-state").textContent,
  }));
  assert.equal(flightControl.checked, true);
  assert.equal(flightControl.indeterminate, true);
  assert.equal(flightControl.state, "AUTO");
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  await page.locator("#flight-mode").click();
  await page.locator("#apply-configuration").click();
  await page.waitForFunction(
    () => cameraTest.renderer.state && !cameraTest.renderer.flightMode,
  );
  assert.equal(await page.locator("#flight-mode-state").textContent(), "OFF");
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  await page.locator("#flight-mode").click();
  await page.locator("#apply-configuration").click();
  await page.waitForFunction(
    () => cameraTest.renderer.state && cameraTest.renderer.flightMode,
  );
  await page.locator("#view").click();
  await assertArenaVisible();
  const overheadView = await page.evaluate(() => {
    const r = cameraTest.renderer;
    return {
      top: r.top,
      frameRotation: r.coordinateFrame.rotation.x,
      camera: [r.camera.position.x, r.camera.position.y, r.camera.position.z],
    };
  });
  assert.equal(overheadView.top, true);
  assert(Math.abs(overheadView.frameRotation) < 1e-8);
  assert(overheadView.camera[2] > 80);
  await rightDrag();
  assert.equal(
    await page.evaluate(() => cameraTest.renderer.coordinateFrame.rotation.x),
    0,
  );
  await page.locator("#view").click();
  assert.equal((await state()).orbit.pitch, 0);
  const flightWorld = await world(),
    flightCenter = (await state()).center;
  // Rotate to an edge-on flight view; picking is disabled but rotation remains usable.
  await rightDrag(-Math.PI / 2 / 0.008, 0);
  const edge = await page.evaluate(() => {
    const r = cameraTest.renderer,
      rect = r.canvas.getBoundingClientRect();
    return r.worldPoint({
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
    });
  });
  assert.equal(edge, null);
  await page.mouse.click(x, y);
  assert.deepEqual(await world(), flightWorld);
  await rightDrag(60, 20);
  assert.deepEqual((await state()).center, flightCenter);
  assert.deepEqual(await world(), flightWorld);
  const recovered = await page.evaluate(() => {
    const r = cameraTest.renderer,
      rect = r.canvas.getBoundingClientRect();
    return r.worldPoint({
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
    });
  });
  recovered.forEach((value, i) =>
    assert(Math.abs(value - flightCenter[i]) < 1e-8),
  );
  await page.locator("#reset-view").click();
  assert.equal((await state()).orbit.pitch, 0);
  await assertArenaVisible();
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  const tiltLimit = await page.evaluate(
    ({ x, y }) => {
      const r = cameraTest.renderer,
        pointerId = r.cameraGesture.id;
      for (const offset of [1000, 990]) {
        r.canvas.dispatchEvent(
          new PointerEvent("pointermove", {
            pointerId,
            clientX: x,
            clientY: y + offset,
          }),
        );
        if (offset === 1000) window.maximumTilt = r.orbit.pitch;
      }
      const result = { maximum: maximumTilt, reversed: r.orbit.pitch };
      r.clearCameraGesture();
      return result;
    },
    { x, y },
  );
  assert(Math.abs(tiltLimit.maximum - (80 * Math.PI) / 180) < 1e-8);
  assert(
    tiltLimit.reversed < tiltLimit.maximum,
    "Reversing at the tilt limit responds immediately",
  );
  await page.mouse.up({ button: "right" });
  await page.locator("#reset-view").click();
  await rightDrag(-40, 60);
  await captureStyles("flight");
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  await page.evaluate(() => cameraTest.renderer.dispose());
  assert.equal((await state()).dragging, false);
  await page.mouse.up({ button: "right" });
  assert.deepEqual(errors, []);
  console.log("Camera browser regression passed");
} finally {
  await browser.close();
}
