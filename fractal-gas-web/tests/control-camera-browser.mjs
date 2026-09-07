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
    viewport: { width: 1536, height: 1100 },
  });
  page.on("pageerror", (e) => console.error(e.message));
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("installStyleControls();", "cancelAnimationFrame(renderer.frame); installStyleControls();")}\nwindow.cameraTest = { renderer, editor };`,
    });
  });
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
  const state = () =>
    page.evaluate(() => {
      const r = cameraTest.renderer;
      return {
        center: [...r.viewCenter],
        zoom: r.zoom,
        follow: r.followBody,
        dragging: !!r.panGesture,
      };
    });
  const box = await page.locator("#world").boundingBox();
  const x = box.x + box.width / 2,
    y = box.y + box.height / 2;
  await page.evaluate(() => {
    window.cameraClicks = 0;
    document
      .querySelector("#world")
      .addEventListener("worldclick", () => cameraClicks++);
  });
  for (const top of [false, true]) {
    console.log("Testing orientation", top);
    await page.evaluate((top) => {
      cameraTest.renderer.top = top;
      cameraTest.renderer.resize();
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
  // Check projection correctness for several moves without rendering a frame.
  await page.mouse.move(x, y);
  await page.mouse.down();
  const error = await page.evaluate(
    ({ x, y }) => {
      const r = cameraTest.renderer,
        c = r.canvas,
        id = r.panGesture.id;
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
    await page.evaluate(() => cameraTest.renderer.size.map((v) => v / 2)),
  );
  for (const button of ["middle", "right", "alt"]) {
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
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.evaluate((type) => {
      const r = cameraTest.renderer;
      r.canvas.dispatchEvent(
        new PointerEvent(type, { pointerId: r.panGesture.id }),
      );
    }, type);
    assert.equal((await state()).dragging, false);
    await page.mouse.up();
  }
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(box.x - 15, y);
  await page.mouse.up();
  assert.equal((await state()).dragging, false);
  await page.locator("#reset-view").click();
  await page.locator("#edit").click();
  const beforeEdit = await state();
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 25, y + 20);
  await page.mouse.up();
  assert.deepEqual((await state()).center, beforeEdit.center);
  assert.equal((await state()).dragging, false);
  await page.waitForFunction(() => !document.querySelector("#run").disabled);
  await page.locator("#world").scrollIntoViewIfNeeded();
  // Drag an actual body through the editor and check the committed scene.
  const body = await page.evaluate(async () => {
    const { Vector3 } = await import("./vendor/three.module.js");
    const r = cameraTest.renderer,
      i = r.controlled[0];
    const position = { x: r.state[8 + i], y: r.state[8 + r.info[1] + i] };
    r.resize();
    r.camera.updateMatrixWorld(true);
    const p = new Vector3(position.x, position.y, 0).project(r.camera);
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
  await page.evaluate(() => cameraTest.renderer.dispose());
  assert.equal((await state()).dragging, false);
  console.log("Camera browser regression passed");
} finally {
  await browser.close();
}
