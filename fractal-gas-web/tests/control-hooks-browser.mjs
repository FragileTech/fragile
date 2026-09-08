import { chromium } from "playwright";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.addInitScript(() => {
    window.hookInits = [];
    const Base = window.Worker;
    window.Worker = class extends Base {
      postMessage(data, ...args) {
        if (data.type === "init") hookInits.push(structuredClone(data));
        return super.postMessage(data, ...args);
      }
    };
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/");
  const ready = () =>
    page.waitForFunction(() => !document.getElementById("run").disabled);
  const count = () => page.evaluate(() => hookInits.length);
  const latest = () => page.evaluate(() => hookInits.at(-1));
  const apply = async () => {
    const n = await count();
    await page.locator("#apply-rocks").click();
    await page.waitForFunction((n) => hookInits.length > n, n);
    await ready();
  };
  await ready();
  await page.locator("#scenario").selectOption("mining");
  await ready();
  assert.equal(await page.locator("#hook-stiffness").inputValue(), "35");
  assert.equal(await page.locator("#rock-weight-slider").inputValue(), "0");
  assert.equal(await page.locator("#rock-weight-value").textContent(), "1×");
  await page.locator("#rock-size").fill("0.1");
  await page.locator("#rock-weight-slider").press("Home");
  assert.equal(await page.locator("#rock-weight-value").textContent(), "0.01×");
  await page.locator("#hook-stiffness").fill("3000");
  await apply();
  let init = await latest();
  assert.equal(init.scene.rock_options.scale, 0.1);
  assert.equal(init.scene.rock_options.weight, 0.01);
  assert.equal(init.scene.bodies.find((b) => b.cargo).mass, 0.24);
  assert.ok(init.scene.tethers.every((t) => t.stiffness === 3000));
  assert.equal(await page.locator("#tick").textContent(), "TICK 000000");
  await page.locator("#scenario").selectOption("harvest");
  await ready();
  assert.equal(await page.locator("#rock-weight-slider").inputValue(), "0");
  assert.equal(await page.locator("#rock-weight-value").textContent(), "1×");
  await page.locator("#scenario").selectOption("mining");
  await ready();
  assert.equal(await page.locator("#hook-stiffness").inputValue(), "3000");
  assert.equal(await page.locator("#rock-size").inputValue(), "0.1");
  assert.equal(await page.locator("#rock-weight-slider").inputValue(), "-2");
  assert.equal(await page.locator("#rock-weight-value").textContent(), "0.01×");
  await page.locator("#hook-stiffness-slider").press("Home");
  assert.equal(await page.locator("#hook-stiffness").inputValue(), "0");
  await apply();
  assert.ok((await latest()).scene.tethers.every((t) => t.stiffness === 0));
  await page.locator("#hook-stiffness-slider").press("End");
  assert.equal(await page.locator("#hook-stiffness").inputValue(), "1000000");
  await page.locator("#rock-weight-slider").press("End");
  assert.equal(await page.locator("#rock-weight-value").textContent(), "10×");
  await apply();
  init = await latest();
  assert.equal(init.scene.rock_options.weight, 10);
  assert.equal(init.scene.bodies.find((b) => b.cargo).mass, 240);
  assert.ok(init.scene.tethers.every((t) => t.stiffness === 1000000));
  const n = await count();
  await page.locator("#rock-size").fill("0.09");
  await page.locator("#apply-rocks").click();
  assert.equal(await count(), n);
  await page.locator("#rock-size").fill("0.1");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#hook-stiffness").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "/tmp/mining-hook-controls.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log(
    "Mining controls: rock size/weight, spring-constant slider, validation, paused reset and persistence passed",
  );
} finally {
  await browser.close();
}
