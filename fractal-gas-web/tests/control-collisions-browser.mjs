import { chromium } from "playwright";
import assert from "node:assert/strict";
import { prepareWorkspace, applyDraft } from "./helpers/workspace-ui.mjs";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await prepareWorkspace(page);
  await page.addInitScript(() => {
    window.collisionFrames = [];
    const Base = window.Worker;
    window.Worker = class extends Base {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", ({ data }) => {
          if (data.type === "frame")
            collisionFrames.push({
              tick: data.tick,
              finite: Array.from(data.state || []).every(Number.isFinite),
            });
        });
      }
    };
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/");
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await page.locator("#tab-setup").click();
  await page.locator("#scenario").selectOption("harvest");
  await applyDraft(page);
  await page.locator("#tab-controller").click();
  await page.locator("#algorithm").selectOption("fmc");
  for (const [id, value] of Object.entries({
    walkers: 8,
    horizon: 2,
    frames: 12,
  })) {
    await page.locator(`#${id}`).fill(String(value));
    await page.locator(`#${id}`).press("Tab");
  }
  await applyDraft(page);
  for (const flight of [true, false]) {
    console.log(`Checking ${flight ? "flight" : "regular"} harvesting motion`);
    await page.locator("#tab-setup").click();
    await page.locator("#world-physics").evaluate((e) => (e.open = true));
    await page.locator("#flight-mode").setChecked(flight);
    await applyDraft(page);
    const before = await page.locator("#tick").textContent();
    await page.locator("#step").click();
    await page.waitForFunction(
      (before) => document.getElementById("tick").textContent !== before,
      before,
    );
    assert.ok(
      await page.evaluate(
        () =>
          collisionFrames.length > 0 && collisionFrames.every((f) => f.finite),
      ),
    );
    await page.locator("#world").click({ position: { x: 20, y: 20 } });
    await page.screenshot({
      timeout: 120000,
      animations: "disabled",
      path: `/tmp/collision-${flight ? "flight" : "regular"}.png`,
    });
  }
  assert.deepEqual(errors, []);
  console.log(
    "Flight and regular harvesting advance with finite state and render successfully",
  );
} finally {
  await browser.close();
}
