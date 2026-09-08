import { chromium } from "playwright";
import assert from "node:assert/strict";
import { prepareWorkspace, applyDraft } from "./helpers/workspace-ui.mjs";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await prepareWorkspace(page);
  await page.addInitScript(() => {
    window.hookUpdates = [];
    window.hookReady = [];
    const Base = window.Worker;
    window.Worker = class extends Base {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", ({ data }) => {
          if (data.type === "rewards-updated")
            hookUpdates.push(structuredClone(data));
          if (data.type === "ready") hookReady.push(structuredClone(data));
        });
      }
    };
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/");
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await page.locator("#tab-setup").click();
  await page.locator("#scenario").selectOption("harvest");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  assert.equal(await page.evaluate(() => hookReady.at(-1).info[1]), 7);
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  assert.equal(await page.locator("#hook-mass").inputValue(), "0.25");
  const tick = await page.locator("#tick").textContent();
  await page.locator("#hook-mass").fill("1");
  await page.locator("#apply-hook-mass").click();
  await page.waitForFunction(() => hookUpdates.length === 1);
  assert.equal(await page.evaluate(() => hookUpdates[0].scene.hook_mass), 1);
  assert.equal(await page.locator("#tick").textContent(), tick);
  assert.equal(await page.locator("#lab-reward-delivery").isVisible(), false);
  await page.locator("#clean").click();
  await page.locator("#focus").click();
  await page.mouse.move(900, 700);
  await page.waitForTimeout(4000);
  await page.screenshot({ path: "/tmp/harvest-hook-side.png" });
  await page.locator("#view").click();
  await page.mouse.move(900, 700);
  await page.waitForTimeout(600);
  await page.screenshot({ path: "/tmp/harvest-hook-overhead.png" });
  assert.deepEqual(errors, []);
  console.log(
    "Hook mass live update, layout, reward controls and both camera modes passed",
  );
} finally {
  await browser.close();
}
