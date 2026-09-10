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
    window.deliveryScenes = [];
    const Base = window.Worker;
    window.Worker = class extends Base {
      postMessage(data, ...args) {
        if (data.type === "init")
          deliveryScenes.push(structuredClone(data.scene));
        return super.postMessage(data, ...args);
      }
    };
    import("/lab/renderer.js").then(({ LabRenderer }) => {
      const makeStatic = LabRenderer.prototype.makeStatic;
      LabRenderer.prototype.makeStatic = function (...args) {
        const result = makeStatic.apply(this, args);
        window.deliveryRings = result.group.children.filter((c) =>
          /^(Delivery|Release) boundary$/.test(c.name),
        );
        return result;
      };
    });
  });
  console.log("Opening lab");
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/");
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  console.log("Opening setup");
  await page.locator("#tab-setup").click();
  await page.locator("#scenario").selectOption("harvest");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  console.log("Opening setup");
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  assert.equal(await page.locator("#keep-delivered-rocks").isChecked(), true);
  assert.equal(
    await page.evaluate(() => deliveryScenes.at(-1).keep_delivered_rocks),
    true,
  );
  await page.waitForFunction(() => deliveryRings?.length === 2);
  assert.deepEqual(
    await page.evaluate(() =>
      deliveryRings.map((r) => [r.name, r.userData.radius]),
    ),
    [
      ["Delivery boundary", 1.5],
      ["Release boundary", 3],
    ],
  );
  console.log("Disabling retention");
  await page.locator("#keep-delivered-rocks").uncheck();
  await applyDraft(page);
  assert.equal(
    await page.evaluate(() => deliveryScenes.at(-1).keep_delivered_rocks),
    false,
  );
  assert.equal(await page.evaluate(() => deliveryRings.length), 0);
  console.log("Opening setup");
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  console.log("Enabling retention");
  await page.locator("#keep-delivered-rocks").check();
  await applyDraft(page);
  assert.equal(await page.locator("#tick").textContent(), "TICK 000000");
  await page.locator("#clean").click();
  assert.ok(
    await page.evaluate(
      () =>
        deliveryRings.length === 2 &&
        deliveryRings.every((r) => {
          for (let p = r; p; p = p.parent) if (!p.visible) return false;
          return true;
        }),
    ),
  );
  await page.mouse.move(1000, 700);
  await page.screenshot({ path: "/tmp/retained-rocks-side.png" });
  await page.locator("#view").click();
  await page.mouse.move(1000, 700);
  await page.waitForTimeout(600);
  await page.screenshot({ path: "/tmp/retained-rocks-overhead.png" });
  console.log("Checking Collaborative mining");
  await page.evaluate(() => {
    window.deliveryRings = [];
  });
  await page.locator("#tab-setup").click();
  await page.locator("#scenario").selectOption("mining");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  assert.equal(await page.locator("#keep-delivered-rocks").isChecked(), true);
  assert.equal(
    await page.evaluate(() => deliveryScenes.at(-1).keep_delivered_rocks),
    true,
  );
  await page.waitForFunction(() => deliveryRings?.length === 2);
  assert.deepEqual(
    await page.evaluate(() =>
      deliveryRings.map((r) => [r.name, r.userData.radius]),
    ),
    [
      ["Delivery boundary", 1.5],
      ["Release boundary", 3],
    ],
  );
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  await page.locator("#keep-delivered-rocks").uncheck();
  await applyDraft(page);
  assert.equal(
    await page.evaluate(() => deliveryScenes.at(-1).keep_delivered_rocks),
    false,
  );
  assert.equal(await page.evaluate(() => deliveryRings.length), 0);
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics").evaluate((e) => (e.open = true));
  await page.locator("#keep-delivered-rocks").check();
  await applyDraft(page);
  assert.equal(
    await page.evaluate(() => deliveryScenes.at(-1).keep_delivered_rocks),
    true,
  );
  await page.waitForFunction(() => deliveryRings?.length === 2);
  assert.deepEqual(errors, []);
  console.log(
    "Retained rock toggle, restart, export configuration, and visible delivery/release boundaries passed for both mining presets",
  );
} finally {
  await browser.close();
}
