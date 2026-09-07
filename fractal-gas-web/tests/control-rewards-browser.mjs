// Verify staged reward edits against the real Lab worker and WebAssembly engine.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    window.labMessages = [];
    const WorkerBase = window.Worker;
    window.Worker = class extends WorkerBase {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", ({ data }) => {
          if (["reward-state", "ready"].includes(data.type))
            window.labMessages.push(structuredClone(data));
        });
      }
      postMessage(data, ...args) {
        if (["init", "reward-state"].includes(data.type))
          window.labMessages.push(structuredClone({ ...data, outgoing: true }));
        return super.postMessage(data, ...args);
      }
    };
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8765/lab/");
  await page.waitForFunction(
    () => document.querySelector("#tick").textContent === "TICK 000012",
  );
  assert.equal(await page.locator("#reward-settings").getAttribute("open"), "");
  const diversity = page.locator("#lab-reward-distance_coef");
  const reward = page.locator("#lab-reward-reward_coef");
  const movement = page.locator("#lab-reward-distance_squared");
  const rockTravel = page.locator("#lab-reward-hooked_rock_distance");
  const apply = page.getByRole("button", {
    name: "Apply settings",
    exact: true,
  });
  assert.equal(await movement.inputValue(), "1");
  const initCount = () =>
    page.evaluate(() => labMessages.filter((m) => m.type === "init").length);
  async function exportedMetadata() {
    const pending = page.waitForEvent("download");
    await page.locator("#export-run").click();
    const file = await pending;
    const bytes = await readFile(await file.path());
    return JSON.parse(
      bytes[0] === 123
        ? bytes.toString()
        : bytes.subarray(12, 12 + bytes.readUInt32LE(8)).toString(),
    );
  }
  const before = await initCount();
  await diversity.fill("2.5");
  await diversity.press("Tab");
  assert.equal(
    await page
      .getByRole("slider", {
        name: "Diversity coefficient slider",
        exact: true,
      })
      .inputValue(),
    "2.5",
  );
  await reward.fill("3");
  await movement.fill("2");
  assert.equal(await rockTravel.inputValue(), "1");
  await rockTravel.fill("4");
  assert.match(
    await page.locator("#reward-settings-status").textContent(),
    /Changes not applied/,
  );
  assert.equal(
    await initCount(),
    before,
    "draft edits must not reset the world",
  );
  const draftExport = await exportedMetadata();
  assert.equal(draftExport.settings.distance_coef, 1);
  assert.equal(draftExport.settings.reward_coef, 1);
  assert.equal(draftExport.scene.rewards?.distance_squared ?? 1, 1);
  await reward.fill("11");
  await apply.click();
  assert.equal(
    await initCount(),
    before,
    "invalid coefficients must not apply",
  );
  await reward.fill("3");
  await apply.click();
  await page.waitForFunction(
    (n) => labMessages.filter((m) => m.type === "ready").length > n,
    before,
  );
  let messages = await page.evaluate(() => labMessages);
  const snapshot = messages.findLast(
    (m) => m.type === "reward-state" && !m.outgoing,
  );
  const init = messages.findLast((m) => m.type === "init");
  assert.deepEqual(init.continuationRows, snapshot.rows);
  assert.equal(init.settings.distance_coef, 2.5);
  assert.equal(init.settings.reward_coef, 3);
  assert.equal(init.scene.rewards.distance_squared, 2);
  assert.equal(init.scene.rewards.hooked_rock_distance, 4);
  assert.match(
    await page.locator("#reward-settings-status").textContent(),
    /^Settings applied/,
  );
  assert.equal(snapshot.running, false);
  assert.equal(
    await page.locator("#record-count").textContent(),
    "No decisions recorded",
  );
  await page.locator("#run").click();
  await page
    .getByRole("slider", { name: "Distance travelled² slider", exact: true })
    .press("Home");
  assert.equal(await movement.inputValue(), "0");
  assert.match(
    await page.locator("#reward-settings-status").textContent(),
    /Changes not applied/,
  );
  await page
    .getByRole("slider", { name: "Hooked rock travel slider", exact: true })
    .press("Home");
  assert.equal(await rockTravel.inputValue(), "0");
  const second = await initCount();
  await apply.click();
  await page.waitForFunction(
    (n) => labMessages.filter((m) => m.type === "ready").length > n,
    second,
  );
  await page.waitForFunction(() =>
    document.querySelector("#run").textContent.includes("Pause"),
  );
  messages = await page.evaluate(() => labMessages);
  assert.equal(
    messages.findLast((m) => m.type === "reward-state" && !m.outgoing).running,
    true,
  );
  assert.equal(
    messages.findLast((m) => m.type === "init").scene.rewards.distance_squared,
    0,
  );
  await page.locator("#run").click();
  const appliedExport = await exportedMetadata();
  assert.equal(appliedExport.settings.distance_coef, 2.5);
  assert.equal(appliedExport.settings.reward_coef, 3);
  assert.equal(appliedExport.scene.rewards.distance_squared, 0);
  assert.equal(appliedExport.scene.rewards.hooked_rock_distance, 0);
  await page
    .getByRole("button", { name: "Reset defaults", exact: true })
    .click();
  assert.equal(await movement.inputValue(), "1");
  assert.match(
    await page.locator("#reward-settings-status").textContent(),
    /Changes not applied/,
  );
  assert.equal(await diversity.inputValue(), "1");
  assert.equal(
    await initCount(),
    second + 1,
    "reset defaults must remain staged",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await diversity.scrollIntoViewIfNeeded();
  assert.ok(await diversity.isVisible());
  assert.ok(
    await page
      .locator("#reward-terms")
      .evaluate((el) => el.scrollWidth <= el.clientWidth),
  );
  await page.screenshot({ path: "/tmp/lab-reward-controls-mobile.png" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await diversity.scrollIntoViewIfNeeded();
  await page.screenshot({ path: "/tmp/lab-reward-controls.png" });
  assert.deepEqual(errors, []);
  console.log(
    "Lab reward controls: staged edits, validation, paused/running continuation, defaults and responsive layout passed",
  );
} finally {
  await browser.close();
}
