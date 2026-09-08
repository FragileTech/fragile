// Serve web/ then run CONTROL_TEST_URL=http://127.0.0.1:8088/lab/ node this-file.
import assert from "node:assert/strict";
import { chromium } from "playwright";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
    serviceWorkers: "block",
    reducedMotion: "reduce",
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    window.coi = { shouldRegister: () => false };
  });
  await page.route("**/lab/asset-gallery.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("Math.min(devicePixelRatio, 2)", "0.5")}\nwindow.actionQA={get ready(){return !!binding;},get commands(){return motion.commands;},get time(){return previewTime;},get effect(){return actionEffects;}};`,
    });
  });
  await page.goto(`${base}asset-gallery.html`);
  await page.waitForFunction(() => window.actionQA?.ready);
  assert.equal(
    await page
      .getByRole("checkbox", { name: "Animations", exact: true })
      .isChecked(),
    false,
  );
  const guides = page.getByRole("checkbox", {
    name: "Action guides",
    exact: true,
  });
  assert.equal(await guides.isChecked(), false);
  const command = () => page.evaluate(() => structuredClone(actionQA.commands));
  const set = async (id, value) =>
    page.locator(`#preview-${id}`).evaluate((input, value) => {
      input.value = value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }, value);
  await set("thrust", 0.8);
  assert.ok(
    Math.abs((await command()).thrust - 0.8) < 1e-6,
    "static thrust updates with animation disabled",
  );
  await set("torque", -0.4);
  assert.ok((await command()).torque < 0);
  await guides.check();
  assert.ok(
    (await page.locator("#action-guide-readout").textContent()).length > 0,
  );
  for (const view of ["Side", "Top"]) {
    await page.getByRole("button", { name: view, exact: true }).click();
    assert.equal(await page.locator("#preview-thrust").inputValue(), "0.8");
    assert.ok((await command()).torque < 0);
  }
  await page.getByRole("button", { name: "Neutral", exact: true }).click();
  assert.equal((await command()).thrust, 0);
  await page.getByRole("button", { name: "Max", exact: true }).click();
  assert.equal((await command()).thrust, 1);
  await page.locator("#vehicle").selectOption("kart");
  await set("throttle", -0.7);
  await set("steering", 0.5);
  await set("brake", 0.2);
  assert.ok((await command()).throttle < 0);
  assert.ok((await command()).steering > 0);
  assert.ok((await command()).brake > 0);
  await page.locator("#vehicle").selectOption("drone");
  await set("force_x", -0.4);
  await set("force_y", 0.75);
  await set("torque", -0.2);
  assert.ok((await command()).forceX < 0);
  assert.ok((await command()).forceY > 0);
  await page.locator("#vehicle").selectOption("rocket");
  await page.locator("#actuator-variant").selectOption("thruster_tug");
  await set("thruster_0", 0.8);
  await set("thruster_1", 0);
  await set("thruster_2", -0.6);
  const thrusters = (await command()).thrusters;
  assert.equal(thrusters.length, 3);
  assert.ok(
    thrusters[0].value > 0 &&
      thrusters[1].value === 0 &&
      thrusters[2].value < 0,
  );
  const before = await command();
  await page.getByRole("checkbox", { name: "Animations", exact: true }).check();
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .click();
  await page.waitForFunction(() => actionQA.time > 0.1);
  assert.deepEqual(
    await command(),
    before,
    "playback never invents changing actuator values",
  );
  await page
    .getByRole("button", { name: "Pause animation", exact: true })
    .click();
  await page.getByRole("button", { name: "Perspective", exact: true }).click();
  await page.screenshot({ path: "/tmp/lab-action-guides-workshop.png" });
  await page.locator('input[name="visual-style"][value="steampunk"]').check();
  await page.waitForFunction(
    () => document.documentElement.dataset.visualStyle === "steampunk",
  );
  assert.deepEqual(
    await command(),
    before,
    "style changes preserve actuator values",
  );
  assert.equal(await guides.isChecked(), true);
  assert.deepEqual(errors, []);
  console.log(
    "PASS: static signed commands, catalog actuator controls, independent thrusters, presets, Side/Top preservation, playback truth, guide preference, style persistence. /tmp/lab-action-guides-workshop.png",
  );
} finally {
  await browser.close();
}
