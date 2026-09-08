import assert from "node:assert/strict";
import { chromium } from "playwright";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text()) +
        "\nwindow.liftTest={workspace,transitions,getReady:()=>ready};",
    });
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8097/lab/");
  await page.waitForFunction(() => window.liftTest?.getReady(), null, {
    timeout: 60000,
  });
  console.log("Lab ready");
  await page.locator("#chooser-explore").click();
  await page.locator("#scenario").selectOption("mining");
  await page.waitForFunction(
    () => liftTest.workspace.draft.scene.name === "Collaborative mining",
  );
  if (await page.locator("#toggle-settings").isVisible())
    await page.locator("#toggle-settings").click();
  await page.locator("#tab-setup").click();
  await page.locator("#world-physics > summary").click();
  assert.match(await page.locator("#rock-lift-budget").innerText(), /32.0 N.*22.0 N.*Thrust exceeds weight/);
  assert.equal(await page.locator("#fit-rock-lift").isDisabled(), true);
  // Overload the now-liftable default to exercise the lighter-load action.
  await page.locator("#rock-weight-slider").press("End");
  await page.locator("#rock-weight-slider").dispatchEvent("change");
  assert.match(
    await page.locator("#rock-lift-budget").innerText(),
    /32.0 N.*43.2 N.*Too heavy/,
  );
  await page
    .locator("#fit-rock-lift")
    .evaluate((node) => node.scrollIntoView({ block: "start" }));
  await page.locator("#fit-rock-lift").click();
  console.log("Lighter load staged");
  const weight = await page.evaluate(
    () => liftTest.workspace.draft.scene.rock_options.weight,
  );
  assert.ok(weight >= 1 && weight < 2);
  assert.match(
    await page.locator("#rock-lift-budget").innerText(),
    /Thrust exceeds weight/,
  );
  assert.equal(await page.evaluate(() => liftTest.workspace.dirty), true);
  await page.locator("#apply-configuration").click();
  await page.waitForFunction(
    () =>
      liftTest.getReady() &&
      !liftTest.transitions.busy &&
      !liftTest.workspace.dirty,
    null,
    { timeout: 60000 },
  );
  console.log("Lighter load applied");
  assert.equal(
    await page.evaluate(
      () => liftTest.workspace.active.scene.bodies.find((b) => b.cargo).mass,
    ),
    0.24 * weight,
  );
  await page.locator("#flight-mode").uncheck();
  assert.equal(await page.locator("#rock-lift-controls").isHidden(), true);
  await page.locator("#flight-mode").check();
  assert.equal(await page.locator("#rock-lift-controls").isVisible(), true);
  await page.setViewportSize({ width: 390, height: 844 });
  if (!(await page.locator("#fit-rock-lift").isVisible()))
    await page.locator("#toggle-settings").click();
  await page.locator("#fit-rock-lift").scrollIntoViewIfNeeded();
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  );
  await page.screenshot({ path: "/tmp/rock-lift-controls.png" });
  assert.deepEqual(errors, []);
  console.log(
    "Lift readout, lighter-load staging, restart, gravity toggle and mobile layout passed",
  );
} finally {
  await browser.close();
}
