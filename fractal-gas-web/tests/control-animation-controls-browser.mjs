// Run against served web/: CONTROL_TEST_URL=http://127.0.0.1:8088/lab/.
// CONTROL_MAIN_ONLY=1 skips the independently verified workshop flow.
// Cosmetic toggle readiness intentionally does not wait for the native worker.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
const errors = [];
console.log("Starting animation controls QA at half raster resolution");
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 960 },
    serviceWorkers: "block",
    reducedMotion: "reduce",
  });
  context.setDefaultTimeout(60000);
  await context.addInitScript(() => {
    window.coi = { shouldRegister: () => false };
  });
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await context.route("**/lab/asset-gallery.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("Math.min(devicePixelRatio, 2)", "0.5")}\nwindow.animationQA = { get ready() { return !!model; }, get clock() { return previewTime; }, pose() { const result=[]; model?.traverse(n=>result.push([n.name,...n.position.toArray(),...n.rotation.toArray(),...n.scale.toArray(),n.visible])); return result; } };`,
    });
  });
  await context.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${await response.text()}\nrenderer.renderer.setPixelRatio(0.5); window.animationQA = {renderer};`,
    });
  });
  const checkbox = page.getByRole("checkbox", {
    name: "Animations",
    exact: true,
  });
  if (process.env.CONTROL_MAIN_ONLY !== "1") {
    await page.goto(`${base}asset-gallery.html`);
    await page.waitForFunction(() => window.animationQA?.ready);
    assert.equal(
      await checkbox.isChecked(),
      false,
      "reduced-motion defaults off",
    );
    assert.equal(await page.locator("#animate-parts").isDisabled(), true);
    await checkbox.check();
    assert.equal(
      await page.getByRole("button", { name: "Play animation" }).isEnabled(),
      true,
    );
    const rest = await page.evaluate(() => animationQA.pose());
    await page.getByRole("button", { name: "Play animation" }).click();
    await page.waitForFunction(() => animationQA.clock > 0.3);
    assert.notDeepEqual(await page.evaluate(() => animationQA.pose()), rest);
    await page.getByRole("button", { name: "Pause animation" }).click();
    const paused = await page.evaluate(() => animationQA.pose());
    await page.waitForTimeout(200);
    assert.deepEqual(
      await page.evaluate(() => animationQA.pose()),
      paused,
      "pause holds pose",
    );
    await page.getByRole("button", { name: "Side", exact: true }).click();
    assert.deepEqual(
      await page.evaluate(() => animationQA.pose()),
      rest,
      "Side resets authored pose",
    );
    for (const view of ["Top", "Side"]) {
      await page.getByRole("button", { name: "Play animation" }).click();
      await page.waitForFunction(() => animationQA.clock > 0.1);
      await page.getByRole("button", { name: view, exact: true }).click();
      assert.equal(await page.evaluate(() => animationQA.clock), 0);
      assert.deepEqual(await page.evaluate(() => animationQA.pose()), rest);
    }
    await page
      .getByRole("button", { name: "Perspective", exact: true })
      .click();
    await page.getByRole("button", { name: "Play animation" }).click();
    await page.waitForFunction(() => animationQA.clock > 0.2);
    await page.screenshot({ path: "/tmp/lab-workshop-animation.png" });
    await checkbox.uncheck();
    assert.deepEqual(
      await page.evaluate(() => animationQA.pose()),
      rest,
      "global off resets cosmetic pose",
    );
    assert.equal(await page.locator("#animate-parts").isDisabled(), true);
    await page.reload();
    await page.waitForFunction(() => window.animationQA?.ready);
    assert.equal(await checkbox.isChecked(), false, "off persists on reload");
    await checkbox.check();
    await page.reload();
    await page.waitForFunction(() => window.animationQA?.ready);
    assert.equal(
      await checkbox.isChecked(),
      true,
      "explicit on overrides reduced motion after reload",
    );
    assert.equal(
      await page.getByRole("button", { name: "Play animation" }).isVisible(),
      true,
      "workshop starts stopped",
    );

    console.log("Workshop tests passed");
  }
  const main = await context.newPage();
  await main.addInitScript(() =>
    localStorage.setItem("fragile.lab.animations", "on"),
  );
  main.on("pageerror", (error) => errors.push(error.message));
  await main.goto(base);
  console.log(
    "Main loaded",
    await main.evaluate(() => ({
      qa: !!window.animationQA,
      backend: document.querySelector("#backend")?.textContent,
      status: document.querySelector("#status")?.textContent,
    })),
  );
  await main.waitForFunction(() => window.animationQA?.renderer, null, {
    timeout: 60000,
  });
  const toggle = main.getByRole("checkbox", {
    name: "Animations",
    exact: true,
  });
  assert.equal(await toggle.isChecked(), true);
  assert.equal(
    await main.evaluate(() => animationQA.renderer.animationsEnabled),
    true,
  );
  await toggle.uncheck();
  assert.equal(
    await main.evaluate(() => animationQA.renderer.animationsEnabled),
    false,
  );
  if (process.env.CONTROL_MAIN_ONLY === "1") {
    await page.route("**/lab/animation-probe.html", (route) =>
      route.fulfill({
        contentType: "text/html",
        body: '<input type="checkbox" name="animations"><script type="module">import {installAnimationControls} from "./animation-controls.js"; installAnimationControls();window.animationProbeReady=true;</script>',
      }),
    );
    await page.goto(`${base}animation-probe.html`);
    await page.waitForFunction(() => window.animationProbeReady);
  }
  await page.waitForFunction(
    () => !document.querySelector('input[name="animations"]').checked,
  );
  if (process.env.CONTROL_MAIN_ONLY !== "1")
    assert.equal(
      await page.locator("#animate-parts").isDisabled(),
      true,
      "other tab reflects global off",
    );
  await toggle.check();
  assert.equal(
    await main.evaluate(() => animationQA.renderer.animationsEnabled),
    true,
  );
  await page.waitForFunction(
    () => document.querySelector('input[name="animations"]').checked,
  );
  await main.screenshot({ path: "/tmp/lab-animation-toggle.png" });
  assert.deepEqual(errors, []);
  console.log(
    process.env.CONTROL_MAIN_ONLY === "1"
      ? "PASS: Lab immediate toggle and cross-tab preference synchronization; /tmp/lab-animation-toggle.png"
      : "PASS: reduced motion; explicit preference/reload; shared tabs; Lab immediate toggle; Workshop play/pause, Side/Top reset, global off; screenshots /tmp/lab-workshop-animation.png and /tmp/lab-animation-toggle.png",
  );
} finally {
  await browser.close();
}
