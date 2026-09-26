import { chromium, firefox } from "playwright";
import assert from "node:assert/strict";
const base =
  process.env.OPTIMIZATION_TEST_URL || "http://127.0.0.1:8081/optimization/";
for (const [name, type] of [
  ["chromium", chromium],
  ["firefox", firefox],
]) {
  const browser = await type.launch({
    headless: true,
    ...(name === "chromium"
      ? { args: ["--no-sandbox", "--use-angle=swiftshader"] }
      : {
          firefoxUserPrefs: {
            "webgl.force-enabled": true,
            "webgl.disabled": false,
            "webgl.out-of-process": false,
          },
        }),
  });
  const page = await browser.newPage({
    viewport: { width: 1536, height: 1080 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  try {
    await page.goto(base);
    await page.waitForFunction(() => window.optimizationReady);
    await page.locator("#algorithm").selectOption("wave");
    await page.locator('[name="walkers"]').fill("8");
    await page.locator("#algorithm-parameters summary").click();
    await page.locator('[name="elites"]').fill("5");
    await page.locator("#perturbation").selectOption("cloning_guided");
    await page.locator("#boundary").selectOption("none");
    await page.locator("#apply").click();
    await page.waitForFunction(() =>
      document.querySelector("#status").textContent.startsWith("Ready."),
    );
    const iteration = await page.locator("#iteration").textContent();
    const evaluations = await page.locator("#evaluations").textContent();
    for (const boundary of ["cma", "periodic", "none"]) {
      await page.locator("#boundary").selectOption(boundary);
      assert.match(
        await page.locator("#live-status").textContent(),
        /Unapplied/,
      );
      await page.locator("#apply-live").click();
      await page.waitForFunction(
        () =>
          document.querySelector("#status").textContent ===
          "Live settings applied.",
      );
      assert.equal(await page.locator("#iteration").textContent(), iteration);
      assert.equal(
        await page.locator("#evaluations").textContent(),
        evaluations,
      );
      assert.equal(await page.locator("#boundary").inputValue(), boundary);
    }
    await page.locator("#run").click();
    await page.waitForFunction(
      () => !document.querySelector("#pause").disabled,
    );
    await page.locator("#boundary").selectOption("cma");
    await page.locator("#apply-live").click();
    await page.waitForFunction(() =>
      /up to date/.test(document.querySelector("#live-status").textContent),
    );
    assert.equal(await page.locator("#pause").isDisabled(), false);
    await page.locator("#pause").click();
    await page.locator("#algorithm").selectOption("cmaes_bipop");
    assert.equal(await page.locator("#boundary").isDisabled(), true);
    assert.equal(await page.locator("#boundary").inputValue(), "cma");
    await page.locator("#algorithm").selectOption("wave");
    assert.equal(await page.locator("#boundary").isDisabled(), false);
    await page.screenshot({ path: `/tmp/optimization-boundary-${name}.png` });
    assert.deepEqual(errors, []);
    console.log(
      `${name}: paused/running boundary changes and CMA restrictions passed`,
    );
  } finally {
    await browser.close();
  }
}
