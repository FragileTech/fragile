import { chromium, firefox } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
const base =
  process.env.OPTIMIZATION_TEST_URL || "http://127.0.0.1:8081/optimization/";
const output =
  process.env.OPTIMIZATION_SCREENSHOTS || "/tmp/fractal-optimization-browser";
await mkdir(output, { recursive: true });
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
    }),
    errors = [],
    workers = [];
  page.setDefaultTimeout(30000);
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("worker", (worker) => workers.push(worker));
  try {
    await page.goto(base);
    await page.waitForFunction(() => window.optimizationReady);
    assert.equal(await page.locator("#iteration").textContent(), "0");
    const apply = async () => {
      await page.locator("#apply").click();
      await page.waitForFunction(() =>
        document.getElementById("status").textContent.startsWith("Ready."),
      );
    };
    const step = async () => {
      const before = Number(await page.locator("#iteration").textContent());
      await page.locator("#step").click();
      await page.waitForFunction(
        (value) =>
          Number(document.getElementById("iteration").textContent) > value,
        before,
      );
    };
    for (const algorithm of ["euclidean", "wave", "graph"]) {
      await page.locator("#algorithm").selectOption(algorithm);
      await page.locator('[name="walkers"]').fill("24");
      await page.locator('[name="periodic"]').check();
      await apply();
      await step();
      await step();
      await step();
      assert.ok(
        Number((await page.locator("#alive").textContent()).split("/")[0]) > 0,
      );
    }
    await page.locator("#algorithm").selectOption("euclidean");
    await page.locator("#benchmark").selectOption("rosenbrock");
    await page.locator("#dimensions").fill("2");
    await apply();
    await page.locator("#view").selectOption("landscape");
    await step();
    await page.locator("#walker-index").fill("1");
    await page.locator("#walker-index").press("Tab");
    await page.waitForFunction(() =>
      document.getElementById("walker-info").textContent.includes("Objective"),
    );
    await page.screenshot({
      path: `${output}/${name}-landscape.png`,
      fullPage: true,
    });
    await step();
    await step();
    const download = page.waitForEvent("download");
    await page.locator("#save").click();
    const saved = `${output}/${name}.fgopt`;
    await (await download).saveAs(saved);
    assert.match(await readFile(saved, "utf8"), /fgopt/);
    await page.locator("#timeline").fill("0");
    await page.locator("#timeline").dispatchEvent("input");
    assert.equal(await page.locator("#iteration").textContent(), "0");
    assert.equal(await page.locator("#run").isDisabled(), true);
    await page.locator("#latest").click();
    await step();
    await page.locator("#file").setInputFiles(saved);
    await page.waitForFunction(() =>
      document
        .getElementById("status")
        .textContent.startsWith("Recording loaded"),
    );
    assert.equal(await page.locator("#run").isDisabled(), true);
    await page.locator("#replay").click();
    await page.waitForFunction(
      () => Number(document.getElementById("iteration").textContent) > 0,
    );
    await page.locator("#latest").click();
    await page.locator("#reset").click();
    await page.waitForFunction(() =>
      document.getElementById("status").textContent.startsWith("Ready."),
    );
    await page.evaluate(
      () => new Promise((resolve) => setTimeout(resolve, 100)),
    );
    assert.equal(await page.locator("#iteration").textContent(), "0");
    await page.locator("#benchmark").selectOption("gaussian_mixture");
    await page.locator("#dimensions").fill("6");
    await apply();
    await page.locator("#axis-x").selectOption("3");
    await page.locator("#axis-y").selectOption("4");
    await step();
    assert.match(
      await page.locator("#slice-note").textContent(),
      /full-dimensional/,
    );
    await page.locator("#benchmark").selectOption("lennard_jones");
    await page.locator('[name="n_atoms"]').fill("3");
    await apply();
    await page.locator("#view").selectOption("spatial");
    await page.locator("#walker-index").fill("1");
    await page.locator("#walker-index").press("Tab");
    await page.waitForFunction(
      () => !document.getElementById("molecule").hidden,
    );
    await page.screenshot({
      path: `${output}/${name}-molecule.png`,
      fullPage: true,
    });
    await page.locator("#benchmark").selectOption("stochastic_gaussian");
    await apply();
    await step();
    assert.match(
      await page.locator("#slice-note").textContent(),
      /expected objective 0/,
    );
    await page.locator("#run").click();
    await page.waitForFunction(
      () => Number(document.getElementById("iteration").textContent) > 3,
    );
    console.log(
      `${name}: running ${await page.locator("#timings").textContent()}`,
    );
    await page.locator("#reset").click();
    await page.waitForFunction(() =>
      document.getElementById("status").textContent.startsWith("Ready."),
    );
    await page.evaluate(
      () => new Promise((resolve) => setTimeout(resolve, 100)),
    );
    assert.equal(await page.locator("#iteration").textContent(), "0");
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({
      path: `${output}/${name}-mobile.png`,
      fullPage: true,
    });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    assert.deepEqual(errors, []);
    assert.equal(workers.length, 1, "one simulation worker per page");
    console.log(`${name}: ${await page.locator("#timings").textContent()}`);
    await page.close();
    console.log(
      `${name}: algorithms, landscape/spatial, 6D slices, molecules, save/load/replay, reset, and mobile passed`,
    );
  } finally {
    await browser.close();
  }
}
