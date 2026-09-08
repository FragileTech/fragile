import { chromium, firefox } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
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
    // Firefox needs an X display for Mesa WebGL on Linux CI runners.
    headless:
      name !== "firefox" || process.env.OPTIMIZATION_FIREFOX_HEADLESS !== "0",
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
    await page.addInitScript(() => {
      window.addEventListener("error", (event) => {
        window.optimizationStartupError = event.message;
      });
      window.addEventListener("unhandledrejection", (event) => {
        window.optimizationStartupError =
          event.reason?.message || String(event.reason);
      });
    });
    await page.goto(base);
    await page.waitForFunction(() => {
      if (window.optimizationStartupError)
        throw new Error(window.optimizationStartupError);
      const status = document.getElementById("status");
      if (status?.classList.contains("error"))
        throw new Error(status.textContent);
      return window.optimizationReady;
    });
    assert.equal(await page.locator("#iteration").textContent(), "0");
    const apply = async () => {
      await page.locator("#apply").click();
      await page.waitForFunction(() => {
        const status = document.getElementById("status");
        if (status.classList.contains("error"))
          throw new Error(status.textContent);
        return status.textContent.startsWith("Ready.");
      });
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
    for (const algorithm of ["cmaes_active", "cmaes_bipop"]) {
      await page.locator("#algorithm").selectOption(algorithm);
      assert.equal(await page.locator('[name="walkers"]').isDisabled(), true);
      assert.equal(await page.locator('[name="periodic"]').isDisabled(), true);
      assert.equal(await page.locator("#perturbation").isDisabled(), true);
      await page.locator('[name="cma_population"]').fill("8");
      if (algorithm === "cmaes_bipop")
        await page.locator('[name="cma_runs"]').fill("2");
      await apply();
      await step();
      assert.match(await page.locator("#cma-note").textContent(), /8/);
    }
    await page.locator("#record-history").check();
    for (const algorithm of ["cmaes_active", "cmaes_bipop"]) {
      await page.locator("#benchmark").selectOption("constant");
      await page.locator("#algorithm").selectOption(algorithm);
      if (algorithm === "cmaes_bipop")
        await page.locator('[name="cma_runs"]').fill("2");
      await apply();
      await page.locator("#run").click();
      await page.waitForFunction(() =>
        document
          .getElementById("status")
          .textContent.startsWith("Optimizer finished:"),
      );
      assert.equal(await page.locator("#step").isDisabled(), true);
      const download = page.waitForEvent("download");
      await page.locator("#save").click();
      const path = `${output}/${name}-${algorithm}.fgopt`;
      await (await download).saveAs(path);
      const saved = JSON.parse(await readFile(path, "utf8"));
      assert.equal(saved.config.precision, "float64");
      assert.ok(saved.frames.at(-1).metadata.finished);
      if (algorithm === "cmaes_bipop")
        assert.ok(saved.frames.at(-1).metadata.restarts > 0);
      await page.locator("#file").setInputFiles(path);
      await page.waitForFunction(
        () => document.getElementById("iteration").textContent === "0",
      );
    }
    await page.locator("#record-history").uncheck();
    await page.locator("#benchmark").selectOption("quadratic");
    for (const algorithm of ["wave", "fmc", "wave_jump", "gas"]) {
      await page.locator("#algorithm").selectOption(algorithm);
      await page.locator("#perturbation").selectOption("local_covariance");
      await page.locator('[name="walkers"]').fill("16");
      await page.locator('[name="covariance_learning_rate"]').fill("0.2");
      await apply();
      await step();
      assert.equal(
        await page.locator('[name="covariance_learning_rate"]').inputValue(),
        "0.2",
      );
      assert.match(
        await page.locator("#perturbation-note").textContent(),
        /local proposal/,
      );
    }
    await page.locator("#algorithm").selectOption("graph");
    assert.equal(
      await page
        .locator('#perturbation option[value="local_covariance"]')
        .count(),
      0,
    );
    await page.locator("#algorithm").selectOption("gas");
    assert.equal(
      await page.locator("#perturbation").inputValue(),
      "gas_adaptive",
    );
    assert.equal(await page.locator('[name="gas_tabu"]').isChecked(), true);
    assert.equal(
      await page.locator('[name="gas_local_search"]').isChecked(),
      true,
    );
    await page.locator('[name="walkers"]').fill("8");
    for (const tabu of [false, true])
      for (const local of [false, true]) {
        await page.locator('[name="gas_tabu"]').setChecked(tabu);
        await page.locator('[name="gas_local_search"]').setChecked(local);
        await apply();
        await step();
        assert.equal(await page.locator('[name="gas_tabu"]').isChecked(), tabu);
        assert.equal(
          await page.locator('[name="gas_local_search"]').isChecked(),
          local,
        );
      }
    await page.locator("#benchmark").selectOption("stochastic_gaussian");
    assert.equal(
      await page.locator('[name="gas_local_search"]').isDisabled(),
      true,
    );
    assert.match(
      await page.locator("#gas-local-note").textContent(),
      /disabled for stochastic/,
    );
    await apply();
    await step();
    await page.locator("#benchmark").selectOption("rastrigin");
    await page.locator("#algorithm").selectOption("euclidean");
    assert.equal(
      await page.locator('#perturbation option[value="gas_adaptive"]').count(),
      0,
    );
    assert.equal(await page.locator("#perturbation").inputValue(), "gaussian");
    await apply();
    assert.equal(await page.locator("#record-history").isChecked(), false);
    await step();
    await step();
    assert.equal(
      await page.locator("#frame-label").textContent(),
      "Live · recording off",
    );
    assert.equal(await page.locator("#save").isDisabled(), true);
    assert.equal(await page.locator("#timeline").isDisabled(), true);
    await page.locator("#record-history").check();
    assert.equal(await page.locator("#objective").inputValue(), "minimize");
    assert.equal(await page.locator("#perturbation").inputValue(), "gaussian");
    for (const algorithm of [
      "euclidean",
      "wave",
      "graph",
      "fmc",
      "wave_jump",
    ]) {
      for (const objective of ["minimize", "maximize"]) {
        await page.locator("#algorithm").selectOption(algorithm);
        await page.locator("#objective").selectOption(objective);
        await page
          .locator("#perturbation")
          .selectOption(objective === "minimize" ? "gaussian" : "uniform");
        await page.locator('[name="perturbation_std"]').fill("0.3");
        if (["fmc", "wave_jump"].includes(algorithm))
          await page.locator('[name="horizon"]').fill("2");
        await page.locator('[name="walkers"]').fill("24");
        await page.locator('[name="periodic"]').check();
        await apply();
        const initialBest = Number(
          (await page.locator("#best").textContent()).replaceAll(",", ""),
        );
        await page.locator("#view").selectOption("landscape");
        await step();
        await page.locator("#view").selectOption("spatial");
        for (let t = 0; t < 5; t++) await step();
        const finalBest = Number(
          (await page.locator("#best").textContent()).replaceAll(",", ""),
        );
        assert.ok(
          objective === "minimize"
            ? finalBest <= initialBest
            : finalBest >= initialBest,
        );
        if (["fmc", "wave_jump"].includes(algorithm)) {
          await page.locator("#walker-index").fill("24");
          await page.locator("#walker-index").press("Tab");
          assert.match(
            await page.locator("#walker-info").textContent(),
            /Committed position/,
          );
        }
        assert.ok(
          Number((await page.locator("#alive").textContent()).split("/")[0]) >
            0,
        );
      }
    }
    // Hard COCO functions, staged instances, budget stop and analyzer export.
    await page.locator("#objective").selectOption("minimize");
    await page.locator("#benchmark").selectOption("bbob_21");
    await page.locator("#dimensions").fill("5");
    await page.locator('[name="coco_instance"]').fill("3");
    await page.locator('[name="walkers"]').fill("8");
    await page.locator('[name="max_evaluations"]').fill("65");
    await apply();
    assert.match(
      await page.locator("#benchmark-note").textContent(),
      /Instance 3/,
    );
    assert.equal(await page.locator("#chart-axis").inputValue(), "evaluations");
    await page.locator("#run").click();
    await page.waitForFunction(() =>
      document
        .getElementById("status")
        .textContent.startsWith("Evaluation budget reached"),
    );
    assert.equal(await page.locator("#step").isDisabled(), true);
    assert.ok(
      Number(
        (await page.locator("#evaluations").textContent()).split("/")[0],
      ) <= 65,
    );
    const csvDownload = page.waitForEvent("download");
    await page.locator("#export-csv").click();
    const csvPath = `${output}/${name}-fixed-budget.csv`;
    await (await csvDownload).saveAs(csvPath);
    assert.match(
      await readFile(csvPath, "utf8"),
      /"evaluations","best","function"/,
    );
    await page.locator("#chart-axis").selectOption("iteration");
    for (const functionId of ["bbob_23", "bbob_24"]) {
      await page.locator("#benchmark").selectOption(functionId);
      await page.locator('[name="max_evaluations"]').fill("0");
      await apply();
      await page.locator("#view").selectOption("landscape");
      await step();
      await page.locator("#view").selectOption("spatial");
      await step();
    }
    await page.screenshot({
      path: `${output}/${name}-coco.png`,
      fullPage: true,
    });
    await page.locator("#objective").selectOption("maximize");
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
    const savedRun = JSON.parse(await readFile(saved, "utf8"));
    assert.equal(savedRun.config.objective, "maximize");
    assert.equal(savedRun.config.perturbation, "uniform");
    assert.equal(savedRun.config.perturbation_std, 0.3);
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
    assert.equal(await page.locator("#objective").inputValue(), "maximize");
    assert.equal(await page.locator("#perturbation").inputValue(), "uniform");
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
  } catch (error) {
    const diagnostics = {
      browser: name,
      url: page.url(),
      error: error.message,
      pageErrors: errors,
      status: await page
        .locator("#status")
        .textContent({ timeout: 1000 })
        .catch(() => null),
    };
    console.error(diagnostics);
    await writeFile(
      `${output}/${name}-failure.json`,
      JSON.stringify(diagnostics, null, 2),
    );
    throw error;
  } finally {
    await browser.close();
  }
}
