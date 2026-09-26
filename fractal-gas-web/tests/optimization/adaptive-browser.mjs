import { chromium, firefox } from "playwright";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const base =
  process.env.OPTIMIZATION_TEST_URL || "http://127.0.0.1:8093/optimization/";
for (const strategy of process.env.PERTURBATION_TEST
  ? [process.env.PERTURBATION_TEST]
  : ["adaptive_fractal", "cloning_guided"])
  for (const type of process.env.BROWSER_TEST === "firefox"
    ? [firefox]
    : process.env.BROWSER_TEST === "chromium"
      ? [chromium]
      : [chromium, firefox]) {
    const browser = await type.launch({
      headless: true,
      ...(type === chromium
        ? { args: ["--no-sandbox", "--use-angle=swiftshader"] }
        : {}),
    });
    const page = await browser.newPage({
      viewport: { width: 1500, height: 1100 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const field = (name) => page.locator(`[name="${name}"]`);
    try {
      await page.goto(base);
      await page.waitForFunction(() => window.optimizationReady);
      await field("benchmark").selectOption("quadratic");
      await field("algorithm").selectOption("wave");
      await field("walkers").fill("8");
      await field("max_walkers").fill("32");
      await field("boundary").selectOption("periodic");
      await field("perturbation").selectOption(strategy);
      await field("max_evaluations").fill("10000");
      await field("controller_enabled").check();
      await page.locator("#apply").click();
      await page.waitForFunction(() =>
        document.querySelector("#status").textContent.startsWith("Ready."),
      );
      await page.locator("#step").click();
      await page.waitForFunction(
        () => Number(document.querySelector("#iteration").textContent) > 0,
      );
      assert.match(
        await page.locator("#exploration-status").textContent(),
        /local models/,
      );
      if (strategy === "cloning_guided") {
        await field("cloning_drift_strength").fill("0.15");
        await field("cloning_geometry").uncheck();
        assert.match(
          await page.locator("#exploration-status").textContent(),
          /signed clone scores/,
        );
      }
      const pausedIteration = await page.locator("#iteration").textContent();
      const pausedEvaluations = await page
        .locator("#evaluations")
        .textContent();
      await field("adaptive_min_scale").fill("0.02");
      await field("adaptive_max_scale").fill("0.2");
      assert.equal(await field("scale_auto").isChecked(), false);
      await page.locator("#apply-live").click();
      await page.waitForFunction(
        () =>
          document.querySelector("#status").textContent ===
          "Live settings applied.",
      );
      assert.equal(
        await page.locator("#iteration").textContent(),
        pausedIteration,
      );
      assert.equal(
        await page.locator("#evaluations").textContent(),
        pausedEvaluations,
      );
      await field("adaptive_min_scale").fill("0.3");
      await page.locator("#apply-live").click();
      await page.waitForFunction(() =>
        document.querySelector("#status").classList.contains("error"),
      );
      assert.equal(await field("adaptive_min_scale").inputValue(), "0.3");
      assert.equal(
        await page.locator("#evaluations").textContent(),
        pausedEvaluations,
      );
      await field("adaptive_min_scale").fill("0.02");
      await page.evaluate(async () => {
        const { EngineClient } = await import("./client.js");
        const original = EngineClient.prototype.request;
        EngineClient.prototype.request = async function (type, fields) {
          const result = await original.call(this, type, fields);
          if (type === "updateSettings")
            await new Promise((resolve) => setTimeout(resolve, 250));
          return result;
        };
      });
      await field("adaptive_max_scale").fill("0.3");
      await page.locator("#apply-live").click();
      await field("adaptive_max_scale").fill("0.4");
      await page.waitForFunction(
        () =>
          document.querySelector("#live-status").textContent ===
          "Unapplied changes.",
      );
      assert.equal(await field("adaptive_max_scale").inputValue(), "0.4");
      await page.locator("#apply-live").click();
      await page.waitForFunction(
        () =>
          document.querySelector("#live-status").textContent ===
          "Live settings are up to date.",
      );
      await page.locator("#run").click();
      await field("adaptive_min_scale").fill("0.03");
      await page.locator("#apply-live").click();
      await page.waitForFunction(
        () =>
          document.querySelector("#live-status").textContent ===
          "Live settings are up to date.",
      );
      assert.equal(await page.locator("#pause").isEnabled(), true);
      await page.locator("#pause").click();
      const roundBefore = Number(
        (await page.locator("#controller-status").textContent()).match(
          /Round (\d+)/,
        )[1],
      );
      await page.locator("#restart-round").click();
      await page.waitForFunction(() =>
        document
          .querySelector("#controller-status")
          .textContent.includes("restart queued"),
      );
      await page.locator("#step").click();
      await page.waitForFunction(
        (before) =>
          Number(
            document
              .querySelector("#controller-status")
              .textContent.match(/Round (\d+)/)[1],
          ) > before,
        roundBefore,
      );
      assert.ok(await page.locator("#basin-rows tr").count());
      const download = page.waitForEvent("download");
      await page.locator("#export-basins").click();
      const archive = await download;
      const text = await readFile(await archive.path(), "utf8");
      assert.equal(JSON.parse(text).format, "fractal-basin-archive");
      await page.locator("#import-basins").setInputFiles({
        name: "round.basins",
        mimeType: "application/json",
        buffer: Buffer.from(text),
      });
      await page.waitForFunction(() =>
        document
          .querySelector("#status")
          .textContent.startsWith("Imported basin values"),
      );
      assert.match(
        await page.locator("#basin-rows").textContent(),
        /reference/,
      );
      await field("algorithm").selectOption("euclidean");
      await field("perturbation").selectOption(strategy);
      await field("adaptive_euclidean_mode").selectOption("position");
      assert.equal(await field("gamma").isDisabled(), true);
      await field("algorithm").selectOption("cmaes_active");
      assert.equal(await page.locator("#controller-panel").isHidden(), true);
      assert.deepEqual(errors, []);
      console.log(
        `${type.name()} ${strategy}: movement controls, restart, archive and CMA restrictions passed`,
      );
    } catch (error) {
      console.error(
        type.name(),
        strategy,
        await page.locator("#status").textContent(),
        await page.locator("#live-status").textContent(),
      );
      console.error("page errors", errors);
      throw error;
    } finally {
      await browser.close();
    }
  }
