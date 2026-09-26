import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1080 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const base =
  process.env.OPTIMIZATION_TEST_URL || "http://127.0.0.1:8081/optimization/";
const field = (key) => page.locator(`[name="${key}"]`);
const live = page.locator("#live-status");
try {
  await page.goto(base);
  await page.waitForFunction(() => window.optimizationReady);
  await page.locator("#benchmark").selectOption("quadratic");
  await page.locator("#algorithm").selectOption("wave");
  await field("walkers").fill("8");
  await field("max_walkers").fill("32");
  await field("boundary").selectOption("periodic");
  await page.locator("#record-history").check();
  await page.locator("#apply").click();
  await page.waitForFunction(() =>
    document.querySelector("#status").textContent.startsWith("Ready."),
  );
  const pausedIteration = await page.locator("#iteration").textContent();
  const pausedEvaluations = await page.locator("#evaluations").textContent();
  await field("perturbation_std").fill("0.2");
  assert.match(await live.textContent(), /Unapplied/);
  await page.locator("#apply-live").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent ===
      "Live settings applied.",
  );
  assert.equal(await page.locator("#iteration").textContent(), pausedIteration);
  assert.equal(
    await page.locator("#evaluations").textContent(),
    pausedEvaluations,
  );
  assert.equal(await page.locator("#pause").isDisabled(), true);
  // Invalid coupled settings reject the entire batch without losing edits.
  await field("walkers").fill("40");
  await page.locator("#apply-live").click();
  await page.waitForFunction(() =>
    document.querySelector("#status").classList.contains("error"),
  );
  assert.match(
    await page.locator("#population-status").textContent(),
    /8 active/,
  );
  await field("max_walkers").fill("48");
  await page.locator("#apply-live").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#population-status")
      .textContent.startsWith("40 active"),
  );
  // Deterministic delayed replies verify that later edits survive an in-flight Apply.
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
  await field("perturbation_std").fill("0.3");
  await page.locator("#apply-live").click();
  await field("perturbation_std").fill("0.4");
  await page.waitForFunction(
    () =>
      document.querySelector("#live-status").textContent ===
      "Unapplied changes.",
  );
  assert.equal(await field("perturbation_std").inputValue(), "0.4");
  // A second Apply can revert a value before the first response arrives.
  await page.locator("#apply-live").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#live-status").textContent ===
      "Live settings are up to date.",
  );
  await field("perturbation_std").fill("0.5");
  await page.locator("#apply-live").click();
  await field("perturbation_std").fill("0.4");
  await page.locator("#apply-live").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#live-status").textContent ===
      "Live settings are up to date.",
  );
  assert.equal(await field("perturbation_std").inputValue(), "0.4");
  await page.locator("#run").click();
  await page.waitForFunction(
    () => Number(document.querySelector("#iteration").textContent) > 0,
  );
  await field("walkers").fill("12");
  await page.locator("#apply-live").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#population-status")
      .textContent.startsWith("12 active"),
  );
  assert.equal(await page.locator("#pause").isEnabled(), true);
  await page.locator("#pause").click();
  await field("max_evaluations").fill("1");
  await page.locator("#apply-live").click();
  await page.waitForFunction(() => document.querySelector("#run").disabled);
  await field("max_evaluations").fill("0");
  await page.locator("#apply-live").click();
  await page.waitForFunction(() => !document.querySelector("#run").disabled);
  assert.equal(await page.locator("#pause").isDisabled(), true);
  const download = page.waitForEvent("download");
  await page.locator("#save").click();
  const saved = await download;
  const content = await readFile(await saved.path(), "utf8");
  const recording = JSON.parse(content);
  assert.equal(recording.version, 3);
  assert.equal(recording.config.walkers, 8);
  assert.ok(recording.frames.some((f) => f.metadata?.settings_update));
  // Worker admission rejects an update before engine mutation when recording space is exhausted.
  const admission = await page.evaluate(async () => {
    const { EngineClient } = await import("./client.js");
    const client = new EngineClient();
    try {
      await client.request("create", {
        config: {
          algorithm: "wave",
          walkers: 8,
          max_walkers: 32,
          periodic: true,
        },
      });
      let rejected = false;
      try {
        await client.request("updateSettings", {
          patch: { walkers: 16 },
          remaining: 1,
        });
      } catch {
        rejected = true;
      }
      const next = await client.request("step", { remaining: Infinity });
      return { rejected, n: next.frame[1], config: next.config.walkers };
    } finally {
      client.dispose();
    }
  });
  assert.deepEqual(admission, { rejected: true, n: 8, config: 8 });
  await page.locator("#file").setInputFiles({
    name: "live.fgopt",
    mimeType: "application/json",
    buffer: Buffer.from(content),
  });
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.startsWith("Recording loaded."),
  );
  assert.equal(await page.locator("#apply-live").isDisabled(), true);
  await field("perturbation_std").fill("0.7");
  assert.equal(await page.locator("#apply-live").isDisabled(), true);
  await page.screenshot({
    path: "/tmp/optimization-live-settings.png",
    fullPage: true,
  });
  // Exercise each live form against its optimizer, including deferred planner application.
  for (const algorithm of [
    "euclidean",
    "gas",
    "graph",
    "fmc",
    "wave_jump",
    "cmaes_active",
  ]) {
    await page.locator("#algorithm").selectOption(algorithm);
    await page
      .locator("details.advanced")
      .evaluateAll((nodes) => nodes.forEach((n) => (n.open = true)));
    if (algorithm !== "cmaes_active") {
      await field("walkers").fill("8");
      await field("max_walkers").fill("32");
      await field("boundary").selectOption("periodic");
      if (["fmc", "wave_jump"].includes(algorithm))
        await field("horizon").fill("2");
    }
    await field("max_evaluations").fill("0");
    await page.locator("#apply").click();
    await page.waitForFunction(() =>
      document.querySelector("#status").textContent.startsWith("Ready."),
    );
    if (["fmc", "wave_jump"].includes(algorithm)) {
      await page.locator("#step").click();
      await page.waitForFunction(
        () => document.querySelector("#iteration").textContent === "1",
      );
      await field("perturbation_std").fill("0.02");
      await field("walkers").fill("12");
      await page.locator("#apply-live").click();
      await page.waitForFunction(() =>
        document
          .querySelector("#live-status")
          .textContent.startsWith("Changes queued"),
      );
      assert.equal(await page.locator("#iteration").textContent(), "1");
      await page.locator("#run").click();
      await page.waitForFunction(() =>
        document
          .querySelector("#population-status")
          .textContent.startsWith("12 active"),
      );
      await page.locator("#pause").click();
    } else if (algorithm === "cmaes_active") {
      assert.equal(await page.locator("#perturbation").isDisabled(), true);
      await field("max_evaluations").fill("1");
      await page.locator("#apply-live").click();
      await page.waitForFunction(() => document.querySelector("#run").disabled);
      await field("max_evaluations").fill("0");
      await page.locator("#apply-live").click();
      await page.waitForFunction(
        () => !document.querySelector("#run").disabled,
      );
    } else {
      await field("walkers").fill("12");
      await page.locator("#apply-live").click();
      await page.waitForFunction(() =>
        document.querySelector("#population-status").textContent.includes("12"),
      );
      assert.equal(await page.locator("#iteration").textContent(), "0");
      if (algorithm === "graph")
        assert.equal(
          await page.locator("#walkers-label").textContent(),
          "Target leaves",
        );
    }
  }
  await page.screenshot({
    path: "/tmp/optimization-live-settings-final.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log("Live settings browser checks passed.");
} finally {
  await browser.close();
}
