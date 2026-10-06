import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { metadata } from "../../web/euclidean-gas/lecture/registry.js";

const crossImportOnly = process.env.LECTURE_QFT_CROSS_IMPORT_ONLY === "1";
const continuationsOnly = process.env.LECTURE_QFT_CONTINUATIONS_ONLY === "1";
const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/partvi-review/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
async function healthy() {
  assert.equal(
    await page.locator("#error").isVisible(),
    false,
    await page.locator("#error").textContent(),
  );
}
async function ready() {
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent.startsWith("Ready") &&
      !document.querySelector("#step").disabled,
    null,
    { timeout: 180000 },
  );
  await healthy();
}
async function open(id) {
  await page.goto(`${base}/euclidean-gas/lecture.html?demo=${id}`);
  await ready();
  assert.equal(await page.locator('[data-demo^="VI-"]').count(), 66);
  assert.equal(await page.locator("#control-source").count(), 0);
}
async function step() {
  await page.locator("#step").click();
  await page.waitForFunction(
    () => !document.querySelector("#step").disabled,
    null,
    { timeout: 180000 },
  );
  await healthy();
}
async function finish() {
  for (let i = 0; i < 32; i++) {
    await step();
    if ((await page.locator("#status").textContent()).startsWith("Complete"))
      return;
  }
  assert.fail("Experiment did not complete within 32 worker advances");
}
async function result() {
  return JSON.parse(
    await page.locator(".calculation-details pre").textContent(),
  );
}
async function download(selector, name) {
  const [file] = await Promise.all([
    page.waitForEvent("download"),
    page.locator(selector).click(),
  ]);
  const path = new URL(name, output).pathname;
  await file.saveAs(path);
  return { path, data: JSON.parse(await readFile(path, "utf8")) };
}
function evidence(data, id) {
  assert.equal(data.request.id, id);
  assert.equal(data.configs.length, data.archives.length);
  assert.ok(data.archives.length > 0);
  for (const archive of data.archives) {
    assert.ok(archive.steps.length > 0);
    assert.ok(archive.anchors.length > 0);
    assert.ok(archive.steps.some((s) => s.field_evaluations.length > 0));
  }
}
async function replay(path, expected, selector = "#native-results") {
  await page.locator(selector).setInputFiles(path);
  await page.waitForFunction(
    (native) =>
      document.querySelector("#status").textContent ===
        (native
          ? "Experiment evidence recomputed in Rust"
          : "Archive validated") || !document.querySelector("#error").hidden,
    selector === "#native-results",
    { timeout: 180000 },
  );
  await healthy();
  const actual = await result();
  assert.deepEqual(
    actual.plots,
    expected.plots,
    "Recomputed scientific plots must match the executed evidence",
  );
  assert.deepEqual(
    actual.metrics,
    expected.metrics,
    "Recomputed measurements must match",
  );
  return actual;
}
async function crossDemoImport() {
  const path = new URL("VI-19-continuation-evidence.json", output).pathname;
  const bundle = JSON.parse(await readFile(path, "utf8"));
  const spec = metadata.find((demo) => demo.id === "VI-19");
  await open("VI-19");
  await page.locator("#native-results").setInputFiles(path);
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent ===
        "Experiment evidence recomputed in Rust" ||
      !document.querySelector("#error").hidden,
    null,
    { timeout: 180000 },
  );
  await healthy();
  const expected = await result();
  await writeFile(
    new URL("VI-19-import-expected-result.json", output),
    JSON.stringify(expected),
  );
  await open("VI-28");
  assert.equal(
    await page.locator("#question").textContent(),
    metadata.find((demo) => demo.id === "VI-28").question,
  );
  await replay(path, expected);
  for (const field of ["question", "prediction", "explanation"])
    assert.equal(await page.locator(`#${field}`).textContent(), spec[field]);
  assert.equal(
    await page.locator("#seed").inputValue(),
    String(bundle.request.seed),
  );
  assert.equal(new URL(page.url()).searchParams.get("demo"), "VI-19");
  assert.equal(
    await page.locator('[data-demo="VI-19"]').getAttribute("aria-current"),
    "page",
  );
  assert.equal(
    await page.locator('[data-demo="VI-28"]').getAttribute("aria-current"),
    "false",
  );
  assert.match(await page.locator("#kind").textContent(), /VI-19/);
  for (const control of spec.controls)
    assert.equal(
      await page.locator(`#control-${control.key}`).inputValue(),
      String(bundle.request.parameters[control.key] ?? control.value),
    );
  const exported = await download("#archive", "VI-19-cross-demo-evidence.json");
  assert.deepEqual(exported.data, bundle);
  await page.screenshot({
    path: new URL("cross-demo-import.png", output).pathname,
    fullPage: true,
  });
}
try {
  if (crossImportOnly) {
    await crossDemoImport();
    assert.deepEqual(errors, []);
    await writeFile(
      new URL("cross-import-validation.json", output),
      JSON.stringify(
        {
          passed: true,
          displayedDemoBeforeImport: "VI-28",
          importedDemo: "VI-19",
          contextControlsAndSeed: true,
          identicalPlotsAndMetrics: true,
          unchangedEvidenceExport: true,
          errors,
        },
        null,
        2,
      ),
    );
    console.log(
      "Cross-demo evidence import passed: VI-19 captions, controls, seed and identity replace VI-28 context; measurements and exported evidence remain identical.",
    );
  } else {
    if (!continuationsOnly) {
      await open("VI-08");
      const colors = await result();
      assert.match(colors.details.stage, /B1 input, step/);
      assert.ok(colors.details.source_slots.length > 0);
      assert.equal(
        colors.details.force.length,
        colors.details.source_slots.length,
      );
      assert.equal(
        colors.details.velocity.length,
        colors.details.source_slots.length,
      );
      assert.equal(
        colors.details.raw_color.length,
        colors.details.valid_mask.length,
      );
      colors.details.raw_color.forEach((row, i) => {
        assert.ok(
          row.every(
            (z) => Number.isFinite(z.real) && Number.isFinite(z.imaginary),
          ),
        );
        if (colors.details.valid_mask[i]) {
          const norm = row.reduce(
            (sum, z) => sum + z.real ** 2 + z.imaginary ** 2,
            0,
          );
          assert.ok(
            Math.abs(norm - 1) < 1e-10,
            "Valid recorded colors have unit norm",
          );
        }
      });
      assert.ok(await page.locator("#charts svg").count());
      const colorEvidence = await download("#archive", "color-evidence.json");
      evidence(colorEvidence.data, "VI-08");
      const colorFrame = colorEvidence.data.archives[0].steps.at(-1);
      const forceRecord = colorFrame.field_evaluations.find(
        (field) => field.stage === "B1" && field.field === "viscous_force",
      );
      const velocityRecord = colorFrame.field_evaluations.find(
        (field) =>
          field.stage === "B1" && field.field === "force_input_velocity",
      );
      assert.equal(forceRecord.version, velocityRecord.version);
      colors.details.source_slots.forEach((slot, index) => {
        assert.deepEqual(
          colors.details.force[index],
          forceRecord.values.slice(3 * slot, 3 * slot + 3),
        );
        assert.deepEqual(
          colors.details.velocity[index],
          velocityRecord.values.slice(3 * slot, 3 * slot + 3),
        );
      });
      await replay(colorEvidence.path, colors, "#archive-import");
      await page.screenshot({
        path: new URL("recorded-colors.png", output).pathname,
        fullPage: true,
      });

      await open("VI-36");
      await finish();
      const spectral = await result();
      assert.equal(spectral.details.readout, "twistor");
      assert.equal(spectral.details.aggregation, "frame_mean");
      assert.ok(spectral.details.fits.length > 0);
      assert.ok(
        spectral.details.lag_counts.every(
          (n) => n.training_pairs > 0 && n.heldout_pairs > 0,
        ),
      );
      for (const fit of spectral.details.fits) {
        assert.equal(
          fit.mass,
          null,
          "A correlation fit must not manufacture a particle mass",
        );
        assert.ok(
          ["empirical_fit", "inconclusive", "unavailable"].includes(fit.status),
        );
        if (fit.status === "inconclusive") {
          assert.equal(fit.decay_rate, null);
          assert.equal(fit.frequency, null);
        }
        if (fit.status !== "unavailable") {
          assert.ok(Number.isFinite(fit.candidate_decay_rate));
          assert.ok(Number.isFinite(fit.heldout_complex_rmse));
          assert.ok(fit.fit_points >= 3);
        }
      }
      const spectralEvidence = await download(
        "#archive",
        "spectral-evidence.json",
      );
      evidence(spectralEvidence.data, "VI-36");
      await replay(spectralEvidence.path, spectral);
      await page.screenshot({
        path: new URL("spectral-support.png", output).pathname,
        fullPage: true,
      });
    }
    for (const id of ["VI-19", "VI-22", "VI-45"]) {
      await open(id);
      for (const [key, value] of [
        ["walkers", "8"],
        ["replicas", "4"],
        ["horizon", "1"],
      ]) {
        const control = page.locator(`#control-${key}`);
        if (!(await control.isVisible()))
          await page.locator(".advanced summary").click();
        await control.selectOption(value);
        await ready();
      }
      await finish();
      const conditional = await result();
      assert.equal(
        conditional.details.calculation_origin,
        "independent_algorithm_continuations",
      );
      assert.equal(
        conditional.details.frozen_step,
        96,
        "Continuations must start at the displayed completed baseline",
      );
      assert.ok(conditional.details.continuation_executed_steps > 0);
      const continuationEvidence = await download(
        "#archive",
        `${id}-continuation-evidence.json`,
      );
      assert.ok(
        (await stat(continuationEvidence.path)).size < 128 * 1024 * 1024,
        "Complete continuation evidence must fit the browser import limit",
      );
      evidence(continuationEvidence.data, id);
      const baseline = continuationEvidence.data.archives[0];
      assert.equal(baseline.steps.length, 96);
      assert.equal(continuationEvidence.data.request.steps, 96);
      assert.equal(continuationEvidence.data.request.parameters.walkers, 8);
      assert.equal(continuationEvidence.data.request.parameters.replicas, 4);
      assert.equal(continuationEvidence.data.request.parameters.horizon, 1);
      assert.equal(
        conditional.details.frozen_step,
        baseline.steps.at(-1).report.step,
      );
      if (id !== "VI-45")
        assert.equal(
          conditional.details.frozen_population_version,
          baseline.steps.at(-1).final_population.version,
        );
      const manifest = continuationEvidence.data.continuation;
      assert.ok(manifest.frozen_checkpoint_cbor.length > 0);
      assert.ok(manifest.runs.length >= 8);
      assert.ok(new Set(manifest.runs.map((run) => run.future_seed)).size > 1);
      for (const run of manifest.runs) {
        assert.equal(run.recorded_steps, 1);
        assert.match(run.archive_fingerprint_fnv1a64, /^[0-9a-f]{16}$/);
        assert.equal(run.horizon, 1);
      }
      const replayed = await replay(continuationEvidence.path, conditional);
      assert.equal(
        replayed.details.frozen_step,
        conditional.details.frozen_step,
      );
      assert.equal(
        replayed.details.frozen_population_version,
        conditional.details.frozen_population_version,
      );
      assert.equal(
        replayed.details.continuation_executed_steps,
        conditional.details.continuation_executed_steps,
      );
      assert.deepEqual(
        replayed.details.replay_evidence.runs,
        conditional.details.replay_evidence.runs,
      );
      await page.screenshot({
        path: new URL(`${id}-complete-state-continuations.png`, output)
          .pathname,
        fullPage: true,
      });
    }

    if (!continuationsOnly) {
      await open("VI-28");
      await page.locator(".formula-index summary").click();
      await page.locator(".formula-index input").fill("reflection");
      await page.locator('[data-formula-demo="VI-28"]').first().waitFor();
      await page.setViewportSize({ width: 390, height: 844 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 2,
        ),
      );
      await page.screenshot({
        path: new URL("mobile-reflection.png", output).pathname,
        fullPage: true,
      });
    }
    await crossDemoImport();
    assert.deepEqual(errors, []);
    await writeFile(
      new URL("browser-validation.json", output),
      JSON.stringify(
        {
          passed: true,
          testedIds: continuationsOnly
            ? ["VI-19", "VI-22", "VI-45"]
            : ["VI-08", "VI-36", "VI-19", "VI-22", "VI-45", "VI-28"],
          recordedColors: !continuationsOnly,
          spectralSupport: !continuationsOnly,
          completedContinuations: true,
          frozenAtDisplayedStep: 96,
          continuationWalkers: 8,
          replicas: 4,
          horizon: 1,
          evidenceWithinImportLimit: true,
          evidenceRecomputed: true,
          crossDemoImport: true,
          formulaIndex: !continuationsOnly,
          mobile: !continuationsOnly,
          errors,
        },
        null,
        2,
      ),
    );
    console.log(
      continuationsOnly
        ? "Part VI continuation browser checks passed: all three experiments freeze at displayed step 96 and replay exactly within the evidence import limit."
        : "Part VI focused browser checks passed: recorded colors, spectral support, completed continuations, evidence recomputation, formula index and mobile layout.",
    );
  }
} finally {
  await browser.close();
}
