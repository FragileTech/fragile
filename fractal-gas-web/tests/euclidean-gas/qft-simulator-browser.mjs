// Browser checks of the QFT Simulator against the compiled Rust engine: the
// variant registry, the availability Rust computes for a configuration, a short
// Einstein-Hilbert run and a re-analysis that does not touch the gas.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";

const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/qft-simulator-review/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
page.setDefaultTimeout(60000);
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});

// Two catalog rows keep the measurement and every re-analysis cheap.
const CHANNELS = [
  "meson/scalar/standard/distance",
  "meson/pseudoscalar/standard/distance",
];
// Rust measures whole specifications, so each ticked row also brings the other
// element kind of its specification into the measurement; only the ticked rows
// are analysed.
const MEASURED = [
  "meson/scalar/standard/distance",
  "meson/scalar/standard/cloning",
  "meson/pseudoscalar/standard/distance",
  "meson/pseudoscalar/standard/cloning",
];
// The sentence `physics/spectroscopy` reports for a run whose viscous force is
// identically zero. The page shows it as Rust wrote it.
const COLOUR_REASON =
  "viscous force is identically zero: configure qft.viscosity or " +
  "qft.graph_viscosity, or select an explicit RecordedField colour source";
// The canonical Euclidean Gas runs in two dimensions, so a triplet operator is
// out of reach for a second, independent reason.
const DIMENSION_REASON = "defined in 3 position dimensions; the run has 2";
const shot = (name) =>
  page.screenshot({ path: new URL(name, output).pathname, fullPage: true });
const text = (selector) => page.locator(selector).innerText();
const healthy = async (label) =>
  assert.equal(
    await page.locator("#error").isVisible(),
    false,
    label + ": " + (await text("#error")),
  );
const availabilityComputed = () =>
  page.waitForFunction(
    () =>
      document.querySelector("#setup-status").textContent ===
      "Availability computed by Rust",
    null,
    { timeout: 120000 },
  );
const analysed = () =>
  page.waitForFunction(
    () =>
      document
        .querySelector("#analysis-status")
        .textContent.startsWith("Analysed"),
    null,
    { timeout: 300000 },
  );
// The body rows of one rendered table; a container can hold several.
const rows = (selector, index = 0) =>
  page.$$eval(
    selector + " table",
    (tables, index) =>
      [...(tables[index]?.tBodies[0]?.rows ?? [])].map((row) =>
        [...row.cells].map((cell) => cell.textContent),
      ),
    index,
  );
// The notes of one panel, as the owner Rust named and the sentence it wrote.
const notes = (selector) =>
  page.$$eval(selector + " .notes li", (items) =>
    items.map((item) => {
      const owner = item.querySelector(".note-owner");
      return {
        owner: owner?.textContent ?? null,
        text: item.textContent.slice(owner?.textContent.length ?? 0),
      };
    }),
  );
// The position of a named column of a rendered table.
const column = async (selector, name, index = 0) => {
  const headers = await page.$$eval(
    selector + " table",
    (tables, index) =>
      [...(tables[index]?.tHead?.rows[0]?.cells ?? [])].map(
        (cell) => cell.textContent,
      ),
    index,
  );
  assert.ok(headers.includes(name), name + " is not a column of " + selector);
  return headers.indexOf(name);
};
// A colour channel is disabled exactly when the selected variant records no
// viscous force, so it tracks the availability Rust computed for the form.
const colourChannel = (blocked) =>
  page.waitForFunction(
    (blocked) =>
      [...document.querySelectorAll('#channels input[name="channel"]')].find(
        (box) => box.value === "meson/scalar/standard/distance",
      )?.disabled === blocked,
    blocked,
    { timeout: 120000 },
  );
const channels = () =>
  page.$$eval("#channels .channel", (list) =>
    list.map((item) => ({
      id: item.querySelector("code").textContent,
      blocked: item.querySelector("input").disabled,
      reason: item.querySelector(".channel-reason")?.textContent ?? null,
    })),
  );
const finite = (values) =>
  values.filter((value) => Number.isFinite(Number(value.replace(/,/g, ""))));

try {
  // ------------------------------------------------ the page and the registry
  await page.goto(`${base}/euclidean-gas/qft.html`);
  await page.waitForFunction(
    () => document.querySelector("#engine").textContent === "Rust engine ready",
    null,
    { timeout: 180000 },
  );
  await availabilityComputed();
  await healthy("load");
  const variants = await page.$$eval("#variant option", (options) =>
    options.map((option) => ({
      name: option.value,
      title: option.textContent,
      disabled: option.disabled,
    })),
  );
  assert.deepEqual(
    variants.map((variant) => variant.name),
    [
      "euclidean",
      "viscous_euclidean",
      "einstein_hilbert",
      "geometric",
      "latent",
      "environment",
    ],
    "the selector is populated from the Rust variant registry",
  );
  assert.deepEqual(
    variants.filter((variant) => variant.disabled).map((v) => v.name),
    ["geometric", "latent", "environment"],
    "a variant the crate does not implement cannot be selected",
  );
  for (const variant of variants.filter((v) => v.disabled))
    assert.match(variant.title, /\(book only\)$/);
  assert.equal(await page.locator("#variant-list dt").count(), variants.length);
  assert.ok(
    (await page.locator("#channels .channel").count()) > 100,
    "the channel catalog comes from Rust",
  );
  await shot("01-setup.png");

  // --------------------------------- availability of the canonical Euclidean Gas
  await page.selectOption("#variant", "euclidean");
  // The list is rebuilt when Rust has answered for the variant just selected.
  await colourChannel(true);
  await availabilityComputed();
  await healthy("euclidean availability");
  const euclidean = await channels();
  const blocked = euclidean.filter((channel) => channel.blocked);
  assert.ok(blocked.length > 0, "the colour channels are unavailable");
  for (const channel of blocked)
    assert.match(
      channel.reason,
      /^Unavailable: \S.*\S$/,
      channel.id + " is disabled without saying why",
    );
  // Two different engine sentences, each copied into the list as Rust wrote it.
  for (const [id, reason] of [
    ["meson/scalar/standard/distance", COLOUR_REASON],
    ["vector/vector/full/raw/distance", COLOUR_REASON],
    ["baryon/complex/triplet", DIMENSION_REASON],
  ])
    assert.equal(
      euclidean.find((channel) => channel.id === id)?.reason,
      "Unavailable: " + reason,
      id + " must say why in the engine's own words",
    );
  for (const id of ["fitness_phase/site", "electroweak_mixed/triplet"])
    assert.equal(
      euclidean.find((channel) => channel.id === id)?.blocked,
      false,
      id + " does not need the viscous force",
    );
  await shot("02-euclidean-availability.png");

  // ------------------------------------------------------------- a short run
  await page.selectOption("#variant", "einstein_hilbert");
  await colourChannel(false);
  await availabilityComputed();
  await page.evaluate((wanted) => {
    const form = document.querySelector("#setup");
    form.elements.walkers.value = "24";
    form.elements.steps.value = "96";
    form.elements.replicas.value = "1";
    form.elements.walkers.dispatchEvent(new Event("change", { bubbles: true }));
    const boxes = [...form.querySelectorAll('[name="channel"]')];
    for (const box of boxes) box.checked = wanted.includes(box.value);
    boxes[0].dispatchEvent(new Event("change", { bubbles: true }));
  }, CHANNELS);
  await page.waitForTimeout(500);
  await availabilityComputed();
  assert.equal(
    await page.locator('#channels input[name="channel"]:checked').count(),
    CHANNELS.length,
  );
  await page.locator("#create").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent === "Complete at step 96",
    null,
    { timeout: 300000 },
  );
  await analysed();
  await healthy("run");
  assert.equal(
    await page.locator("#progress").evaluate((bar) => bar.value),
    96,
    "the progress bar is at the last step",
  );
  const coverage = await rows("#coverage");
  assert.deepEqual(
    coverage.map((row) => row[0]),
    MEASURED,
    "one coverage row per measured channel",
  );
  const counts = Object.fromEntries(
    await Promise.all(
      ["Estimator", "Frames", "Valid"].map(async (name) => [
        name,
        await column("#coverage", name),
      ]),
    ),
  );
  for (const row of coverage) {
    assert.ok(Number(row[counts.Frames]) > 0, row[0] + " measured no frame");
    assert.ok(
      Number(row[counts.Valid].replace(/,/g, "")) > 0,
      row[0] + " counted no valid element",
    );
  }
  // A measured channel the live estimator cannot report is not silently
  // dropped: it keeps its coverage row, shows no estimator, and the sentence
  // Rust attached to it is listed under the session's notes.
  const drawable = await page.$$eval(
    '#live-select input[name="live"]',
    (list) => list.map((box) => box.value),
  );
  const silent = coverage.filter((row) => row[counts.Estimator] === "—");
  assert.deepEqual(
    coverage.filter((row) => row[counts.Estimator] !== "—").map((r) => r[0]),
    drawable,
    "the live charts offer exactly the channels Rust gave a live estimator",
  );
  const runNotes = await notes("#run-notes");
  for (const row of silent)
    assert.ok(
      runNotes.some((note) => note.owner === row[0] && note.text.length > 20),
      row[0] + " has no live correlator and no note saying why",
    );
  const live = await page.$$eval("#run-charts .chart h3", (heads) =>
    heads.map((head) => head.textContent),
  );
  assert.deepEqual(live.slice(0, 2), [
    "Executed walker positions",
    "Live C(τ)",
  ]);
  assert.ok(live.includes("Live effective decay rate"));
  const drawn = await page.$$eval(
    "#run-charts .chart:nth-of-type(2) svg path",
    (paths) => paths.map((path) => path.getAttribute("d") || ""),
  );
  assert.ok(
    drawn.some((d) => /[0-9]/.test(d)),
    "the live correlator is drawn from real numbers",
  );
  await shot("03-run.png");

  // ------------------------------- a re-analysis that never advances the gas
  await page.locator("#tab-correlators").click();
  await page.waitForTimeout(200);
  const before = {
    status: await text("#status"),
    progress: await page.locator("#progress").evaluate((bar) => bar.value),
    replicas: await rows("#replicas"),
    correlator: await rows("#correlator-table"),
    samples: await rows("#samples-table"),
    analysis: await text("#analysis-status"),
  };
  const value = await column("#correlator-table", "C(τ)");
  assert.ok(
    finite(before.correlator.map((row) => row[value])).length > 4,
    "C(τ) is reported as numbers",
  );
  await page.locator('#analysis [name="connected"]').click();
  await analysed();
  await healthy("re-analysis");
  const after = {
    status: await text("#status"),
    progress: await page.locator("#progress").evaluate((bar) => bar.value),
    replicas: await rows("#replicas"),
    correlator: await rows("#correlator-table"),
    samples: await rows("#samples-table"),
    analysis: await text("#analysis-status"),
  };
  assert.equal(after.status, before.status, "the gas did not advance");
  assert.equal(after.progress, before.progress);
  assert.deepEqual(after.replicas, before.replicas, "no replica moved");
  assert.equal(
    after.analysis,
    before.analysis,
    "the same measured frames were re-analysed",
  );
  assert.notDeepEqual(
    after.correlator,
    before.correlator,
    "subtracting the disconnected part changes C(τ)",
  );
  assert.deepEqual(
    after.correlator.map((row) => row[0]),
    before.correlator.map((row) => row[0]),
    "over the same lags",
  );
  const connected = await column("#samples-table", "Connected");
  assert.deepEqual(
    after.samples.map((row) => row[connected]),
    after.samples.map(() => "no"),
    "the report states the subtraction Rust applied",
  );
  await shot("04-reanalysis.png");

  // -------------------------------------------------- fits read like a report
  await page.locator("#tab-fits").click();
  await page.waitForTimeout(200);
  const fits = await rows("#fit-table");
  const cell = Object.fromEntries(
    await Promise.all(
      ["Channel", "Rate", "χ²", "dof", "Q", "Window (frames)", "No signal"].map(
        async (name) => [name, await column("#fit-table", name)],
      ),
    ),
  );
  assert.deepEqual(
    [...new Set(fits.map((row) => row[cell.Channel]))],
    CHANNELS,
    "the analysis covers the ticked rows, not the whole measurement",
  );
  const fitted = fits.filter((row) => finite([row[cell.Rate]]).length);
  assert.ok(fitted.length > 0, "at least one channel is fitted");
  for (const row of fitted)
    for (const name of ["χ²", "dof", "Q", "Window (frames)"])
      assert.notEqual(
        row[cell[name]],
        "—",
        name + " is missing from a fitted row: " + row.join(" | "),
      );
  // A channel without a rate carries the reason Rust reported instead.
  for (const row of fits.filter((row) => !finite([row[cell.Rate]]).length))
    assert.notEqual(
      row[cell["No signal"]],
      "—",
      "an unfitted row says nothing: " + row.join(" | "),
    );
  await shot("05-fits.png");

  assert.deepEqual(errors, []);
  await writeFile(
    new URL("qft-simulator-browser.json", output),
    JSON.stringify(
      {
        passed: true,
        variants: variants.map((variant) => variant.name),
        bookOnly: variants.filter((v) => v.disabled).map((v) => v.name),
        blockedWithoutColour: blocked.length,
        measuredChannels: MEASURED,
        liveChannels: drawable,
        explainedWithoutLiveCorrelator: silent.map((row) => row[0]),
        analysedChannels: CHANNELS,
        steps: 96,
        walkers: 24,
        reanalysedWithoutRunning: true,
        errors,
      },
      null,
      2,
    ),
  );
  console.log(
    "QFT Simulator browser checks passed: the Rust variant registry, the " +
      "colour-state reasons of the canonical Euclidean Gas, a 96-step " +
      "Einstein-Hilbert run whose live correlators and silent channels are " +
      "both accounted for, and a re-analysis that leaves the gas at step 96.",
  );
} finally {
  await browser.close();
}
