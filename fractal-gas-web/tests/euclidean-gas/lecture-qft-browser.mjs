import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/partvi-review/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
async function ready() {
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent.startsWith("Ready") &&
      !document.querySelector("#step").disabled,
    null,
    { timeout: 60000 },
  );
  assert.equal(await page.locator("#error").isVisible(), false);
}
async function open(id) {
  await page.goto(`${base}/euclidean-gas/lecture.html?demo=${id}`);
  await ready();
}
try {
  for (const id of [
    "VI-01",
    "VI-08",
    "VI-19",
    "VI-28",
    "VI-34",
    "VI-39",
    "VI-48",
    "VI-52",
    "VI-58",
    "VI-66",
  ]) {
    await open(id);
    assert.ok(await page.locator("#charts svg").count(), id);
    if (id === "VI-48") {
      assert.match(
        await page.locator("#kind").textContent(),
        /Recorded algorithm measurements/,
      );
      assert.equal(
        await page
          .locator('#control-engine_boundary option[value="periodic_box"]')
          .count(),
        0,
      );
    }
    assert.equal(await page.locator('[data-demo^="VI-"]').count(), 66);
    await page.locator("#step").click();
    await page.waitForFunction(() => !document.querySelector("#step").disabled);
    assert.equal(
      await page.locator("#error").isVisible(),
      false,
      await page.locator("#error").textContent(),
    );
    await page.locator("#reset").click();
    await ready();
    await page.screenshot({
      path: new URL(id + ".png", output).pathname,
      fullPage: true,
    });
  }
  await open("VI-08");
  await page.locator("#control-source").selectOption("recorded");
  await ready();
  assert.equal(await page.locator(".scene").isVisible(), true);
  assert.match(await page.locator(".scene svg").textContent(), /x₁, x₂, x₃/);
  assert.equal(await page.locator("#archive").isVisible(), true);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#archive").click(),
  ]);
  const archivePath = new URL("color-archive.json", output).pathname;
  await download.saveAs(archivePath);
  await page.locator("#archive-import").setInputFiles(archivePath);
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Archive validated",
  );
  assert.match(await page.locator("#title").textContent(), /imported archive/);
  await open("VI-28");
  await page.locator(".formula-index summary").click();
  await page.locator(".formula-index input").fill("reflection");
  await page.waitForFunction(
    () => document.querySelectorAll("[data-formula-demo]").length > 0,
  );
  assert.ok(await page.locator('[data-formula-demo="VI-28"]').count());
  const [resultDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#save").click(),
  ]);
  await resultDownload.saveAs(
    new URL("reflection-replay.json", output).pathname,
  );
  await page
    .locator("#native-results")
    .setInputFiles(new URL("reflection-replay.json", output).pathname);
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.includes("native calculations loaded"),
  );
  assert.equal(await page.locator("#error").isVisible(), false);
  assert.match(
    await page.locator("#title").textContent(),
    /Imported native sweep/,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await open("VI-58");
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 2,
    ),
  );
  await page.screenshot({
    path: new URL("mobile-response.png", output).pathname,
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  await writeFile(
    new URL("browser-validation.json", output),
    JSON.stringify({ passed: true, experiments: 66, errors }, null, 2),
  );
  console.log(
    "Part VI browser controls, archive/replay, formula index and mobile layout passed.",
  );
} finally {
  await browser.close();
}
