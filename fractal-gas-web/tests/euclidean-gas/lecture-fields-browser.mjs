import { chromium } from "playwright";
import assert from "node:assert/strict";
const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
async function ready() {
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Ready") && !document.querySelector("#step").disabled, null, { timeout: 180000 });
}
const results = [];
try {
  for (const observable of ["density", "momentum", "stress", "energy", "phase_space"]) {
    await page.goto(`${base}/euclidean-gas/lecture.html?demo=VI-51`);
    await ready();
    const control = page.locator("#control-field_observable");
    if (!await control.isVisible()) await page.locator(".advanced summary").click();
    await control.selectOption(observable);
    await ready();
    await page.locator("#step").click();
    await page.waitForFunction(() => !document.querySelector("#step").disabled, null, { timeout: 180000 });
    assert.equal(await page.locator("#error").isVisible(), false);
    const result = JSON.parse(await page.locator(".calculation-details pre").textContent());
    for (const key of ["weak_field_equation", "clone_field_equation"]) {
      assert.equal(result.details[key].status, "available", `${observable}: ${key}`);
      assert.ok(result.details[key].analyzed_updates >= 8);
      assert.equal(result.details[key].reports.length, result.details[key].analyzed_updates);
      assert.equal(result.details[key].unsupported.length, 0);
    }
    assert.ok(result.plots.some(p => p.title === "Conditional field equation: cloning term"));
    results.push({ observable, updates: result.details.weak_field_equation.analyzed_updates, status: "available" });
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify(results, null, 2));
} finally { await browser.close(); }
