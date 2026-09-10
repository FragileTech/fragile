// Capture actual lab views. LLM requests use the deterministic offline test provider.
// Start tools/serve-control.py on port 8099 first; no API credentials are needed.
import { chromium } from "playwright";
import { fakeOpenRouter } from "../tests/llm/fixtures.mjs";
import { resolve } from "node:path";
const base = process.env.LANDING_URL || "http://127.0.0.1:8099/fragile/";
const output = resolve(import.meta.dirname, "../web/landing");
const browser = await chromium.launch({
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  await page.goto(new URL("optimization/", base).href);
  await page.waitForFunction(() => window.optimizationReady, null, {
    timeout: 60000,
  });
  await page.locator("#view").selectOption("landscape");
  for (let i = 0; i < 8; i++) {
    const before = await page.locator("#iteration").textContent();
    await page.locator("#step").click();
    await page.waitForFunction(
      (v) => document.querySelector("#iteration").textContent !== v,
      before,
    );
  }
  await page
    .locator(".viewport")
    .screenshot({ path: resolve(output, "optimization.png") });
  console.log("Captured optimization landscape");
  const fake = fakeOpenRouter();
  await page
    .context()
    .route("https://openrouter.ai/api/v1/**", async (route) => {
      const headers = {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET,POST,OPTIONS",
        "access-control-allow-headers": "authorization,content-type",
      };
      if (route.request().method() === "OPTIONS")
        return route.fulfill({ status: 204, headers });
      const response = await fake.fetch(route.request().url(), {
        body: route.request().postData(),
      });
      await route.fulfill({
        status: response.status,
        headers,
        contentType: "application/json",
        body: JSON.stringify(await response.json()),
      });
    });
  await page.goto(new URL("llm/", base).href);
  await page.locator("#api-key").fill("offline-preview");
  await page.locator('[name="algorithm"]').selectOption("wave");
  await page.locator('[name="chunk_tokens"]').fill("2");
  await page.locator('[name="sequence_tokens"]').fill("16");
  await page.locator('[name="iterations"]').fill("6");
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Paused",
    null,
    { timeout: 30000 },
  );
  await page.locator("#tab-analysis").click();
  await page.locator("#run").click();
  await page.waitForFunction(
    () =>
      [
        "Iteration limit reached",
        "Generated-token budget reached",
        "No active branches remain",
      ].includes(document.querySelector("#status").textContent),
    null,
    { timeout: 30000 },
  );
  await page.locator("#analysis-axis").selectOption("iteration");
  await page.locator(".analysis-stage").evaluate((el) => {
    el.style.height = `${el.clientWidth / 2.4}px`;
    el.style.minHeight = "0";
    el.querySelector("#analysis-canvas").style.height = "100%";
  });
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  await page.locator("#analysis-fit").click();
  await page
    .locator(".analysis-stage")
    .screenshot({ path: resolve(output, "llm.png") });
  console.log("Captured LLM demo generation tree");
} finally {
  await browser.close();
}
