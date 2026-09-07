// Run against tools/serve-control.py. The page exercises actual loaded GLBs,
// two viewports, replay, crowd instancing and WebGL context restoration.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";

const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    // Lower only raster resolution on software-GPU CI hosts. CSS geometry and
    // projected-pixel LOD thresholds remain unchanged.
    deviceScaleFactor: Number(process.env.CONTROL_PIXEL_RATIO || 1),
  });
  await page.goto(new URL("tests/visual-style.html", base).href);
  await page
    .waitForFunction(
      () =>
        ["passed", "failed"].includes(
          document.getElementById("result")?.dataset.status,
        ),
      null,
      { timeout: Number(process.env.CONTROL_BROWSER_TIMEOUT_MS || 240000) },
    )
    .catch(async (error) => {
      console.error(
        "Last browser check:",
        await page.locator("#result").textContent(),
      );
      throw error;
    });
  const result = JSON.parse(await page.locator("#result").textContent());
  result.deviceScaleFactor = await page.evaluate(() => devicePixelRatio);
  if (process.env.CONTROL_STYLE_REPORT) {
    await writeFile(
      process.env.CONTROL_STYLE_REPORT,
      `${JSON.stringify(result, null, 2)}\n`,
    );
  }
  console.log(JSON.stringify(result, null, 2));
  assert.equal(result.status, "passed", result.error);
} finally {
  await browser.close();
}
