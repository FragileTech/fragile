// Serve web/ locally, then run with CONTROL_TEST_URL=http://127.0.0.1:8088/lab/.
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
const directory =
  process.env.ANIMATION_REVIEW_DIR || "/tmp/lab-animation-review";
await mkdir(directory, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 940, height: 720 },
    serviceWorkers: "block",
  });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "log") console.log(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(
    new URL(
      "tests/animations.html",
      process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/",
    ).href,
  );
  await page.waitForFunction(() => !!window.animationFixture, {
    timeout: 30000,
  });
  for (const style of ["futuristic", "steampunk"]) {
    await page.evaluate((style) => animationFixture.showcase(style), style);
    await page
      .locator("#view")
      .screenshot({ path: `${directory}/${style}.png` });
  }
  if (!process.env.ANIMATION_SCREENSHOTS_ONLY) {
    try {
      const result = await page.evaluate(
        (contextLoss) => animationFixture.run({ contextLoss }),
        !!process.env.ANIMATION_CONTEXT_LOSS,
      );
      assert.equal(result.rows.length, 8);
      console.log(
        JSON.stringify(
          { checks: result.checks.length, rows: result.rows, directory },
          null,
          2,
        ),
      );
      await writeFile(
        `${directory}/measurements.json`,
        JSON.stringify(result, null, 2) + "\n",
      );
    } catch (error) {
      const progress = await page.evaluate(() => window.animationProgress);
      await writeFile(
        `${directory}/incomplete-measurements.json`,
        JSON.stringify(progress, null, 2) + "\n",
      );
      throw error;
    }
  }
  assert.equal(errors.length, 0, errors.join("\n"));
  console.log(`Screenshots saved to ${directory}`);
} finally {
  await browser.close();
}
