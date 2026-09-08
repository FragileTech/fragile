// Serve web/ on CONTROL_TEST_URL (default http://127.0.0.1:8088/lab/).
// ACTION_CONTEXT_ONLY=1 runs the bounded restore diagnostic; ACTION_CUE_CAPTURE=1
// captures only close-up cues. Every benchmark row is saved before the next case.
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
const directory =
  process.env.ACTION_REVIEW_DIR || "/tmp/lab-action-readability";
await mkdir(directory, { recursive: true });
const rows = [];
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 940, height: 740 },
    serviceWorkers: "block",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.exposeFunction("saveReadabilityRow", async (row) => {
    rows.push(row);
    await writeFile(
      `${directory}/progress.json`,
      JSON.stringify({ rows }, null, 2) + "\n",
    );
    console.log(
      `${row.style}/${row.count} animations=${row.animations} guides=${row.guides}: ${row.cosmeticAndEffectsMs}ms cosmetic+effects`,
    );
  });
  await page.goto(
    new URL(
      "tests/action-readability.html",
      process.env.CONTROL_TEST_URL || "http://127.0.0.1:8088/lab/",
    ).href,
  );
  await page.waitForFunction(() => !!window.readabilityFixture, {
    timeout: 30000,
  });
  if (process.env.ACTION_CONTEXT_ONLY) {
    const result = await page.evaluate(() => readabilityFixture.contextCheck());
    await writeFile(
      `${directory}/context.json`,
      JSON.stringify(result, null, 2) + "\n",
    );
    console.log(JSON.stringify(result));
  }
  if (
    !process.env.ACTION_SCREENSHOTS_ONLY &&
    !process.env.ACTION_CONTEXT_ONLY &&
    !process.env.ACTION_CUE_CAPTURE
  ) {
    let timer;
    const result = await Promise.race([
      page.evaluate(() => readabilityFixture.run()),
      new Promise((_, reject) => {
        timer = setTimeout(
          () =>
            reject(
              new Error(
                "QA exceeded 180s; completed rows saved in progress.json",
              ),
            ),
          180000,
        );
      }),
    ]).finally(() => clearTimeout(timer));
    assert.equal(result.rows.length, 32);
    await writeFile(
      `${directory}/report.json`,
      JSON.stringify(result, null, 2) + "\n",
    );
    console.log(`${result.checks.length} checks passed; ${result.gpu}`);
  }
  if (process.env.ACTION_CUE_CAPTURE) {
    for (const style of process.env.ACTION_CUE_STYLE
      ? [process.env.ACTION_CUE_STYLE]
      : ["futuristic", "steampunk"])
      for (const kind of process.env.ACTION_CUE_KIND
        ? [process.env.ACTION_CUE_KIND]
        : ["rocket", "drone", "center-jet", "kart", "harvester"])
        for (const lod of kind === "kart" || kind === "harvester"
          ? ["high"]
          : ["high", "low"]) {
          await page.evaluate(
            (options) => readabilityFixture.showcaseCue(options),
            { style, kind, lod },
          );
          if (process.env.ACTION_CANVAS_CAPTURE) {
            const data = await page.evaluate(() => {
              const r = readabilityFixture.renderer;
              r.renderer.setPixelRatio(1);
              r.renderer.render(r.world, r.camera);
              return r.canvas.toDataURL("image/png");
            });
            await writeFile(
              `${directory}/${style}-${kind}-${lod}.png`,
              Buffer.from(data.split(",")[1], "base64"),
            );
          } else
            await page
              .locator("#view")
              .screenshot({ path: `${directory}/${style}-${kind}-${lod}.png` });
          console.log(`Captured ${style}/${kind}/${lod}`);
        }
  }
  if (process.env.ACTION_SCREENSHOTS_ONLY || process.env.ACTION_CAPTURE) {
    for (const style of ["futuristic", "steampunk"]) {
      await page.evaluate((s) => readabilityFixture.showcase(s), style);
      await page
        .locator("#view")
        .screenshot({ path: `${directory}/${style}.png` });
    }
  }
  assert.equal(errors.length, 0, errors.join("\n"));
  console.log(`Review: ${directory}`);
} finally {
  await browser.close();
}
