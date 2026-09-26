import { chromium } from "playwright";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
try {
  await page.goto(
    (process.env.OPTIMIZATION_TEST_URL ||
      "http://127.0.0.1:8081/optimization/") + "populations.html",
  );
  await page.locator("#walkers").fill("20");
  await page.locator("#members details").nth(1).locator("summary").click();
  await page
    .locator("#members details")
    .nth(1)
    .locator('[name="perturbation_std"]')
    .fill("0.3");
  await page.locator("#reset").click();
  await page.waitForFunction(
    () => document.querySelector("#message").textContent.includes("ready"),
    null,
    { timeout: 120000 },
  );
  assert.equal(await page.locator("#statistics tr").count(), 4);
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.querySelector("#round-label").textContent === "1",
  );
  assert.equal(await page.locator("#provenance li").count(), 20);
  await page.screenshot({
    path: "/tmp/fractal-populations-desktop.png",
    fullPage: true,
  });
  const resolved = await page.evaluate(async () => {
    const { FractalPopulation } = await import("/optimization/populations.js");
    const p = new FractalPopulation();
    try {
      const r = await p.create({
        defaults: {
          benchmark: "quadratic",
          walkers: 16,
          elites: 5,
          boundary: "periodic",
        },
        members: Array.from({ length: 7 }, (_, i) => ({
          id: `s-${i}`,
          settings: { reward_coef: i / 4 },
        })),
        concurrency: 2,
        max_evaluations: 10000,
      });
      await p.step();
      return r.config.members.map((m) => m.settings.reward_coef);
    } finally {
      p.dispose();
    }
  });
  assert.deepEqual(resolved, [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]);
  await page.locator("#live-scale").fill("0.07");
  await page.locator("#apply").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#message").textContent === "Swarm scale updated.",
  );
  await page.locator("#swarm-count").fill("2");
  await page.locator("#swarm-count").dispatchEvent("change");
  assert.equal(await page.locator("#members details").count(), 2);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "/tmp/fractal-populations-mobile.png",
    fullPage: true,
  });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
