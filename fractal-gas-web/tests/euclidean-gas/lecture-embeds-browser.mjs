import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";

const base = process.env.LECTURE_DOCS_URL || "http://127.0.0.1:8896";
const manifest = JSON.parse(
  await readFile(
    new URL(
      "../../../docs/_static_theory/gas-demos/manifest.json",
      import.meta.url,
    ),
  ),
);
const output = new URL("../../outputs/lecture-review/", import.meta.url);
await mkdir(output, { recursive: true });
for (const chapter of new Set(manifest.map((entry) => entry.chapter))) {
  const html = await readFile(
    new URL(
      (process.env.LECTURE_DOCS_BUILD_DIR ||
        "../../../docs/_build/theory-site/_build/html/") +
        chapter +
        ".html",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    (html.match(/<figure class="gas-demo feynman-added"/g) || []).length,
    manifest.filter((entry) => entry.chapter === chapter).length,
    chapter,
  );
  for (const entry of manifest.filter((entry) => entry.chapter === chapter)) {
    assert.ok(html.includes('id="gas-demo-' + entry.id + '"'), entry.id);
    assert.ok(
      html.includes('class="gas-demo-title"'),
      entry.id + " updated extension",
    );
  }
}
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
const localErrors = [];
page.on("response", (response) => {
  if (response.url().startsWith(base) && response.status() >= 400)
    localErrors.push(response.url());
});
try {
  await page.goto(
    base + "/fragile/docs/theory/" + manifest[0].chapter + ".html",
    { waitUntil: "domcontentloaded" },
  );
  assert.equal(await page.locator(".gas-demo").count(), 2);
  assert.equal(await page.locator(".gas-demo iframe").count(), 0);
  const first = page.locator("#gas-demo-I-01");
  await first.scrollIntoViewIfNeeded();
  assert.equal(
    await first
      .locator("img")
      .evaluate((image) => image.complete && image.naturalWidth > 0),
    true,
  );
  await first.locator("button").click();
  const firstFrame = first.frameLocator("iframe");
  await firstFrame.locator("#status").filter({ hasText: "Ready" }).waitFor();
  await firstFrame.locator("#step").click();
  await firstFrame.locator("#status").filter({ hasText: "Step 1" }).waitFor();
  assert.equal(await firstFrame.locator("#error").isVisible(), false);
  await firstFrame.locator("#run").click();
  await firstFrame.locator("#run").filter({ hasText: "Pause" }).waitFor();
  await page.evaluate(() => window.scrollTo(0, 0));
  await firstFrame.locator("#run").filter({ hasText: "Run" }).waitFor();
  const second = page.locator("#gas-demo-I-02");
  await second.scrollIntoViewIfNeeded();
  await second.locator("button").click();
  assert.equal(await first.locator("iframe").count(), 0);
  assert.equal(await first.locator(".gas-demo-poster").isVisible(), true);
  await second
    .frameLocator("iframe")
    .locator("#status")
    .filter({ hasText: "Ready" })
    .waitFor();
  assert.equal(await page.locator(".gas-demo iframe").count(), 1);
  await page.screenshot({
    path: new URL("lecture-embed.png", output).pathname,
  });
  await page.evaluate(() =>
    document.documentElement.classList.add("expert-mode"),
  );
  await page.waitForFunction(() => !document.querySelector(".gas-demo iframe"));
  assert.equal(await second.isVisible(), false);
  await page.evaluate(() =>
    document.documentElement.classList.remove("expert-mode"),
  );
  assert.equal(await second.isVisible(), true);
  await second.locator("button").click();
  await second
    .frameLocator("iframe")
    .locator("#status")
    .filter({ hasText: "Ready" })
    .waitFor();
  await second.locator("button").click();
  assert.equal(await page.locator(".gas-demo iframe").count(), 0);
  // With JS disabled, a static figure and working full-view link remain.
  const staticPage = await browser.newPage({ javaScriptEnabled: false });
  await staticPage.goto(
    base + "/docs/theory/" + manifest[0].chapter + ".html",
    { waitUntil: "domcontentloaded" },
  );
  assert.equal(await staticPage.locator(".gas-demo").count(), 2);
  assert.ok(
    await staticPage.locator(".gas-demo a").first().getAttribute("href"),
  );
  const qft = manifest.find(e => e.id === "VI-39");
  await page.goto(base + "/docs/theory/" + qft.chapter + ".html",{waitUntil:"domcontentloaded"});
  const qftFigure=page.locator("#gas-demo-VI-39");
  await qftFigure.scrollIntoViewIfNeeded();
  await qftFigure.locator("button").click();
  const qftFrame=qftFigure.frameLocator("iframe");
  await qftFrame.locator("#status").filter({hasText:"Ready"}).waitFor({timeout:60000});
  assert.equal(await qftFrame.locator("#error").isVisible(),false);
  assert.match(await qftFrame.locator("#title").textContent(),/curvature/i);
  await page.screenshot({path:new URL("partvi-embed.png",output).pathname});
  assert.deepEqual(localErrors, []);
  console.log(
    `${manifest.length} built figures; lazy loading, project prefix, hidden pause, one iframe, Expert Mode, close/reopen and no-JS fallback passed`,
  );
} finally {
  await browser.close();
}
