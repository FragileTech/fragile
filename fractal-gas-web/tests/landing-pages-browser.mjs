// Verify the shipped page under the Pages project prefix, without isolation headers.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { resolve, extname, sep } from "node:path";
import { chromium } from "playwright";
const root = resolve(import.meta.dirname, "../web");
const docs = resolve(import.meta.dirname, "../../docs/_build/html");
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".json": "application/json",
};
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://local");
    if (url.pathname === "/fragile") {
      res.writeHead(301, { Location: "/fragile/" });
      return res.end();
    }
    if (!url.pathname.startsWith("/fragile/")) throw Error("Outside project");
    let relative = decodeURIComponent(url.pathname.slice("/fragile/".length));
    const mount = relative.startsWith("docs/") ? docs : root;
    if (mount === docs) relative = relative.slice(5);
    let file = resolve(mount, relative);
    if (file !== mount && !file.startsWith(mount + sep))
      throw Error("Outside root");
    if ((await stat(file)).isDirectory()) file = resolve(file, "index.html");
    const data = await readFile(file);
    res.writeHead(200, {
      "Content-Type": mime[extname(file)] || "application/octet-stream",
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const base = `http://127.0.0.1:${server.address().port}/fragile/`;
let browser;
try {
  browser = await chromium.launch({
    args: ["--no-sandbox", "--use-angle=swiftshader"],
  });
  for (const javaScriptEnabled of [true, false]) {
    const context = await browser.newContext({ javaScriptEnabled });
    const page = await context.newPage();
    const requests = [],
      errors = [];
    page.on("request", (req) => requests.push(req.url()));
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("response", (res) => {
      if (res.status() >= 400) errors.push(`${res.status()} ${res.url()}`);
    });
    for (const width of [1440, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(base.slice(0, -1));
      await page.locator(".preview img").last().waitFor();
      assert.equal(await page.locator(".lab").count(), 4);
      assert(
        await page
          .locator(".preview img")
          .evaluateAll((images) =>
            images.every((img) => img.complete && img.naturalWidth > 0),
          ),
      );
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `Overflow at ${width}`,
      );
      assert.deepEqual(
        await page
          .locator(".lab-link")
          .evaluateAll((links) =>
            links.map((link) => link.getAttribute("href")),
          ),
        ["arcade.html", "lab/", "optimization/", "llm/"],
      );
      assert.equal(
        await page.locator('.site-header nav a[href="euclidean-gas/"]').count(),
        1,
      );
      if (
        process.env.CAPTURE_LANDING_SCREENSHOTS === "1" &&
        javaScriptEnabled &&
        [1440, 768, 390].includes(width)
      )
        await page.screenshot({
          path: `/tmp/fragile-landing-${width}.png`,
          fullPage: true,
        });
    }
    assert.equal(await page.locator("script").count(), 0);
    assert(
      !requests.some((url) => /\.wasm|worker|\/engine\//.test(url)),
      "Landing starts an engine or worker",
    );
    if (javaScriptEnabled) {
      assert.equal(
        await page.evaluate(
          async () => (await navigator.serviceWorker.getRegistrations()).length,
        ),
        0,
      );
      await page.keyboard.press("Tab");
      assert.equal(
        await page.locator(":focus").textContent(),
        "Skip to the labs",
      );
    }
    assert.deepEqual(errors, []);
    await context.close();
  }
  // Lab entry files and brand navigation must remain usable without running engines.
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  for (const route of [
    "arcade.html",
    "lab/",
    "optimization/",
    "euclidean-gas/",
    "llm/",
  ]) {
    assert.equal((await page.goto(new URL(route, base).href)).status(), 200);
    await page.locator(".app-brand").click();
    assert.equal(page.url(), base);
  }
  await context.close();
  // Optional complete local bundle check: exercises the real Arcade startup and worker.
  if (process.env.LANDING_ARCADE_SMOKE === "1") {
    const ctx = await browser.newContext();
    const arcade = await ctx.newPage();
    arcade.on("pageerror", (e) => console.log("Arcade error:", e.message));
    arcade.on("console", (msg) => {
      if (msg.type() === "error") console.log("Arcade console:", msg.text());
    });
    await arcade.goto(new URL("arcade.html", base).href);
    await arcade.waitForFunction(() => crossOriginIsolated, null, {
      timeout: 60000,
    });
    await arcade.waitForFunction(
      () =>
        /Ready|Enter the ROM password/.test(
          document.querySelector("#status").textContent,
        ),
      null,
      { timeout: 60000 },
    );
    console.log(
      "Arcade startup:",
      await arcade.locator("#status").textContent(),
    );
    const requests = [];
    arcade.on("request", (req) => requests.push(req.url()));
    await arcade.locator(".app-brand").click();
    await arcade
      .getByRole("heading", { name: "Explore intelligent search." })
      .waitFor();
    assert.equal(arcade.url(), base);
    assert(
      !requests.some((url) => /\.wasm|\/engine\//.test(url)),
      "Returning home starts an engine",
    );
    assert(await arcade.evaluate(() => !!navigator.serviceWorker.controller));
    assert(
      await arcade
        .locator(".preview img")
        .evaluateAll((images) =>
          images.every((img) => img.complete && img.naturalWidth > 0),
        ),
    );
    await ctx.close();
  }
  console.log(
    "Landing page passed: five widths, no JavaScript, no engines, five lab routes, and home navigation.",
  );
} finally {
  await browser?.close();
  await new Promise((done) => server.close(done));
}
