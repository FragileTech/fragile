// Exercise the actual static files under a GitHub Pages project prefix. This
// server deliberately sends no COOP/COEP headers: the shipped service worker
// must establish isolation, and a browser without service workers must fall back.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { extname, resolve, sep } from "node:path";

const web = fileURLToPath(new URL("../web/", import.meta.url));
const prefix = "/fragile/";
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css",
  ".png": "image/png",
  ".glb": "model/gltf-binary",
};
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(
      new URL(request.url, "http://local").pathname,
    );
    if (!pathname.startsWith(prefix)) throw new Error("Outside project");
    let path = resolve(web, pathname.slice(prefix.length));
    if (!path.startsWith(resolve(web) + sep) && path !== resolve(web))
      throw new Error("Outside web root");
    if ((await stat(path)).isDirectory()) path = resolve(path, "index.html");
    const data = await readFile(path);
    response.writeHead(200, {
      "Content-Type": mime[extname(path)] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    response.end(request.method === "HEAD" ? undefined : data);
  } catch {
    response.writeHead(404);
    response.end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
  const url = `http://127.0.0.1:${server.address().port}${prefix}lab/`;
  for (const serviceWorkers of ["allow", "block"]) {
    const context = await browser.newContext({ serviceWorkers });
    if (serviceWorkers === "block") {
      // Playwright's register stub resolves undefined, unlike a browser that
      // lacks the API. Model an unavailable API rather than that test stub.
      await context.addInitScript(() => {
        Object.defineProperty(navigator, "serviceWorker", { value: undefined });
      });
    }
    const page = await context.newPage();
    const messages = [];
    page.on("console", (m) => messages.push(m.text()));
    try {
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      page.setDefaultTimeout(60000);
      await page.goto(url);
      await page.waitForFunction(
        (isolated) =>
          window.crossOriginIsolated === isolated &&
          !document.getElementById("run").disabled &&
          document
            .getElementById("backend")
            .textContent.startsWith(isolated ? "4 THREADS" : "1 THREAD"),
        serviceWorkers === "allow",
      );
      assert.equal(new URL(page.url()).pathname, `${prefix}lab/`);
      assert.equal(
        new URL(
          await page.getByRole("link", { name: /User guide/ }).getAttribute("href"),
          page.url(),
        ).pathname,
        `${prefix}docs/lab/`,
      );
      assert(
        await page
          .locator(".brand-logo")
          .evaluate((i) => i.complete && i.naturalWidth > 0),
      );
      await page.locator("#scenario").selectOption("racing");
      await page.waitForFunction(
        () => !document.getElementById("run").disabled,
      );
      await page.locator("#step").click();
      await page.waitForFunction(
        () => document.getElementById("tick").textContent === "TICK 000006",
      );
      assert.match(
        await page.locator("#score-note").textContent(),
        /Checkpoint/,
      );
      assert.deepEqual(errors, []);
      console.log(
        `Pages project URL: ${serviceWorkers === "allow" ? "service-worker isolation and threaded planning" : "serial fallback"} passed`,
      );

      const arcade = await context.newPage();
      await arcade.goto(`http://127.0.0.1:${server.address().port}${prefix}`);
      assert.equal(
        new URL(
          await arcade
            .getByRole("link", { name: /Lab user guide/ })
            .getAttribute("href"),
          arcade.url(),
        ).pathname,
        `${prefix}docs/lab/`,
      );
      await arcade.close();
    } catch (error) {
      console.error(
        "Pages smoke test failed:",
        serviceWorkers,
        messages,
        await page.evaluate(() => ({
          url: location.href,
          isolated: crossOriginIsolated,
          controller: navigator.serviceWorker?.controller?.scriptURL,
          backend: document.getElementById("backend")?.textContent,
          status: document.getElementById("status")?.textContent,
        })),
      );
      throw error;
    } finally {
      await context.close();
    }
  }
} finally {
  await browser?.close();
  await new Promise((done) => server.close(done));
}
