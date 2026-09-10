import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import assert from "node:assert/strict";
import { fakeOpenRouter } from "./fixtures.mjs";
import {
  parseBenchmark,
  collectRuns,
  processBenchmark,
} from "../../web/llm/benchmark-data.js";
const root = resolve(import.meta.dirname, "../../web");
const server = createServer(async (req, res) => {
  try {
    const path = resolve(
      root,
      "." +
        decodeURIComponent(req.url.split("?")[0]) +
        (req.url.endsWith("/") ? "index.html" : ""),
    );
    if (!path.startsWith(root + "/")) throw Error();
    res.setHeader(
      "Content-Type",
      {
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".css": "text/css",
        ".html": "text/html",
      }[extname(path)] ?? "application/octet-stream",
    );
    res.end(await readFile(path));
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const fake = fakeOpenRouter();
let release,
  gate,
  gated = 0;
await context.route("https://openrouter.ai/api/v1/**", async (route) => {
  const r = route.request(),
    headers = {
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "authorization,content-type",
    };
  if (r.method() === "OPTIONS") {
    await route.fulfill({ status: 204, headers });
    return;
  }
  if (
    gate &&
    r.url().endsWith("/chat/completions") &&
    !JSON.parse(r.postData()).messages[0].content.startsWith(
      "Copy the following text exactly",
    ) &&
    !JSON.parse(r.postData()).messages[0].content.startsWith(
      "Continue this list",
    )
  ) {
    gated++;
    await gate;
  }
  const response = await fake.fetch(r.url(), { body: r.postData() });
  await route
    .fulfill({
      status: response.status,
      headers,
      contentType: "application/json",
      body: JSON.stringify(await response.json()),
    })
    .catch(() => {});
});
const url = `http://127.0.0.1:${server.address().port}/llm/`;
async function ready() {
  await page.waitForFunction(
    () => !document.querySelector("#benchmark-start").disabled,
  );
}
async function download() {
  const promise = page.waitForEvent("download");
  await page.locator("#benchmark-export").click();
  return readFile(await (await promise).path(), "utf8");
}
try {
  await page.goto(url);
  await page.locator("#tab-generation").click();
  await page.locator("#api-key").fill("browser-benchmark-secret");
  await page
    .locator("summary")
    .filter({ hasText: "Algorithm settings" })
    .click();
  for (const [field, value] of Object.entries({
    walkers: 3,
    chunk_tokens: 2,
    sequence_tokens: 5,
    iterations: 4,
    max_walkers: 12,
    concurrency: 2,
  }))
    await page.locator(`[name="${field}"]`).fill(String(value));
  await page.locator("#tab-benchmark").click();
  await page.locator("#tab-benchmark").click();
  await page.locator("#benchmark-comparison").selectOption("both");
  await page.locator("#benchmark-repetitions").fill("2");
  gate = new Promise((resolve) => {
    release = resolve;
  });
  await page.locator("#benchmark-start").click();
  await page.waitForFunction(() =>
    document.querySelector("#benchmark-status").textContent.includes("fractal"),
  );
  assert.ok(await page.locator("#run").isDisabled());
  await page.locator("#benchmark-pause").click();
  gate = null;
  release();
  await page.waitForFunction(() =>
    document
      .querySelector("#benchmark-status")
      .textContent.startsWith("Paused —"),
  );
  const count = fake.requests.length;
  await page.waitForTimeout(100);
  assert.equal(fake.requests.length, count);
  const id = await page.locator("#benchmark-saved").inputValue();
  // A second tab cannot acquire the writer lock, even at a paused boundary.
  const second = await context.newPage();
  await second.goto(url);
  await second.locator("#api-key").fill("browser-benchmark-secret");
  await second.locator("#tab-benchmark").click();
  await second
    .locator(`#benchmark-saved option[value="${id}"]`)
    .waitFor({ state: "attached" });
  await second.locator("#tab-benchmark").click();
  await second.locator("#benchmark-saved").selectOption(id);
  await second.locator("#benchmark-retry").click();
  await second.waitForFunction(() =>
    document
      .querySelector("#benchmark-status")
      .textContent.includes("already running"),
  );
  await second.close();
  await page.locator("#benchmark-continue").click();
  await ready();
  assert.match(
    await page.locator("#benchmark-status").textContent(),
    /Completed/,
  );
  const text = await download(),
    archive = parseBenchmark(text);
  assert.ok(!text.includes("browser-benchmark-secret"));
  assert.equal(
    [...collectRuns(archive.manifest, archive.events).runs.values()].filter(
      (r) => r.status === "completed",
    ).length,
    8,
  );
  const expected = processBenchmark(archive.manifest, archive.events);
  await page.reload();
  await page.locator("#tab-benchmark").click();
  await page.locator("#tab-benchmark").click();
  await page
    .locator(`#benchmark-saved option[value="${id}"]`)
    .waitFor({ state: "attached" });
  await page.locator("#benchmark-saved").selectOption(id);
  assert.equal(await download(), text);
  // Offline import in a separate browser profile does not contact the provider.
  const offlineContext = await browser.newContext();
  await offlineContext.route("https://openrouter.ai/**", (route) =>
    route.abort(),
  );
  const offline = await offlineContext.newPage();
  await offline.goto(url);
  await offline.locator("#tab-benchmark").click();
  await offline.locator("#benchmark-file").setInputFiles({
    name: "archive.fgllmbench",
    mimeType: "application/x-ndjson",
    buffer: Buffer.from(text),
  });
  await offline.waitForFunction(() =>
    document.querySelector("#benchmark-status").textContent.includes("8/8"),
  );
  const tables = await offline.evaluate(async (id) => {
    const { BrowserBenchmarkStore } = await import("./benchmark-store.js");
    const { processBenchmark } = await import("./benchmark-data.js");
    const store = await BrowserBenchmarkStore.open(id);
    return processBenchmark(store.manifest, await store.readEvents());
  }, id);
  assert.deepEqual(tables, expected);
  await offlineContext.close();
  // Reload while an iteration is pending; retry creates another Fractal attempt.
  await page.locator("#tab-generation").click();
  await page.locator("#api-key").fill("browser-benchmark-secret");
  await page.locator("#tab-benchmark").click();
  gate = new Promise((resolve) => {
    release = resolve;
  });
  gated = 0;
  await page.locator("#benchmark-start").click();
  await page.waitForFunction(() =>
    document.querySelector("#benchmark-status").textContent.includes("fractal"),
  );
  const interruptedId = await page.locator("#benchmark-saved").inputValue();
  await page.reload();
  await page.locator("#tab-generation").click();
  gate = null;
  release();
  await page.locator("#api-key").fill("browser-benchmark-secret");
  await page.locator("#tab-benchmark").click();
  await page
    .locator(`#benchmark-saved option[value="${interruptedId}"]`)
    .waitFor({ state: "attached" });
  await page.locator("#benchmark-saved").selectOption(interruptedId);
  await page.locator("#benchmark-retry").click();
  await ready();
  const recovered = parseBenchmark(await download());
  const fractal = [
    ...collectRuns(recovered.manifest, recovered.events).runs.values(),
  ].filter((r) => r.method === "fractal" && r.trial === 0);
  assert.equal(fractal.length, 2);
  assert.equal(fractal[0].status, "interrupted");
  assert.equal(fractal[1].status, "completed");
  // Stop remains usable during provider work and exports a valid partial archive.
  gate = new Promise((resolve) => {
    release = resolve;
  });
  await page.locator("#benchmark-start").click();
  await page.waitForFunction(() =>
    document.querySelector("#benchmark-status").textContent.includes("fractal"),
  );
  await page.locator("#benchmark-stop").click();
  gate = null;
  release();
  await ready();
  parseBenchmark(await download());
  assert.deepEqual(errors, []);
  console.log(
    "Benchmark browser: generation, pause/continue, stop, reload/retry, writer locks, autosave and offline round trip passed",
  );
} finally {
  release?.();
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
