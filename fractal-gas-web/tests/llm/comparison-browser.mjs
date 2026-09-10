import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import assert from "node:assert/strict";
import { BenchmarkRunner } from "../../web/llm/benchmark.js";
import {
  manifest,
  exportBenchmark,
  collectRuns,
  fractalRecording,
} from "../../web/llm/benchmark-data.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import { parseReport, processReport } from "../../web/llm/comparison-report.js";
import { CRITERIA } from "../../web/llm/grading.js";
import { fakeOpenRouter } from "./fixtures.mjs";
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
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ headless: true }),
  context = await browser.newContext(),
  page = await context.newPage(),
  errors = [];
page.on("pageerror", (e) => {
  errors.push(e.message);
  console.error(e);
});
const store = new MemoryBenchmarkStore(
  manifest({
    comparison: "both",
    repetitions: 2,
    config: {
      walkers: 2,
      chunk_tokens: 2,
      sequence_tokens: 4,
      iterations: 3,
      max_walkers: 8,
    },
  }),
);
await new BenchmarkRunner(store, "fixture-key", {
  fetchImpl: fakeOpenRouter().fetch,
}).run();
const archive = exportBenchmark(store.manifest, store.events),
  record = fractalRecording(
    [...collectRuns(store.manifest, store.events).runs.values()][0],
  );
const url = `http://127.0.0.1:${server.address().port}/llm/`,
  judgeCalls = [];
await context.route("https://openrouter.ai/api/v1/**", async (route) => {
  const req = route.request();
  if (req.method() === "OPTIONS") {
    await route.fulfill({
      status: 204,
      headers: {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET,POST,OPTIONS",
        "access-control-allow-headers": "authorization,content-type",
      },
    });
    return;
  }
  let data;
  if (req.url().endsWith("/models"))
    data = {
      data: [
        { id: "google/gemini-3.8-flash", created: 100 },
        { id: "google/gemini-9-flash-preview", created: 200 },
      ],
    };
  else if (req.url().includes("gemini-flash-latest/endpoints"))
    data = { data: { endpoints: [] } };
  else if (req.url().endsWith("/endpoints"))
    data = {
      data: {
        endpoints: [
          {
            tag: "judge-only",
            supported_parameters: [
              "structured_outputs",
              "response_format",
              "temperature",
              "max_tokens",
            ],
            pricing: { prompt: "0.00001", completion: "0.00002" },
          },
        ],
      },
    };
  else if (req.url().endsWith("/chat/completions")) {
    const body = JSON.parse(req.postData());
    judgeCalls.push(body);
    data = {
      id: "judge-response",
      model: body.model,
      provider: "Judge",
      usage: { cost: 0.001, total_tokens: 25 },
      choices: [
        {
          finish_reason: "stop",
          message: {
            content: JSON.stringify({
              scores: Object.fromEntries(CRITERIA.map((k) => [k, 3])),
              explanation: "Mock judge evidence.",
            }),
          },
        },
      ],
    };
  } else data = { data: [] };
  await route.fulfill({
    json: data,
    headers: { "access-control-allow-origin": "*" },
  });
});
async function settled() {
  await page.waitForFunction(
    () =>
      document.querySelector("#compare-results table") &&
      !document
        .querySelector("#compare-status")
        .textContent.includes("Updating"),
  );
}
try {
  // An existing version-one database upgrades additively.
  await page.goto(url.replace("/llm/", "/"));
  await page.evaluate(async () => {
    await new Promise((resolve, reject) => {
      const r = indexedDB.open("fragile-llm-benchmarks", 1);
      r.onupgradeneeded = () => {
        r.result.createObjectStore("benchmarks", { keyPath: "id" });
        r.result.createObjectStore("events", {
          keyPath: ["benchmark_id", "seq"],
        });
      };
      r.onsuccess = () => {
        r.result.close();
        resolve();
      };
      r.onerror = () => reject(r.error);
    });
  });
  await page.goto(url);
  await page.locator("#api-key").fill("browser-judge-secret");
  assert.equal(
    await page.locator("#compare-judge-model").inputValue(),
    "~google/gemini-flash-latest",
  );
  assert.equal(
    await page
      .locator('[role="tab"]')
      .allTextContents()
      .then((a) => a.map((t) => t.trim()).join(",")),
    "Generation,Analysis,Benchmark,Evaluation",
  );
  await page.locator("#tab-generation").focus();
  await page.keyboard.press("End");
  assert.equal(
    await page.locator("#tab-evaluation").getAttribute("aria-selected"),
    "true",
  );
  await page.keyboard.press("ArrowRight");
  assert.equal(
    await page.locator("#tab-generation").getAttribute("aria-selected"),
    "true",
  );
  await page.locator("#import-file").setInputFiles({
    name: "run.fgllm",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(record)),
  });
  await page.locator("#tab-benchmark").click();
  await settled();
  assert.match(
    await page.locator("#compare-results").textContent(),
    /Archived answers/,
  );
  assert.match(
    await page.locator("#compare-results").textContent(),
    /Retained population/,
  );
  assert.equal(
    await page
      .locator("#compare-results .compare-legend")
      .first()
      .textContent(),
    "Fractal",
  );
  assert.equal(judgeCalls.length, 0);
  assert.equal(await page.locator("#generation-settings").isVisible(), false);
  const chartDownload = page.waitForEvent("download");
  await page
    .locator("#compare-results .compare-chart button")
    .filter({ hasText: "PNG" })
    .first()
    .click();
  const png = await readFile(await (await chartDownload).path());
  assert.equal(png.subarray(1, 4).toString(), "PNG");
  await page.locator("#tab-evaluation").click();
  await page
    .locator("#compare-traces button")
    .filter({ hasText: "Pin A" })
    .first()
    .click();
  await page
    .locator("#compare-traces button")
    .filter({ hasText: "Pin B" })
    .last()
    .click();
  assert.equal(
    await page.locator("#compare-pins .compare-trace-card").count(),
    2,
  );
  await page.locator("#tab-benchmark").click();
  await page.locator('#compare-results svg [role="button"]').first().click();
  assert.equal(
    await page.locator("#tab-evaluation").getAttribute("aria-selected"),
    "true",
  );
  assert.match(await page.locator("#compare-traces").textContent(), /rows/);
  await page
    .locator("#compare-traces button")
    .filter({ hasText: "Clear chart selection" })
    .click();
  await page.locator("#benchmark-file").setInputFiles({
    name: "all.fgllmbench",
    mimeType: "application/x-ndjson",
    buffer: Buffer.from(archive),
  });
  await page.waitForFunction(
    () => document.querySelector("#benchmark-saved").value !== "",
  );
  await settled();
  await page.waitForFunction(() =>
    document
      .querySelector("#compare-results")
      .textContent.includes("Independent token budget"),
  );
  assert.equal(judgeCalls.length, 0);
  await page.locator("#tab-evaluation").click();
  await page.locator("#compare-judge-start").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#compare-judge-status")
      .textContent.startsWith("Grading idle"),
  );
  await settled();
  assert.ok(judgeCalls.length > 0);
  assert.ok(
    judgeCalls.every(
      (b) =>
        b.model === "google/gemini-3.8-flash" &&
        b.temperature === 0 &&
        b.max_tokens === 1024 &&
        !b.logprobs &&
        b.provider.only[0] === "judge-only",
    ),
  );
  assert.ok(
    judgeCalls.every(
      (b) =>
        !b.messages[1].content.includes("fractal") &&
        !b.messages[1].content.includes("temperature"),
    ),
  );
  const calls = judgeCalls.length;
  await page.locator("#compare-judge-preview").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#compare-judge-estimate")
      .textContent.includes("0 missing or failed"),
  );
  assert.equal(judgeCalls.length, calls);
  const downloading = page.waitForEvent("download");
  await page.locator("#compare-export").click();
  const reportText = await readFile(await (await downloading).path(), "utf8"),
    report = parseReport(reportText);
  assert.ok(!reportText.includes("browser-judge-secret"));
  const expected = processReport(report);
  assert.ok(expected.grades.length > 0);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page
      .locator(".compare-chart-grid")
      .first()
      .evaluate(
        (e) => getComputedStyle(e).gridTemplateColumns.split(" ").length,
      ),
    1,
  );
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  );
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: "/tmp/llm-benchmark-comparison.png",
    fullPage: false,
  });
  // Import the report in a new, offline profile and reproduce identical numbers.
  const offlineContext = await browser.newContext();
  await offlineContext.route("https://openrouter.ai/**", (r) => r.abort());
  const offline = await offlineContext.newPage();
  await offline.goto(url);
  await offline.locator("#tab-benchmark").click();
  await offline.locator("#compare-file").setInputFiles({
    name: "report.fgllmcompare",
    mimeType: "application/json",
    buffer: Buffer.from(reportText),
  });
  await offline.waitForFunction(() =>
    document
      .querySelector("#compare-results")
      .textContent.includes("Independent token budget"),
  );
  const actual = await offline.evaluate(async (report) => {
    const { processReport } = await import("./comparison-report.js");
    return processReport(report);
  }, report);
  assert.deepEqual(actual, expected);
  await offlineContext.close();
  // Failed report storage leaves the source and grades available for export.
  await page.evaluate(async () => {
    const { BrowserComparisonStore } = await import("./benchmark-store.js");
    window.restoreComparisonSave = BrowserComparisonStore.save;
    BrowserComparisonStore.save = async () => {
      throw Error("mock quota exceeded");
    };
  });
  await page.locator("#compare-metric").selectOption("tokens");
  await page.waitForFunction(() =>
    document
      .querySelector("#compare-storage")
      .textContent.includes("Storage failed"),
  );
  assert.equal(await page.locator("#compare-export").isEnabled(), true);
  await page.evaluate(async () => {
    const { BrowserComparisonStore } = await import("./benchmark-store.js");
    BrowserComparisonStore.save = window.restoreComparisonSave;
  });
  // Current-source selection restores the imported Fractal recording.
  await page.locator("#compare-evaluation-source").selectOption("");
  await settled();
  await page.waitForFunction(
    () =>
      document.querySelector("#compare-results .compare-legend")
        ?.textContent === "Fractal",
  );
  assert.equal(await page.locator("#compare-judge-session option").count(), 1);
  assert.deepEqual(errors, []);
  console.log(
    "Comparison browser: Fractal-only, all methods, tab navigation, linked pins, grading, offline reports, migration and mobile layout passed",
  );
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
