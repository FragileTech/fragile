import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import assert from "node:assert/strict";
import { rankingSource, rankingJudge } from "./ranking-fixtures.mjs";
import { BenchmarkRunner } from "../../web/llm/benchmark.js";
import { manifest } from "../../web/llm/benchmark-data.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import { fakeOpenRouter } from "./fixtures.mjs";
import {
  createReport,
  parseReport,
  processReport,
} from "../../web/llm/comparison-report.js";
const root = resolve(import.meta.dirname, "../../web"),
  server = createServer(async (req, res) => {
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
          ".html": "text/html",
          ".css": "text/css",
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
  errors = [],
  fake = rankingJudge({ delay: 5 });
page.setDefaultTimeout(60000);
page.on("pageerror", (e) => {
  errors.push(e.message);
  console.error(e);
});
const url = `http://127.0.0.1:${server.address().port}/llm/`;
await context.route("https://openrouter.ai/api/v1/**", async (route) => {
  if (route.request().method() === "OPTIONS") {
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
  const response = await fake.fetch(route.request().url(), {
    body: route.request().postData(),
  });
  await route.fulfill({
    status: response.status,
    body: await response.text(),
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
    },
  });
});
try {
  await page.goto(url + "migration-test.html");
  const oldReport = createReport(rankingSource());
  oldReport.version = 1;
  delete oldReport.rankings;
  delete oldReport.selected_ranking;
  delete oldReport.evaluation_mode;
  await page.evaluate(
    (report) =>
      new Promise((resolve, reject) => {
        const request = indexedDB.open("fragile-llm-benchmarks", 3);
        request.onupgradeneeded = () => {
          request.result
            .createObjectStore("comparisons", { keyPath: "id" })
            .add(report);
          request.result
            .createObjectStore("comparison_index", { keyPath: "id" })
            .add({
              id: report.id,
              created_at: report.created_at,
              source: { kind: report.source.kind },
            });
        };
        request.onsuccess = () => {
          request.result.close();
          resolve();
        };
        request.onerror = () => reject(request.error);
      }),
    oldReport,
  );
  await page.goto(url);
  assert.equal(
    await page.evaluate(async (id) => {
      const { transact } = await import("./benchmark-store.js");
      return transact(["comparisons"], "readonly", (tx, result) => {
        tx.objectStore("comparisons").get(id).onsuccess = (e) =>
          result(e.target.result.version);
      });
    }, oldReport.id),
    1,
  );
  // Additive upgrade from the prior comparison database.
  assert.equal(
    await page.evaluate(
      () =>
        new Promise((resolve) => {
          const r = indexedDB.open("fragile-llm-benchmarks");
          r.onsuccess = () => {
            const v = r.result.version;
            r.result.close();
            resolve(v);
          };
        }),
    ),
    4,
  );
  await page.locator("#import-file").setInputFiles({
    name: "answers.fgllm",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(rankingSource(12).record)),
  });
  await page.waitForFunction(() =>
    document.querySelector("#status").textContent.startsWith("Imported"),
  );
  await page.locator("#tab-evaluation").click();
  await page.locator("#compare-evaluation-key").fill("ranking-browser-secret");
  await page.locator("#compare-evaluation-mode").selectOption("pairwise");
  assert.equal(await page.locator("#compare-judge-start").isVisible(), false);
  await page.locator("#ranking-budget").fill("80");
  await page.locator("#ranking-preview").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#ranking-estimate")
      .textContent.includes("google/gemini"),
  );
  assert.equal(fake.calls.length, 0);
  await page.locator("#ranking-start").click();
  await page.waitForFunction(
    () =>
      document
        .querySelector("#ranking-status")
        .textContent.startsWith("completed") &&
      !document.querySelector("#ranking-start").disabled,
    null,
    { timeout: 60000 },
  );
  assert.ok(fake.calls.length > 0 && fake.calls.length <= 80);
  assert.ok(
    fake.calls.every(
      (b) => b.model === "google/gemini-3.8-flash" && b.max_tokens === 2048,
    ),
  );
  assert.match(
    await page.locator("#ranking-results").textContent(),
    /Model predictions/,
  );
  assert.match(
    await page.locator("#ranking-results").textContent(),
    /No baseline/,
  );
  await page.getByRole("button", { name: "Compare pair", exact: true }).click();
  await page.waitForFunction(() =>
    document
      .querySelector("#ranking-prediction")
      .textContent.includes("Model prediction"),
  );
  assert.equal(
    await page.locator("#compare-pins .compare-trace-card").count(),
    2,
  );
  // Snapshot export and offline processing have the same numerical core.
  const downloading = page.waitForEvent("download");
  await page.locator("#compare-export").click();
  const text = await readFile(await (await downloading).path(), "utf8"),
    report = parseReport(text);
  assert.equal(report.rankings.length, 1);
  assert.equal(report.version, 2);
  assert.ok(!text.includes("ranking-browser-secret"));
  const expected = processReport(report);
  assert.ok(expected.ranking_ratings.length > 0);
  const offline = await browser.newContext();
  await offline.route("https://openrouter.ai/**", (r) => r.abort());
  const p = await offline.newPage();
  await p.goto(url);
  await p.locator("#tab-evaluation").click();
  await p.locator("#compare-file").setInputFiles({
    name: "ranking.fgllmcompare",
    mimeType: "application/json",
    buffer: Buffer.from(text),
  });
  await p.waitForFunction(
    () =>
      document
        .querySelector("#ranking-results")
        .textContent.includes("Model predictions"),
    null,
    { timeout: 60000 },
  );
  const actual = await p.evaluate(
    async (report) =>
      (await import("./comparison-report.js")).processReport(report),
    report,
  );
  const close = (a, b) => {
    if (typeof a === "number" && typeof b === "number") {
      assert.ok(Math.abs(a - b) < 1e-8);
      return;
    }
    if (a && b && typeof a === "object" && typeof b === "object") {
      assert.deepEqual(Object.keys(a), Object.keys(b));
      for (const k of Object.keys(a)) close(a[k], b[k]);
      return;
    }
    assert.equal(a, b);
  };
  close(actual.ranking_ratings, expected.ranking_ratings);
  assert.equal(
    await p.locator("#compare-evaluation-mode").inputValue(),
    "pairwise",
  );
  await p.setViewportSize({ width: 390, height: 844 });
  assert.ok(
    await p.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  );
  await p.screenshot({ path: "/tmp/llm-ranking-mobile.png", fullPage: false });
  await offline.close();
  // Independent human audit is blinded and never changes primary ratings.
  const before = fake.calls.length;
  await page.getByRole("button", { name: "Start blinded human audit" }).click();
  await page.getByRole("button", { name: "Assess next pair" }).click();
  await page.waitForSelector('[id^="human-"]');
  await page.getByRole("button", { name: "Save blinded assessment" }).click();
  await page.waitForFunction(
    () => !document.querySelector(".ranking-human-dialog[open]"),
    null,
    { timeout: 60000 },
  );
  assert.equal(fake.calls.length, before);
  // Model audit uses its own route and paid cap.
  await page.locator("#ranking-audit-model").fill("judge/independent");
  await page.locator("#ranking-audit-budget").fill("80");
  assert.equal(
    await page.locator("#ranking-audit-model").inputValue(),
    "judge/independent",
  );
  await page
    .getByRole("button", { name: "Run model audit", exact: true })
    .click();
  await page.waitForFunction(
    () =>
      document
        .querySelector("#ranking-results")
        .textContent.includes("judge/independent · completed"),
    null,
    { timeout: 60000 },
  );
  assert.ok(fake.calls.some((b) => b.model === "judge/independent"));
  // A separate extension retains earlier fixed-budget evidence and is recoverable.
  await page.locator("#ranking-extension").click();
  await page.locator("#ranking-budget").fill("80");
  await page.locator("#ranking-start").click();
  await page.waitForFunction(
    () => !document.querySelector("#ranking-stop").disabled,
  );
  await page.locator("#ranking-pause").click();
  await page.locator("#ranking-continue").click();
  await page.locator("#ranking-stop").click();
  await page.waitForFunction(
    () => !document.querySelector("#ranking-start").disabled,
  );
  assert.ok((await page.locator("#ranking-sessions option").count()) >= 3);
  const reportId = await page.locator("#compare-reports").inputValue();
  await page.reload();
  await page.locator("#tab-evaluation").click();
  await page.locator("#compare-reports").selectOption(reportId);
  await page.waitForFunction(
    () => document.querySelectorAll("#ranking-sessions option").length >= 3,
  );
  await page.locator("#compare-evaluation-key").fill("ranking-browser-secret");
  if (await page.locator("#ranking-resume").isEnabled()) {
    await page.locator("#ranking-resume").click();
    await page.waitForFunction(
      () =>
        !document.querySelector("#ranking-start").disabled &&
        /^(completed|budget_exhausted)/.test(
          document.querySelector("#ranking-status").textContent,
        ),
      null,
      { timeout: 60000 },
    );
  }
  await page.screenshot({
    path: "/tmp/llm-ranking-desktop.png",
    fullPage: false,
  });
  const storage = await page.evaluate(async (session) => {
    const { BrowserRankingStore, rankingHeader } = await import(
      "./ranking-store.js"
    );
    const { transact } = await import("./benchmark-store.js");
    const fresh = rankingHeader({ ...session, id: crypto.randomUUID() });
    const a = await BrowserRankingStore.create(fresh, "storage-test");
    const b = await BrowserRankingStore.open(fresh.id);
    let locked = false,
      stale = false,
      aborted = false;
    await a.lock(async () => {
      try {
        await b.lock(async () => {});
      } catch {
        locked = true;
      }
    });
    await a.append({ seq: 1, time: 1, type: "status", payload: "running" });
    try {
      await b.append({ seq: 1, time: 2, type: "status", payload: "stopped" });
    } catch {
      stale = true;
    }
    try {
      await transact(
        ["ranking_sessions", "ranking_events"],
        "readwrite",
        (tx) => {
          tx.objectStore("ranking_sessions").put({
            id: fresh.id,
            header: fresh,
            lastSeq: 99,
          });
          tx.objectStore("ranking_events").add({
            session_id: fresh.id,
            seq: 2,
            type: "status",
            payload: "stopped",
          });
          tx.abort();
        },
      );
    } catch {
      aborted = true;
    }
    const restored = await (await BrowserRankingStore.open(fresh.id)).read();
    return {
      locked,
      stale,
      aborted,
      seq: restored.seq,
      status: restored.status,
    };
  }, report.rankings[0]);
  assert.deepEqual(storage, {
    locked: true,
    stale: true,
    aborted: true,
    seq: 1,
    status: "running",
  });
  // The same Evaluation workflow uses all saved benchmark methods and trial weights.
  const benchmark = new MemoryBenchmarkStore(
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
  await new BenchmarkRunner(benchmark, "fixture", {
    fetchImpl: fakeOpenRouter().fetch,
  }).run();
  const comparison = createReport({
    kind: "benchmark",
    manifest: benchmark.manifest,
    events: benchmark.events,
  });
  comparison.evaluation_mode = "pairwise";
  await page.locator("#compare-file").setInputFiles({
    name: "methods.fgllmcompare",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(comparison)),
  });
  await page.waitForFunction(
    () => document.querySelectorAll("#ranking-sessions option").length === 1,
  );
  await page.locator("#compare-evaluation-key").fill("ranking-browser-secret");
  await page.locator("#ranking-budget").fill("120");
  await page.locator("#ranking-start").click();
  await page.waitForFunction(
    () =>
      !document.querySelector("#ranking-start").disabled &&
      /^(completed|budget_exhausted)/.test(
        document.querySelector("#ranking-status").textContent,
      ),
    null,
    { timeout: 60000 },
  );
  const methodText = await page.locator("#ranking-results").textContent();
  for (const expected of [
    "Independent population",
    "Independent token budget",
    "Temperature zero",
    "archive",
    "retained",
  ])
    assert.ok(methodText.includes(expected), expected);
  await page.locator("#ranking-results").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "/tmp/llm-ranking-methods.png",
    fullPage: false,
  });
  assert.deepEqual(errors, []);
  console.log(
    "Pairwise browser: direct Gemini judging, both orders, ratings, predictions, offline reports, migration, human/model audits, extension recovery and mobile layout passed",
  );
} catch (e) {
  console.error("Ranking browser failure:", e.message, "at", page.url());
  console.error(
    "Ranking browser state:",
    await page
      .locator("#ranking-status")
      .textContent({ timeout: 1000 })
      .catch(() => "Ranking not mounted"),
    await page
      .locator("#ranking-prediction")
      .textContent({ timeout: 1000 })
      .catch(() => "No prediction"),
    await page
      .locator("#compare-storage")
      .textContent({ timeout: 1000 })
      .catch(() => "Storage not mounted"),
  );
  throw e;
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
