import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import assert from "node:assert/strict";
import { importRecording } from "../../web/llm/recording.js";
import { game, gameEcho } from "./game-fixtures.mjs";
import { fakeOpenRouter, completion } from "./fixtures.mjs";
const root = resolve(import.meta.dirname, "../../web");
const server = createServer(async (req, res) => {
  try {
    const file = resolve(
      root,
      "." +
        decodeURIComponent(req.url.split("?")[0]) +
        (req.url.endsWith("/") ? "index.html" : ""),
    );
    if (!file.startsWith(root + "/")) throw Error();
    const bytes = await readFile(file);
    res.setHeader(
      "Content-Type",
      {
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".css": "text/css",
        ".html": "text/html",
      }[extname(file)] ?? "application/octet-stream",
    );
    res.end(bytes);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const fake = fakeOpenRouter();
const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "authorization,content-type",
};
let invalidGame = false,
  baselineCalls = 0;
await page.context().route("https://openrouter.ai/api/v1/**", async (route) => {
  const r = route.request();
  if (r.method() === "OPTIONS")
    return route.fulfill({ status: 204, headers: cors });
  const b = r.postDataJSON();
  let data;
  if (b?.response_format?.json_schema?.name === "fixed_target_game") {
    data = {
      model: b.model,
      id: "game-creation",
      choices: [
        {
          finish_reason: "stop",
          message: {
            content: invalidGame
              ? "invalid"
              : JSON.stringify({
                  title: game.title,
                  background: game.background,
                  target: game.target,
                }),
          },
        },
      ],
      usage: { prompt_tokens: 20, completion_tokens: 30 },
    };
  } else
    data = await (await fake.fetch(r.url(), { body: r.postData() })).json();
  await route.fulfill({
    status: 200,
    headers: cors,
    contentType: "application/json",
    body: JSON.stringify(data),
  });
});
await page
  .context()
  .route("https://api.together.ai/v1/completions", async (route) => {
    if (route.request().method() === "OPTIONS")
      return route.fulfill({ status: 204, headers: cors });
    const b = route.request().postDataJSON();
    if (b.prompt.includes("Additional context:\n<|im_end|>")) baselineCalls++;
    await route.fulfill({
      status: 200,
      headers: cors,
      contentType: "application/json",
      body: JSON.stringify(gameEcho(b)),
    });
  });
try {
  await page.goto(`http://127.0.0.1:${server.address().port}/llm/`);
  await page.locator("#api-key").fill("browser-router-secret");
  await page.locator('[name="objective"]').selectOption("xent_game");
  assert.equal(await page.locator("#game-settings").isVisible(), true);
  assert.equal(await page.locator("#game-benchmark").isDisabled(), true);
  await page.locator("#together-key").fill("browser-together-secret");
  await page.locator("#game-brief").fill("A detective finding a missing key");
  await page.locator("#game-generate").click();
  await page.waitForFunction(() =>
    document.querySelector("#game-status").textContent.includes("Game saved"),
  );
  assert.equal(await page.locator("#game-target").textContent(), game.target);
  invalidGame = true;
  await page.locator("#game-generate").click();
  await page.waitForFunction(() =>
    document.querySelector("#game-status").textContent.includes("invalid JSON"),
  );
  assert.equal(await page.locator("#game-target").textContent(), game.target);
  invalidGame = false;
  for (const [k, value] of Object.entries({
    walkers: 2,
    chunk_tokens: 2,
    sequence_tokens: 6,
    iterations: 8,
    concurrency: 2,
  }))
    await page.locator(`[name="${k}"]`).fill(String(value));
  await page.locator("#run").click();
  await page.waitForFunction(() =>
    /budget reached|limit reached|target reached/.test(
      document.querySelector("#status").textContent,
    ),
  );
  assert.equal(
    await page.locator("#best-title").textContent(),
    "Best sampled context",
  );
  assert.match(
    await page.locator("#best-score").textContent(),
    /unsurprising: 3.000/,
  );
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export").click();
  const downloaded = await downloadPromise;
  const text = await readFile(await downloaded.path(), "utf8");
  const saved = importRecording(text);
  assert.equal(saved.config.game.target, game.target);
  assert.ok(
    !text.includes("browser-router-secret") &&
      !text.includes("browser-together-secret"),
  );
  await page.locator("#import-file").setInputFiles({
    name: "game.fgllm",
    mimeType: "application/json",
    buffer: Buffer.from(text),
  });
  await page.waitForFunction(() =>
    document.querySelector("#status").textContent.startsWith("Imported"),
  );
  await page.locator("#tab-generation").click();
  assert.equal(await page.locator("#game-target").textContent(), game.target);
  await page.locator("#reset").click();
  await page.locator('[name="game_mode"]').selectOption("surprising");
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Paused",
  );
  assert.match(
    await page.locator("#best-score").textContent(),
    /surprising: -3.000/,
  );
  await page.locator("#reset").click();
  const baselinesBefore = baselineCalls;
  await page.locator("#game-benchmark").click();
  try {
    await page.waitForFunction(
      () =>
        document
          .querySelector("#benchmark-status")
          .textContent.includes("both game benchmarks saved"),
      null,
      { timeout: 120000 },
    );
  } catch (error) {
    console.error(
      "Game benchmark status:",
      await page.locator("#benchmark-status").textContent(),
      { errors, requests: fake.requests.length, baselineCalls },
    );
    throw error;
  }
  await page.waitForFunction(
    () =>
      document.querySelectorAll("#benchmark-pair-results section").length === 2,
  );
  assert.equal(baselineCalls - baselinesBefore, 1);
  assert.equal(await page.locator("#compare-metric").inputValue(), "reward");
  assert.equal(
    await page.locator("#compare-status-filter").inputValue(),
    "all",
  );
  assert.equal(await page.locator("#compare-pool").inputValue(), "archive");
  await page
    .getByRole("heading", { name: "Best sampled contexts", exact: true })
    .waitFor();
  await page.getByText("Best game score so far", { exact: true }).waitFor();
  assert.match(
    await page.locator("#benchmark-pair-results").textContent(),
    /Make it unsurprising/,
  );
  assert.match(
    await page.locator("#benchmark-pair-results").textContent(),
    /Make it surprising/,
  );
  const state = await page.evaluate(async () => {
    const { BrowserBenchmarkStore } = await import("./benchmark-store.js");
    const entries = await BrowserBenchmarkStore.list();
    return Promise.all(
      entries.map(async (e) => {
        const s = await BrowserBenchmarkStore.open(e.id);
        return { header: s.manifest, events: await s.readEvents() };
      }),
    );
  });
  assert.equal(state.length, 2);
  assert.deepEqual(state[0].header.pair, state[1].header.pair);
  await page.screenshot({
    path: "/tmp/xent-game-benchmark.png",
    fullPage: false,
  });
  // Reopening either side restores the pair; completed retries make no API calls.
  const requestsBefore = fake.requests.length;
  await page.locator("#benchmark-retry").click();
  await page.waitForFunction(
    () =>
      !document.querySelector("#benchmark-retry").disabled &&
      document
        .querySelector("#benchmark-status")
        .textContent.includes("both game benchmarks saved"),
  );
  assert.equal(fake.requests.length, requestsBefore);
  await page.locator("#tab-generation").click();
  await page.screenshot({
    path: "/tmp/xent-game-generation.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
  );
  assert.deepEqual(errors, []);
  console.log("Xent game browser workflow passed");
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
