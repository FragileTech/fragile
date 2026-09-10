import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import assert from "node:assert/strict";
import { configuration } from "../../web/llm/config.js";
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
let gate,
  waiting = 0,
  eos = false,
  unsupported = false;
await page.context().route("https://openrouter.ai/api/v1/**", async (route) => {
  const request = route.request();
  const headers = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "authorization,content-type",
  };
  if (request.method() === "OPTIONS") {
    await route.fulfill({ status: 204, headers });
    return;
  }
  if (gate && request.url().endsWith("chat/completions")) {
    waiting++;
    await gate;
    waiting--;
  }
  let response =
    unsupported && request.url().endsWith("/endpoints")
      ? { status: 200, json: async () => ({ data: { endpoints: [] } }) }
      : await fake.fetch(request.url(), {
          body: request.postData(),
        });
  if (
    eos &&
    request.url().endsWith("chat/completions") &&
    !JSON.parse(request.postData()).messages[0].content.startsWith(
      "Copy the following text exactly",
    )
  ) {
    response = { status: 200, json: async () => completion("x", -0.5, "stop") };
  }
  await route
    .fulfill({
      status: response.status,
      headers,
      contentType: "application/json",
      body: JSON.stringify(await response.json()),
    })
    .catch(() => {}); // A stopped run can cancel an already intercepted request.
});
try {
  await page.goto(`http://127.0.0.1:${server.address().port}/llm/`);
  await page.locator("#api-key").fill("browser-test-key");
  for (const algorithm of ["wave", "graph"]) {
    if (algorithm === "graph") await page.locator("#reset").click();
    await page.locator('[name="algorithm"]').selectOption(algorithm);
    await page.locator('[name="chunk_tokens"]').fill("2");
    await page.locator('[name="sequence_tokens"]').fill("8");
    await page.locator('[name="iterations"]').fill("4");
    if (algorithm === "graph") {
      await page.locator('[name="objective"]').selectOption("mean");
      await page.locator('[name="embedding_input"]').selectOption("prompt");
    }
    await page.locator("#step").click();
    await page.waitForFunction(
      () => document.querySelector("#status").textContent === "Paused",
      null,
      { timeout: 20000 },
    );
    assert.ok((await page.locator("#traces tr").count()) >= 8);
    await page.locator("#tab-analysis").click();
    assert.equal(await page.locator("#generation-settings").isVisible(), false);
    assert.equal(
      await page.locator("#analysis-mode").inputValue(),
      algorithm === "graph" ? "graph" : "tree",
    );
    await page.locator("#run").click();
    await page.waitForFunction(
      () =>
        [
          "Iteration limit reached",
          "Generated-token budget reached",
          "No active branches remain",
        ].includes(document.querySelector("#status").textContent),
      null,
      { timeout: 20000 },
    );
    assert.equal(await page.locator("#analysis-step").textContent(), "4 / 4");
    await page.locator("#tab-generation").click();
    await page.locator("#show-best").click();
    assert.ok((await page.locator("#tokens span").count()) > 0);
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloadPromise;
    const text = await readFile(await download.path(), "utf8");
    assert.ok(!text.includes("browser-test-key"));
    const recording = JSON.parse(text);
    assert.equal(recording.snapshots.length, 4);
    if (algorithm === "wave")
      await page.screenshot({ path: "/tmp/llm-lab.png", fullPage: true });
    await page.locator("#import-file").setInputFiles({
      name: "recording.json",
      mimeType: "application/json",
      buffer: Buffer.from(text),
    });
    await page.waitForFunction(() =>
      document.querySelector("#status").textContent.startsWith("Imported"),
    );
    assert.equal(await page.locator("#run").isDisabled(), true);
    assert.equal(
      await page.locator("#tab-analysis").getAttribute("aria-selected"),
      "true",
    );
    const callsBeforeAnalysis = fake.requests.length;
    await page.locator("#analysis-mode").selectOption("tree");
    await page.locator("#analysis-fit").click();
    assert.equal(
      Number(await page.locator("#analysis-canvas").getAttribute("data-nodes")),
      recording.nodes.length,
    );
    if (!(await page.locator(".analysis-node-list").evaluate((n) => n.open)))
      await page.locator("#analysis-node-count").click();
    const choose = async (id) =>
      page
        .locator("#analysis-node-links button")
        .filter({ hasText: new RegExp(`^#${id}$`) })
        .click();
    await choose(1);
    await page.locator("#tab-generation").click();
    assert.equal(
      await page.locator("#traces tr.selected button").textContent(),
      "#1",
    );
    await page.locator("#tab-analysis").click();
    assert.equal(
      await page.locator("#analysis-answer").textContent(),
      recording.nodes[1].text,
    );
    await page.locator("#analysis-pin").click();
    await choose(2);
    await page.locator("#analysis-pin").click();
    assert.equal(
      await page.locator("#analysis-compare-columns article").count(),
      2,
    );
    assert.match(
      await page.locator("#analysis-compare-summary").textContent(),
      /Shared ancestry/,
    );
    await choose(1);
    const decision = page.locator("#analysis-decisions button").first();
    if (await decision.count()) {
      await decision.click();
      assert.equal(await page.locator("#analysis-overlays").isChecked(), true);
      assert.match(
        await page.locator("#analysis-decision-detail").textContent(),
        /Uniform draw/,
      );
    }
    if (await page.locator("#analysis-next").isDisabled())
      await page.locator("#analysis-prev").click();
    await page.locator("#analysis-metric").selectOption("distance");
    const legend = await page.locator("#analysis-legend").textContent();
    await page.locator("#analysis-next").click();
    assert.equal(await page.locator("#analysis-legend").textContent(), legend);
    await page.locator("#analysis-metric").selectOption("probability");
    assert.match(
      await page.locator("#analysis-legend").textContent(),
      /logarithmic/,
    );
    await page.locator("#analysis-prev").click();
    const earlier = await page.locator("#analysis-step").textContent();
    await page.locator("#analysis-play").click();
    await page.waitForFunction(
      (previous) =>
        document.querySelector("#analysis-step").textContent !== previous,
      earlier,
    );
    await page.locator("#analysis-play").click();
    await page.locator("#analysis-follow").check();
    await page.locator("#analysis-axis").selectOption("iteration");
    await page.locator("#analysis-search").fill("1");
    await page.waitForFunction(
      () =>
        Number(document.querySelector("#analysis-canvas").dataset.nodes) === 2,
    );
    await page.locator("#analysis-search").fill("");
    await page.waitForFunction(
      (n) =>
        Number(document.querySelector("#analysis-canvas").dataset.nodes) === n,
      recording.nodes.length,
    );
    for (const filter of [
      "population",
      "finished",
      "discarded",
      "subtree",
      "all",
    ])
      await page.locator("#analysis-filter").selectOption(filter);
    await choose(1);
    await page.locator("#analysis-collapse").click();
    await page.locator("#analysis-expand").click();
    await page.locator("#analysis-focus").click();
    const canvas = page.locator("#analysis-canvas");
    await canvas.focus();
    await page.keyboard.press("Home");
    assert.match(
      await page.locator("#analysis-node-title").textContent(),
      /Prompt root/,
    );
    await page.keyboard.press("ArrowRight");
    assert.match(
      await page.locator("#analysis-node-title").textContent(),
      /Node #/,
    );
    await page
      .locator("#analysis-minimap")
      .click({ position: { x: 60, y: 40 } });
    await page.locator("#analysis-zoom-in").click();
    await page.locator("#analysis-zoom-out").click();
    await page.locator("#analysis-fit").click();
    const pngPromise = page.waitForEvent("download");
    await page.locator("#analysis-png").click();
    const png = await readFile(await (await pngPromise).path());
    assert.equal(png.subarray(1, 4).toString(), "PNG");
    await page.locator("#analysis-best").click();
    if (await page.locator("#analysis-origins button").count()) {
      await page.locator("#analysis-origins button").first().click();
      assert.match(
        await page.locator("#analysis-decision-detail").textContent(),
        /Evaluated/,
      );
      assert.equal(await page.locator("#analysis-overlays").isChecked(), true);
    }
    await page.screenshot({
      path: `/tmp/llm-analysis-${algorithm}.png`,
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    const graphBox = await page.locator(".analysis-graph").boundingBox(),
      inspectorBox = await page.locator(".analysis-inspector").boundingBox();
    assert.ok(inspectorBox.y > graphBox.y + graphBox.height);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.screenshot({
      path: `/tmp/llm-analysis-mobile-${algorithm}.png`,
      fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    assert.equal(
      fake.requests.length,
      callsBeforeAnalysis,
      "Offline analysis must not request models",
    );
    assert.deepEqual(
      await page.evaluate(() => [localStorage.length, sessionStorage.length]),
      [0, 0],
    );
  }
  await page.locator("#reset").click();
  await page.locator('[name="algorithm"]').selectOption("wave");
  await page.locator('[name="sequence_tokens"]').fill("32");
  await page.locator('[name="iterations"]').fill("16");
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Paused",
  );
  let release;
  gate = new Promise((resolve) => {
    release = resolve;
  });
  await page.locator("#run").click();
  while (!waiting) await new Promise((resolve) => setTimeout(resolve, 10));
  await page.locator("#pause").click();
  gate = null;
  release();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Paused",
  );
  const exportRecord = async () => {
    const download = page.waitForEvent("download");
    await page.locator("#export").click();
    return JSON.parse(await readFile(await (await download).path(), "utf8"));
  };
  const boundary = await exportRecord();
  assert.equal(boundary.snapshots.length, 2);
  gate = new Promise((resolve) => {
    release = resolve;
  });
  await page.locator("#run").click();
  while (!waiting) await new Promise((resolve) => setTimeout(resolve, 10));
  await page.locator("#stop").click();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Stopped",
  );
  gate = null;
  release();
  const stopped = await exportRecord();
  assert.deepEqual(stopped.snapshots, boundary.snapshots);
  assert.deepEqual(stopped.nodes, boundary.nodes);
  assert.equal(await page.locator("#run").isDisabled(), true);
  for (const algorithm of ["wave", "graph"]) {
    await page.locator("#reset").click();
    await page.locator('[name="algorithm"]').selectOption(algorithm);
    eos = true;
    await page.locator("#run").click();
    await page.waitForFunction(
      () =>
        document.querySelector("#status").textContent ===
        "Completed — EOS target reached",
    );
    const completed = await exportRecord();
    assert.equal(completed.version, 3);
    assert.equal(completed.run.stop_reason, "eos_target");
    assert.equal(completed.run.eos_node_ids.length, completed.config.walkers);
    assert.match(
      await page.locator("#stats").textContent(),
      /8\/8 EOS completions/,
    );
    assert.equal(await page.locator("#run").isDisabled(), true);
    assert.equal(await page.locator("#step").isDisabled(), true);
    const requests = fake.requests.length;
    // Exercise stale UI commands as well as the disabled controls.
    await page.evaluate(() => document.querySelector("#run").onclick());
    await page.waitForFunction(
      () => !document.querySelector("#reset").disabled,
    );
    await page.evaluate(() => document.querySelector("#step").onclick());
    await page.waitForFunction(
      () => !document.querySelector("#reset").disabled,
    );
    assert.equal(fake.requests.length, requests);
    assert.deepEqual((await exportRecord()).nodes, completed.nodes);
    eos = false;
  }
  await page.locator("#reset").click();
  unsupported = true;
  await page.locator("#step").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.includes("No available endpoint"),
  );
  assert.equal(await page.locator("#run").isDisabled(), true);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
  );
  // Large offline history validates culling, bounded labels and responsive controls.
  const synthetic = {
    format: "fgllm",
    version: 1,
    engine: "fgllm-1",
    config: configuration({
      walkers: 2,
      chunk_tokens: 1,
      sequence_tokens: 100,
    }),
    metadata: { dimensions: 3 },
    requests: [],
    attempts: [],
    errors: [],
    nodes: [
      {
        id: 0,
        parent: null,
        text: "",
        tokens: 0,
        logp: 0,
        status: 0,
        embedding: [],
        token_data: [],
      },
    ],
    snapshots: [],
  };
  for (let id = 1; id <= 10000; id++) {
    const parent = Math.floor((id - 1) / 4),
      n = synthetic.nodes[parent];
    synthetic.nodes.push({
      id,
      parent,
      text: n.text + "a",
      tokens: n.tokens + 1,
      logp: n.logp - 0.1,
      status: 0,
      embedding: [1, 2, 3],
      token_data: [{ text: "a", bytes: [97], logprob: -0.1 }],
      action: id,
      duration: 1,
      actual_tokens: 1,
      finish_reason: "length",
    });
  }
  synthetic.snapshots = [
    {
      step: 1,
      iteration: 1,
      node_count: 10001,
      walkers: [9999, 10000].map((node, slot) => ({
        node,
        slot,
        parentSlot: 0,
        score: synthetic.nodes[node].logp,
        fitness: 1,
        cloned: false,
        leaf: true,
        alive: true,
        cloneCompanion: 0,
        fitnessCompanion: 0,
      })),
    },
  ];
  await page.locator("#import-file").setInputFiles({
    name: "large.fgllm",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(synthetic)),
  });
  await page.waitForFunction(
    () =>
      Number(document.querySelector("#analysis-canvas").dataset.nodes) ===
      10001,
  );
  await page.locator("#analysis-focus").click();
  await page.waitForFunction(
    () =>
      Number(document.querySelector("#analysis-canvas").dataset.drawn) < 1000,
  );
  assert.ok(
    Number(
      await page.locator("#analysis-canvas").getAttribute("data-labels"),
    ) <= 180,
  );
  const begin = Date.now();
  await page.locator("#analysis-search").fill("10000");
  await page.waitForFunction(
    () => Number(document.querySelector("#analysis-canvas").dataset.nodes) < 20,
  );
  assert.ok(Date.now() - begin < 3000);
  await page.locator("#tab-benchmark").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#compare-status")
      .textContent.startsWith("Partial preview"),
  );
  assert.ok((await page.locator("#compare-results table").count()) > 0);
  await page.locator("#compare-metric").selectOption("tokens");
  await page.waitForFunction(
    () =>
      !document
        .querySelector("#compare-status")
        .textContent.includes("Updating"),
  );
  await page.locator("#tab-generation").click();
  assert.equal(await page.locator("#generation-settings").isVisible(), true);
  assert.deepEqual(errors, []);
  console.log(
    "LLM browser: Wave/Graph generation and Analysis, playback, selection, decisions, filtering, comparison, PNG, offline imports, mobile, keyboard, 10,000-node culling passed",
  );
} catch (e) {
  console.error(
    "Browser status:",
    await page.locator("#status").textContent(),
    errors,
    fake.requests.map((r) => r.url),
  );
  await page.screenshot({ path: "/tmp/llm-error.png", fullPage: true });
  throw e;
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
