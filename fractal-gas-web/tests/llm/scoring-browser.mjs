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

let scoringCalls = 0;
const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "authorization,content-type",
};
await page.context().route("https://openrouter.ai/api/v1/**", async (route) => {
  const r = route.request();
  if (r.method() === "OPTIONS")
    return route.fulfill({ status: 204, headers: cors });
  const response = await fake.fetch(r.url(), { body: r.postData() });
  await route.fulfill({
    status: response.status,
    headers: cors,
    contentType: "application/json",
    body: JSON.stringify(await response.json()),
  });
});
await page
  .context()
  .route("https://api.together.ai/v1/completions", async (route) => {
    if (route.request().method() === "OPTIONS")
      return route.fulfill({ status: 204, headers: cors });
    scoringCalls++;
    const b = route.request().postDataJSON(),
      end = b.prompt.lastIndexOf("</think>\n\n") + "</think>\n\n".length;
    const prefix = b.prompt.slice(0, end),
      answer = [...b.prompt.slice(end)];
    const lp = prefix.includes("user\n<|") ? -2 : -1;
    const body = {
      model: b.model,
      prompt: [
        {
          text: b.prompt,
          logprobs: {
            tokens: [prefix, ...answer],
            token_ids: [1, ...answer.map((c) => c.codePointAt(0) + 10)],
            token_logprobs: [null, ...answer.map(() => lp)],
          },
        },
      ],
      usage: {
        prompt_tokens: answer.length + 12,
        completion_tokens: b.max_tokens,
      },
    };
    await route.fulfill({
      status: 200,
      headers: cors,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
try {
  await page.goto(`http://127.0.0.1:${server.address().port}/llm/`);
  assert.equal(await page.locator('[name="objective"]').inputValue(), "beam");
  assert.equal(await page.locator("#xed-settings").isVisible(), false);
  assert.equal(await page.locator("#beam-settings").isVisible(), true);
  assert.equal(await page.locator('[name="beam_alpha"]').inputValue(), "0.6");
  await page.locator("#api-key").fill("browser-key");
  for (const mode of ["beam", "xed"]) {
    await page.locator('[name="objective"]').selectOption(mode);
    if (mode === "beam") {
      assert.equal(await page.locator("#beam-settings").isVisible(), true);
      assert.equal(
        await page.locator('[name="beam_alpha"]').inputValue(),
        "0.6",
      );
      await page.locator('[name="beam_alpha"]').fill("0.4");
    } else {
      assert.equal(await page.locator("#xed-settings").isVisible(), true);
      await page.locator('[name="xed_direction"]').selectOption("minimize");
      await page.locator("#together-key").fill("browser-together-key");
    }
    await page.locator('[name="walkers"]').fill("2");
    await page.locator('[name="chunk_tokens"]').fill("2");
    await page.locator('[name="sequence_tokens"]').fill("4");
    await page.locator('[name="iterations"]').fill("2");
    await page.locator("#run").click();
    await page.waitForFunction(
      () =>
        document.querySelector("#run").disabled &&
        /budget reached|limit reached|target reached/.test(
          document.querySelector("#status").textContent,
        ),
      null,
      { timeout: 30000 },
    );
    assert.equal(await page.locator("#step").isDisabled(), true);
    const downloaded = page.waitForEvent("download");
    await page.locator("#export").click();
    const data = JSON.parse(
      await readFile(await (await downloaded).path(), "utf8"),
    );
    assert.equal(data.config.objective, mode);
    if (mode === "xed") {
      assert.equal(data.config.xed_direction, "minimize");
      assert.ok(data.nodes.slice(1).every((n) => n.xed));
      assert.ok(scoringCalls > 0);
      assert.ok(!JSON.stringify(data).includes("browser-together-key"));
      assert.match(await page.locator("#stats").textContent(), /Scoring:/);
    } else assert.equal(scoringCalls, 0);
    await page.locator("#tab-analysis").click();
    assert.equal(
      await page.locator("#analysis-metric").inputValue(),
      "objective",
    );
    await page.locator("#analysis-best").click();
    assert.match(
      await page.locator("#analysis-node-metrics").textContent(),
      /Internal utility/,
    );
    await page.screenshot({
      path: `/tmp/llm-scoring-${mode}.png`,
      fullPage: true,
    });
    await page.locator("#tab-generation").click();
    await page.locator("#reset").click();
    const upload = await page.locator('input[type="file"]').count();
    assert.ok(upload > 0);
  }
  assert.deepEqual(errors, []);
  console.log("Scoring browser checks passed");
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
