// Opt-in paid acceptance check. Start the local lab, then run test:llm-live.
// The key is read locally and entered into the browser; requests run in its worker.
import { chromium } from "playwright";
import { readFile, writeFile } from "node:fs/promises";
import { parseEnv } from "node:util";
import assert from "node:assert/strict";
import { importRecording } from "../../web/llm/recording.js";
const local = await readFile(
  new URL("../../../.env", import.meta.url),
  "utf8",
).catch(() => "");
const key =
  process.env.OPENROUTER_API_KEY || parseEnv(local).OPENROUTER_API_KEY;
if (!key)
  throw new Error(
    "Set OPENROUTER_API_KEY locally before running the live acceptance check",
  );
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
page.on("console", (message) => {
  if (message.type() === "error")
    console.log(message.text().replaceAll(key, "[redacted]"));
});
page.on("requestfailed", (request) =>
  console.log(
    JSON.stringify({
      url: request.url(),
      failure: request.failure()?.errorText,
    }),
  ),
);
try {
  await page.goto(process.env.LLM_LAB_URL || "http://127.0.0.1:8876/llm/");
  await page.locator("#api-key").fill(key);
  await page
    .locator("summary")
    .filter({ hasText: "Algorithm settings" })
    .click();
  for (const algorithm of ["wave", "graph"]) {
    if (algorithm === "graph") await page.locator("#reset").click();
    await page.locator('[name="algorithm"]').selectOption(algorithm);
    await page.locator('[name="walkers"]').fill("4");
    await page.locator('[name="chunk_tokens"]').fill("32");
    await page.locator('[name="sequence_tokens"]').fill("256");
    await page.locator('[name="iterations"]').fill("24");
    await page.locator('[name="concurrency"]').fill("2");
    await page.locator('[name="max_walkers"]').fill("32");
    await page.locator('[name="temperature"]').fill("0.6");
    await page.locator("#run").click();
    await page.waitForFunction(
      () => !document.querySelector("#reset").disabled,
      null,
      { timeout: 240000 },
    );
    const status = await page.locator("#status").textContent();
    const download = page.waitForEvent("download");
    await page.locator("#export").click();
    const text = await readFile(await (await download).path(), "utf8");
    assert.ok(!text.includes(key), "Credential must never enter a recording");
    const record = importRecording(text);
    await writeFile(`/tmp/llm-eos-live-${algorithm}.fgllm`, text);
    console.log(
      JSON.stringify({
        algorithm,
        status,
        steps: record.snapshots.length,
        provider: record.metadata.provider,
        nodes: record.nodes.length - 1,
        clones: record.snapshots.reduce(
          (n, s) => n + s.walkers.filter((w) => w.cloned).length,
          0,
        ),
        requests: record.requests.length,
        errors: record.errors,
        run: record.run,
        max_depth: Math.max(...record.nodes.map((n) => n.tokens)),
      }),
    );
    assert.ok(
      record.snapshots.length >= 2,
      "Live route must complete at least two chunks",
    );
    assert.ok(
      record.nodes.some(
        (n) => n.parent > 0 && record.nodes[n.parent].parent > 0,
      ),
      "A prefix with multiple inherited chunks must be continued",
    );
    assert.ok(
      record.snapshots.some((s) => s.walkers.some((w) => w.cloned)),
      "Live acceptance must exercise cloning",
    );
    assert.equal(record.errors.length, 0);
    assert.equal(record.version, 3);
    assert.ok(record.run.stop_reason);
    assert.ok(record.run.generated_tokens <= record.run.token_budget);
    assert.equal(await page.locator("#run").isDisabled(), true);
    assert.equal(await page.locator("#step").isDisabled(), true);
    for (const request of record.requests.filter(
      (r) => r.status === "started" && r.source > 0,
    )) {
      const source = record.nodes[request.source];
      assert.equal(
        source.status,
        0,
        "EOS and capped nodes must never be extended",
      );
      assert.equal(request.request.messages.at(-1).content, source.text);
    }
    // This sky prompt should not restart its opening paragraph at a chunk boundary.
    for (const n of record.nodes.filter((n) => n.parent > 0)) {
      let first = record.nodes[n.parent];
      while (first.parent > 0) first = record.nodes[first.parent];
      const chunk = n.text.slice(record.nodes[n.parent].text.length);
      assert.ok(
        !chunk.startsWith(first.text.slice(0, 32)),
        "Provider restarted the opening instead of continuing",
      );
    }
    const decisions = record.snapshots.flatMap((s) => s.decisions);
    assert.ok(decisions.length > 0);
    assert.ok(decisions.some((d) => d.cloned));
    assert.ok(
      decisions.every(
        (d) => Number.isFinite(d.distance) && Number.isFinite(d.draw),
      ),
    );
    console.log(
      JSON.stringify({
        algorithm,
        diagnostics: decisions.length,
        decisionClones: decisions.filter((d) => d.cloned).length,
      }),
    );
    await page.locator("#tab-analysis").click();
    await page.screenshot({
      path: `/tmp/llm-live-analysis-${algorithm}.png`,
      fullPage: true,
    });
  }
} catch (e) {
  console.error(String(e.message).replaceAll(key, "[redacted]"));
  process.exitCode = 1;
} finally {
  await browser.close();
}
