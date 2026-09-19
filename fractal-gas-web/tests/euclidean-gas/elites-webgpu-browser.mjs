import { chromium } from "playwright";
import assert from "node:assert/strict";

const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader"],
});
let timedOut = false;
const timeout = setTimeout(() => {
  timedOut = true;
  console.error("WebGPU runtime timed out after 90 seconds");
  void browser.close();
}, 90000);
try {
  const page = await browser.newPage();
  page.on("console", (message) => {
    if (message.type() === "error" || message.text().startsWith("Elite GPU:"))
      console.log(message.text());
  });
  let rejectPageError;
  const pageError = new Promise((_, reject) => {
    rejectPageError = reject;
  });
  page.on("pageerror", rejectPageError);
  await page.goto(base + "/euclidean-gas/index.html");
  const result = await Promise.race([
    pageError,
    page.evaluate(
      async (count) => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return { skipped: "No WebGPU adapter" };
        console.log("Elite GPU: loading module");
        const engine = await import("/euclidean-gas/engine/webgpu/gas.js");
        await engine.default();
        const config = JSON.parse(engine.default_config());
        config.walkers = 8;
        config.gas.backend = "wgpu";
        config.gas.n_elite = count;
        console.log(`Elite GPU: creating run with ${count} elites`);
        const gas = await engine.BrowserGas.create(JSON.stringify(config));
        try {
          console.log("Elite GPU: stepping");
          await gas.step(1);
          const saved = gas.checkpoint();
          const frame = await gas.step(1);
          console.log("Elite GPU: two steps completed");
          return {
            count: frame.elite_count,
            protected:
              count === 0 ||
              frame.report.clone_plan.choices
                .slice(0, 2)
                .every((c) => !c.accepted),
            population: frame.population,
            saved: Array.from(saved),
          };
        } finally {
          gas.free();
        }
      },
      Number(process.env.ELITE_COUNT ?? 2),
    ),
  ]);
  if (result.skipped) console.log(`WebGPU elites: skipped (${result.skipped})`);
  else {
    assert.equal(result.count, Number(process.env.ELITE_COUNT ?? 2));
    assert.equal(result.protected, true);
    // A fresh page gives the restored run an independent WASM/device context.
    // This also exercises the save-and-reload checkpoint workflow.
    const restoredPage = await browser.newPage();
    restoredPage.on("pageerror", rejectPageError);
    await restoredPage.goto(base + "/euclidean-gas/index.html");
    const restored = await Promise.race([
      pageError,
      restoredPage.evaluate(async (saved) => {
        const engine = await import("/euclidean-gas/engine/webgpu/gas.js");
        await engine.default();
        const run = await engine.BrowserGas.restore(new Uint8Array(saved));
        try {
          return await run.step(1);
        } finally {
          run.free();
        }
      }, result.saved),
    ]);
    assert.equal(restored.elite_count, result.count);
    assert.deepEqual(restored.population, result.population);
    console.log(
      "WebGPU elites: clone protection and checkpoint continuation passed",
    );
  }
} finally {
  clearTimeout(timeout);
  await browser.close();
  if (timedOut) process.exitCode = 1;
}
