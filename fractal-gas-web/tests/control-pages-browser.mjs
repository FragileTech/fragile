// Exercise the actual static files under a GitHub Pages project prefix. This
// server deliberately sends no COOP/COEP headers: the shipped service worker
// must establish isolation, and a browser without service workers must fall back.
import { chromium, firefox } from "playwright";
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
  const browserName = process.env.CONTROL_BROWSER || "chromium";
  const browserType = { chromium, firefox }[browserName];
  assert(browserType, `Unsupported browser: ${browserName}`);
  browser = await browserType.launch({
    headless: true,
    executablePath: process.env.CONTROL_BROWSER_EXECUTABLE,
  });
  const url = `http://127.0.0.1:${server.address().port}${prefix}lab/`;
  for (const serviceWorkers of ["allow", "block"]) {
    console.log(`Pages smoke test: service workers ${serviceWorkers}`);
    const context = await browser.newContext({ serviceWorkers });
    if (serviceWorkers === "block") {
      // Playwright's register stub resolves undefined, unlike a browser that
      // lacks the API. Model an unavailable API rather than that test stub.
      await context.addInitScript(() => {
        Object.defineProperty(navigator, "serviceWorker", { value: undefined });
      });
    }
    await context.addInitScript(() =>
      localStorage.setItem("lab.workspace.onboarded", "true"),
    );
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
            .textContent.startsWith(isolated ? "8 THREADS" : "1 THREAD"),
        serviceWorkers === "allow",
      );
      assert.equal(new URL(page.url()).pathname, `${prefix}lab/`);
      assert.equal(
        new URL(
          await page
            .getByRole("link", { name: /User guide/ })
            .getAttribute("href"),
          page.url(),
        ).pathname,
        `${prefix}docs/lab/`,
      );
      assert(
        await page
          .locator(".brand-logo")
          .evaluate((i) => i.complete && i.naturalWidth > 0),
      );
      // Exercise real planning in both backends with a small smoke-test budget.
      // Set six action frames so tick and replay assertions stay meaningful.
      await page.locator("#tab-controller").click();
      for (const [id, value] of [
        ["walkers", "8"],
        ["horizon", "2"],
        ["frames", "6"],
      ]) {
        await page.locator(`#${id}`).fill(value);
        await page.locator(`#${id}`).press("Tab");
        await page.waitForFunction(
          () => !document.getElementById("run").disabled,
        );
      }
      await page.locator("#scenario").selectOption("racing");
      await applyDraft(page);
      await page.waitForFunction(
        () => !document.getElementById("run").disabled,
      );
      const helpIcons = page.locator(".help-icon");
      console.log("Pages smoke test: initial scene loaded");
      await page.waitForFunction(
        () => document.querySelectorAll(".help-icon").length >= 20,
      );
      assert(
        (await helpIcons.count()) >= 20,
        "lab parameter controls should expose tooltip help icons",
      );
      // FMC has no controller-specific fields; CEM exercises dynamic help.
      await page.locator("#algorithm").selectOption("cem");
      await applyDraft(page);
      await page.waitForFunction(
        () =>
          !document.getElementById("run").disabled &&
          document.querySelectorAll("#algorithm-settings .help-icon").length >
            0,
      );
      assert(
        (await page.locator("#algorithm-settings .help-icon").count()) > 0,
        "controller-specific parameters should expose tooltip help icons",
      );
      const walkersHelp = page.locator("label:has(#walkers) > .help-icon");
      await walkersHelp.click();
      await assertTooltip(page, /candidate worlds/i);
      assert.equal(
        await walkersHelp.getAttribute("aria-describedby"),
        "tooltip",
      );
      await page.keyboard.press("Escape");
      assert.equal(await page.locator("#tooltip").getAttribute("hidden"), "");
      await page.locator("#algorithm").selectOption("fmc");
      await applyDraft(page);
      await page.waitForFunction(
        () => !document.getElementById("run").disabled,
      );
      await applyDraft(page);
      await page.locator("#step").click();
      await page.waitForFunction(
        () => document.getElementById("tick").textContent === "TICK 000006",
      );
      assert.match(
        await page.locator("#score-note").textContent(),
        /Checkpoint/,
      );
      await page.locator("#tab-setup").click();
      await checkAntsControls(page);
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
      console.error("Browser check failed:", error);
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
          tick: document.getElementById("tick")?.textContent,
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

async function assertTooltip(page, text) {
  const tooltip = page.locator("#tooltip");
  await tooltip.waitFor({ state: "visible" });
  assert.match(await tooltip.textContent(), text);
}

async function checkAntsControls(page) {
  console.log("Ants & Drops: checking fleet controls");
  const ready = (count, type) =>
    page.waitForFunction(
      ({ count, type }) =>
        !document.getElementById("run").disabled &&
        document
          .getElementById("description")
          .textContent.startsWith(`${count} ${type}`) &&
        document
          .getElementById("footer-stats")
          .textContent.startsWith(`${count} BODIES`),
      { count, type },
    );
  assert.equal(await page.locator("#ants-controls").isVisible(), true);
  await page.locator("#scenario").selectOption("ants");
  await applyDraft(page);
  await ready(5, "harvesters");
  console.log("Ants & Drops: default fleet loaded");
  assert.equal(await page.locator("#ants-controls").isVisible(), true);
  assert.equal(
    await page.locator("#ants-vehicle-type").inputValue(),
    "harvester",
  );
  assert.equal(await page.locator("#ants-vehicle-count").inputValue(), "5");
  const count = page.locator("#ants-vehicle-count");
  // Keep validation checks cheap on headless software renderers after verifying
  // the default fleet. Native tests cover both vehicle types at the 128 limit.
  assert.equal(await count.getAttribute("max"), "128");
  await count.fill("1");
  await count.press("Tab");
  await applyDraft(page);
  await ready(1, "harvester");
  for (const invalid of ["", "0", "-1", "1.5", "129"]) {
    console.log(`Ants & Drops: reject count ${JSON.stringify(invalid)}`);
    await count.fill(invalid);
    await count.dispatchEvent("change");
    assert.equal(await count.evaluate((input) => input.checkValidity()), false);
    assert.match(
      await page.locator("#footer-stats").textContent(),
      /^1 BODIES/,
    );
    assert.equal(await page.locator("#run").isEnabled(), true);
  }
  await count.fill("1");
  console.log("Ants & Drops: loading one harvester");
  await count.press("Tab");
  await applyDraft(page);
  await ready(1, "harvester");
  await page.locator("#ants-vehicle-type").selectOption("drone");
  await applyDraft(page);
  console.log("Ants & Drops: loading one drone");
  await ready(1, "drone");
  await count.fill("3");
  await count.press("Tab");
  await applyDraft(page);
  await ready(3, "drones");
  assert.match(
    await page.locator("#footer-stats").textContent(),
    /9 ACTION DIMENSIONS/,
  );
  console.log("Ants & Drops: count limits and vehicle switching passed");
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000006",
  );
  await page.locator("#reset").click();
  await ready(3, "drones");
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000000",
  );
  assert.equal(await page.locator("#tick").textContent(), "TICK 000000");
  await page.locator("#save-menu").click();
  const downloading = page.waitForEvent("download");
  await page.locator("#export-scene").click();
  const stream = await (await downloading).createReadStream(),
    chunks = [];
  console.log("Ants & Drops: reading exported scene");
  for await (const chunk of stream) chunks.push(chunk);
  const buffer = Buffer.concat(chunks),
    scene = JSON.parse(buffer.toString());
  assert.equal(scene.bodies.length, 3);
  assert(
    scene.bodies.every(
      (body) => body.agent_type === "drone" && body.mass === undefined,
    ),
  );
  assert.equal(scene.pickups.length, 24);
  assert.equal(scene.respawn_seconds, 3);
  await page.locator("#files-dialog .dialog-close").click();
  await page.locator("#scenario").selectOption("racing");
  await applyDraft(page);
  await page.waitForFunction(
    () =>
      !document.getElementById("run").disabled &&
      document.getElementById("ants-vehicle-type").value === "kart",
  );
  await page.locator("#scenario").selectOption("ants");
  await applyDraft(page);
  await ready(3, "drones");
  // Import a saved fleet with edited initial conditions; do not regenerate it.
  scene.bodies[0].position = [10, 10];
  await page.locator("#save-menu").click();
  const choosing = page.waitForEvent("filechooser");
  await page.locator("#import-scene").click();
  await (
    await choosing
  ).setFiles({
    name: "ants.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(scene)),
  });
  await page.locator("#files-dialog .dialog-close").click();
  await applyDraft(page);
  await ready(3, "drones");
  await page.locator("#mode-edit").click();
  await page.locator("#edit-json").click();
  const restored = JSON.parse(await page.locator("#scene-json").inputValue());
  assert.deepEqual(restored, scene);
  await page.keyboard.press("Escape");
  await page.locator("#close-editor").click();
  await checkDropRendering(page);
  console.log("Ants & Drops: saved scenes and drop rendering passed");
}

async function checkDropRendering(page) {
  const result = await page.evaluate(async () => {
    const { LabRenderer } = await import("./renderer.js");
    const { loadNative, NativeEngine } = await import("./native.js");
    const { configureAntsScene } = await import("./ants-scene.js");
    const template = await (await fetch("./scenarios/ants.json")).json();
    const scene = configureAntsScene(template, {
      agentType: "harvester",
      count: 1,
    });
    scene.bodies[0].position = [10, 10];
    scene.pickups = [{ position: [10, 10], radius: 0.4 }];
    const engine = new NativeEngine(await loadNative(false), scene);
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:320px;height:240px";
    document.body.append(canvas);
    const renderer = new LabRenderer(canvas);
    try {
      renderer.load(scene, engine.info, engine.channels);
      engine.step(engine.neutralAction(), 1);
      renderer.update(engine.states());
      const hidden = !renderer.food[0].visible;
      const collected = engine.snapshot();
      engine.step(engine.neutralAction(), 181);
      renderer.update(engine.states());
      const visible = renderer.food[0].visible;
      const offset = engine.info[8],
        row = engine.states();
      const moved =
        renderer.food[0].position.x === row[offset] &&
        renderer.food[0].position.y === row[offset + 1] &&
        (row[offset] !== 10 || row[offset + 1] !== 10);
      engine.restore(collected);
      renderer.update(engine.states());
      return {
        hidden,
        visible,
        moved,
        replayHidden: !renderer.food[0].visible,
      };
    } finally {
      renderer.dispose();
      engine.dispose();
      canvas.remove();
    }
  });
  assert.deepEqual(result, {
    hidden: true,
    visible: true,
    moved: true,
    replayHidden: true,
  });
}

async function applyDraft(page) {
  await page.waitForFunction(() => !document.body.dataset.loadingPreset);
  if (await page.locator("#pending-settings").isVisible()) {
    await page.locator("#apply-configuration").click();
    await page.waitForFunction(
      () =>
        !document.querySelector("main").inert &&
        document.querySelector("#pending-settings").hidden,
    );
  }
}
