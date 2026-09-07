// Generic controllers, complete checkpoints, experiments, and durable replay.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
const output =
  process.env.CONTROL_SCREENSHOTS || "/tmp/fractal-control-browser";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1100 } }),
  errors = [];
page.setDefaultTimeout(30000);
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
const ready = () =>
  page.waitForFunction(() => !document.getElementById("run").disabled);
const tick = (n) =>
  page.waitForFunction(
    (n) =>
      document.getElementById("tick").textContent ===
      `TICK ${String(n).padStart(6, "0")}`,
    n,
  );
const download = async (id, name) => {
  const pending = page.waitForEvent("download");
  await page.locator("#" + id).click();
  const file = `${output}/${name}`;
  await (await pending).saveAs(file);
  return file;
};
const upload = async (id, file) => {
  const pending = page.waitForEvent("filechooser");
  await page.locator("#" + id).click();
  await (await pending).setFiles(file);
};
try {
  await page.goto(base);
  await ready();
  // Keep the replay fixture at six frames after checking the new default.
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000012",
  );
  await page.locator("#frames").fill("6");
  await page.locator("#frames").press("Tab");
  await ready();
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000006",
  );
  await tick(6);
  // Set all dimensions together; the selected controller triggers one rebuild.
  await page.evaluate(() => {
    for (const [id, value] of Object.entries({
      walkers: 12,
      horizon: 3,
      frames: 2,
      elites: 2,
    }))
      document.getElementById(id).value = value;
  });
  for (const algorithm of ["random", "cem", "icem", "mppi"]) {
    await page.locator("#algorithm").selectOption(algorithm);
    await ready();
    await tick(0);
    await page.locator("#step").click();
    await tick(2);
    if (algorithm === "icem") {
      const saved = await download("save-checkpoint", "icem-search.fgcp");
      await page.locator("#step").click();
      await tick(4);
      const future = await download("save-state", "icem-future.fgcs");
      await upload("load-checkpoint", saved);
      await ready();
      await page.waitForFunction(() =>
        document
          .getElementById("status")
          .textContent.includes("Planner restored"),
      );
      await tick(2);
      await page.locator("#step").click();
      await tick(4);
      const replayed = await download("save-state", "icem-restored.fgcs");
      assert.deepEqual(await readFile(replayed), await readFile(future));
    }
  }
  await page.locator("#inspect-physics").check();
  await page.waitForFunction(() =>
    document
      .getElementById("physics-readout")
      .textContent.includes("external force"),
  );
  assert.match(await page.locator("#performance-readout").textContent(), /FPS/);
  const checkpoint = await download("save-checkpoint", "search.fgcp");
  await page.locator("#step").click();
  await tick(4);
  const expected = await download("save-state", "checkpoint-future.fgcs");
  await upload("load-checkpoint", checkpoint);
  await ready();
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.includes("Planner restored"),
  );
  await tick(2);
  await page.locator("#step").click();
  await tick(4);
  const restored = await download("save-state", "checkpoint-restored.fgcs");
  assert.deepEqual(await readFile(restored), await readFile(expected));
  await page.locator("#inspect-physics").uncheck();
  await page
    .locator(".controls summary").filter({ hasText: "Planner settings" })
    .evaluate((e) => (e.parentElement.open = true));
  await page.locator("#wave").click();
  await page.waitForFunction(
    () =>
      document.getElementById("motion-segment").textContent ===
      "Recording executed world",
  );
  const waveCheckpoint = await download("save-checkpoint", "wave.fgcp");
  await page.locator("#wave").click();
  const waveFuture = await download("save-state", "wave-future.fgcs");
  await upload("load-checkpoint", waveCheckpoint);
  await ready();
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.startsWith("Wave restored"),
  );
  await page.locator("#wave").click();
  const waveRestored = await download("save-state", "wave-restored.fgcs");
  assert.deepEqual(await readFile(waveRestored), await readFile(waveFuture));

  await page.locator("#persistent-recording").check();
  await page.locator("#reset").click();
  await ready();
  await tick(0);
  await page.locator("#step").click();
  await tick(2);
  await page.locator("#event-note").fill("Persistent marker");
  await page.locator("#add-event").click();
  await page.locator("#flush-run").click();
  await page.waitForFunction(() =>
    document
      .getElementById("status")
      .textContent.includes("saved on this device"),
  );
  const recorded = await download("export-run", "persistent.fgcrec");
  await upload("import-run", recorded);
  await ready();
  await tick(0);
  assert.match(
    await page.locator("#motion-events").textContent(),
    /Persistent marker/,
  );
  await page.locator("#motion-live").click();
  await tick(2);
  await page.locator("#experiments").click();
  for (const [id, value] of Object.entries({
    "benchmark-seeds": "7",
    "benchmark-frames": "6",
    "benchmark-walkers": "8",
    "benchmark-horizon": "2",
    "benchmark-action-frames": "2",
    "benchmark-target": "6",
  }))
    await page.locator("#" + id).fill(value);
  await page.locator("#benchmark-goal").selectOption("survival");
  await page.locator("#variant-a").selectOption("icem");
  await page.locator("#variant-b").selectOption("mppi");
  await page.locator("#run-benchmark").click();
  await page.waitForFunction(
    () =>
      document.getElementById("experiment-status").textContent ===
      "Benchmark complete",
  );
  assert.equal(await page.locator("#benchmark-results tr").count(), 2);
  const report = JSON.parse(
    await readFile(
      await download("export-experiment", "benchmark.json"),
      "utf8",
    ),
  );
  assert.equal(report.results.length, 2);
  assert.equal(report.summary[0].successRate, 1);
  assert.ok(report.results.every((r) => r.simulatorFrames > 0));
  await page.locator("#compare-branches").click();
  await page.waitForFunction(
    () => document.querySelectorAll("#comparison-worlds canvas").length === 2,
  );
  await page.locator("#comparison-timeline").fill("6");
  await page.locator("#comparison-timeline").dispatchEvent("input");
  await page.locator("#experiment-dialog details summary").click();
  await page.locator("#profile-batch").click();
  await page.waitForFunction(() =>
    document.getElementById("batch-profile").textContent.includes("GB/s"),
  );
  await page.screenshot({ path: `${output}/experiments.png`, fullPage: true });
  await page.locator("#close-experiments").click();
  // Editor operations use scene-independent numeric and channel controls.
  await page.locator("#edit").click();
  await page.locator("#edit-json").click();
  await page.locator("#scene-json").fill(
    JSON.stringify({
      name: "Editor contract",
      size: [100, 100],
      bodies: [
        {
          controlled: true,
          position: [45, 50],
          radius: 1,
          mass: 1,
          actuator: { kind: "kart" },
        },
        {
          controlled: true,
          position: [55, 50],
          radius: 1,
          actuator: { kind: "holonomic" },
        },
      ],
      tethers: [{ a: 0, b: 1, rest_length: 10 }],
    }),
  );
  await page.locator("#apply-json").click();
  await ready();
  await tick(0);
  await page.locator("#view").click();
  const selectBody = async (index, shift = false) => {
    const box = await page.locator("#world").boundingBox();
    if (shift) await page.keyboard.down("Shift");
    await page.mouse.click(
      box.x + box.width / 2 + ((index === 0 ? -5 : 5) * box.height) / 132,
      box.y + box.height / 2,
    );
    if (shift) await page.keyboard.up("Shift");
  };
  await selectBody(0);
  assert.match(
    await page.locator("#selection-name").textContent(),
    /bodies \/ 0/,
  );
  assert.match(
    await page.locator("#entity-properties").textContent(),
    /mass \(kg\)/,
  );
  await page.locator('#entity-properties input[data-path="mass"]').fill("2");
  await page.locator("#apply-properties").click();
  await ready();
  await selectBody(0);
  await selectBody(1, true);
  assert.match(
    await page.locator("#selection-name").textContent(),
    /2 selected/,
  );
  await page.locator("#duplicate-entities").click();
  await ready();
  assert.match(await page.locator("#footer-stats").textContent(), /4 BODIES/);
  await page.locator("#edit-json").click();
  const duplicated = JSON.parse(await page.locator("#scene-json").inputValue());
  assert.equal(duplicated.tethers.length, 2);
  assert.deepEqual([duplicated.tethers[1].a, duplicated.tethers[1].b], [2, 3]);
  await page.keyboard.press("Escape");
  await page.locator("#undo").click();
  await ready();
  await selectBody(0);
  await page.locator("#template-name").fill("Custom kart");
  await page.locator("#save-template").click();
  await ready();
  assert.equal(await page.locator("#agent-type option").count(), 1);
  await page.locator("#editor details summary").click();
  assert.equal(await page.locator("#action-channels input").count(), 6);
  await page.locator("#action-channels input").first().fill("1");
  await page.locator("#apply-action").click();
  await tick(1);
  await page.locator("#close-editor").click();
  const canvas = await page.locator("#world").boundingBox();
  await page.mouse.move(
    canvas.x + canvas.width / 2,
    canvas.y + canvas.height / 2,
  );
  await page.mouse.down({ button: "right" });
  await page.mouse.move(
    canvas.x + canvas.width / 2 + 50,
    canvas.y + canvas.height / 2 + 20,
  );
  await page.mouse.up({ button: "right" });
  await page.screenshot({
    path: `${output}/editor-controls.png`,
    fullPage: true,
  });
  // Exercise more than eight resident chunks, export/import, and cold reopening.
  const stored = await page.evaluate(async () => {
    const fixture = await new Promise((resolve, reject) => {
      const code = `import {loadNative,NativeEngine} from '${location.origin}/lab/native.js';import {WorldCapture} from '${location.origin}/lab/motion.js';const scene={name:'Storage stress test',size:[100,100],bodies:[{controlled:true,position:[50,50],velocity:[.1,0],drag:0,actuator:{kind:'kart'}}]};const e=new NativeEngine(await loadNative(false),scene);e.reset(17);const root=e.snapshot();const capture=new WorldCapture(e,({packet})=>postMessage({packet,info:e.info,root,channels:e.channels,scene},[packet.buffer]));capture.step(new Float32Array(e.dim),2401,0);e.dispose();`;
      const url = URL.createObjectURL(
        new Blob([code], { type: "text/javascript" }),
      );
      const worker = new Worker(url, { type: "module" });
      worker.onmessage = ({ data }) => {
        worker.terminate();
        URL.revokeObjectURL(url);
        resolve(data);
      };
      worker.onerror = (e) => {
        worker.terminate();
        reject(new Error(e.message));
      };
    });
    const { StoredMotionRecording, importStoredFile } = await import(
      "/lab/storage/recording-store.js"
    );
    const r = new StoredMotionRecording(fixture.info, fixture.root, 1 / 60, {
      scene: fixture.scene,
      settings: { algorithm: "random" },
    });
    r.channels = fixture.channels;
    r.append(fixture.packet, "Initial");
    r.addEvent(123, "Saved event");
    await r.flush();
    if (r.chunks.filter(Boolean).length > 8)
      throw new Error("Resident chunk budget exceeded");
    const first = Array.from(new Uint8Array((await r.getRows(123)).buffer));
    r.saveObject("checkpoint", 1, {
      bytes: new Uint8Array([1, 2, 3]),
      rng: 17,
    });
    await r.flush();
    const size = r.storedBytes;
    await r.flush();
    if (r.storedBytes !== size)
      throw new Error("Stored byte count changed on repeat flush");
    const file = await r.exportFile(),
      imported = await importStoredFile(file);
    const copy = Array.from(
      new Uint8Array((await imported.getRows(123)).buffer),
    );
    if (JSON.stringify(first) !== JSON.stringify(copy))
      throw new Error("Compressed replay mismatch");
    const bytes = new Uint8Array(await file.arrayBuffer());
    bytes[bytes.length - 1] ^= 1;
    let rejected = false;
    try {
      await importStoredFile(new Blob([bytes]));
    } catch {
      rejected = true;
    }
    if (!rejected) throw new Error("Corrupt archive accepted");
    return {
      id: r.id,
      length: r.length,
      first,
      raw: r.bytes,
      compressed: file.size,
    };
  });
  assert.equal(stored.length, 2401);
  assert.ok(stored.compressed < stored.raw);
  await page.reload();
  await ready();
  await tick(6);
  const reopened = await page.evaluate(async (id) => {
    const { StoredMotionRecording } = await import(
      "/lab/storage/recording-store.js"
    );
    const r = await StoredMotionRecording.open(id);
    return {
      length: r.length,
      first: Array.from(new Uint8Array((await r.getRows(123)).buffer)),
      saved: await r.loadObject("checkpoint", 1),
      events: r.events,
    };
  }, stored.id);
  assert.equal(reopened.length, stored.length);
  assert.deepEqual(reopened.first, stored.first);
  assert.equal(reopened.saved.rng, 17);
  assert.equal(reopened.events.at(-1).label, "Saved event");
  await page.locator("#open-library").click();
  await page.waitForFunction(
    () => document.querySelectorAll(".stored-run").length >= 3,
  );
  await page.locator("#close-storage").click();
  assert.deepEqual(errors, []);
  console.log(
    "Browser passed: variable controllers, exact planner continuation, experiments, fork comparison, instrumentation, compressed storage, eviction, corruption rejection and cold replay.",
  );
} catch (error) {
  console.error(
    "Lab status:",
    await page.locator("#status").textContent(),
    "Browser errors:",
    errors,
  );
  await page.screenshot({
    path: `${output}/experiments-error.png`,
    fullPage: true,
  });
  throw error;
} finally {
  await browser.close();
}
