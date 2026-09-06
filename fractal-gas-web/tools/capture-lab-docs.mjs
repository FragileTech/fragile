// Rebuild the Lab first when native code changes, then run its preview server:
//   make control-web
//   CONTROL_PORT=8097 make control-lab
//   CONTROL_TEST_URL=http://127.0.0.1:8097/lab/ node fractal-gas-web/tools/capture-lab-docs.mjs
// Optional: CONTROL_CAPTURE_GROUP=tasks|tracks|editor; CONTROL_SCREENSHOTS=/tmp/lab-captures.
// These are real browser captures. Nothing substitutes for physics or renderer output.
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { chromium } from "playwright";

const repo = fileURLToPath(new URL("../../", import.meta.url));
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8097/lab/";
const output =
  process.env.CONTROL_SCREENSHOTS ||
  resolve(repo, "docs/_static/control_lab/tutorials");
const group = process.env.CONTROL_CAPTURE_GROUP || process.argv[2] || "all";
const only = process.env.CONTROL_CAPTURE_ONLY;
assert(
  ["all", "tasks", "tracks", "editor"].includes(group),
  "Unknown capture group",
);
const viewport = { width: 1536, height: 1100 };
await mkdir(output, { recursive: true });
let browser, page;
const errors = [];
let top = false;
let captures = {};
try {
  captures = JSON.parse(
    await readFile(resolve(output, "capture-manifest.json")),
  ).captures;
} catch {
  /* First capture run. */
}
const ready = async () => {
  await page.waitForFunction(() => !document.getElementById("run").disabled);
};
async function openLab() {
  // Fresh renderer processes bound GLB/GPU memory across long capture runs.
  if (browser) await browser.close();
  browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--use-angle=swiftshader"],
  });
  page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  page.setDefaultTimeout(120000);
  page.on("pageerror", (error) => {
    errors.push(error.message);
    console.error(error.message);
  });
  page.on("crash", () => console.error("Capture page crashed"));
  page.on("console", (message) => {
    if (message.type() === "error") console.error("Browser:", message.text());
  });
  page.on("requestfailed", (request) =>
    console.error(
      "Request failed:",
      request.url(),
      request.failure()?.errorText,
    ),
  );
  top = false;
  await page.goto(base);
  await ready();
  await graphicsReady();
  await page.locator(".controls > details > summary").click();
  await field("threads", 1);
  await page.locator(".controls > details > summary").click();
}
async function graphicsReady() {
  await page.waitForFunction(
    () =>
      document.documentElement.dataset.visualStyle === "futuristic" &&
      document.getElementById("style-status").textContent === "" &&
      !document.getElementById("world").getContext("webgl2").isContextLost() &&
      !document
        .getElementById("performance-readout")
        .textContent.includes("· 0 draws"),
  );
  await page.waitForTimeout(400);
}
async function capture(name, note) {
  await graphicsReady();
  await page.mouse.move(10, 10);
  await page.screenshot({
    path: resolve(output, `${name}.png`),
    fullPage: false,
    timeout: 60000,
  });
  captures[name] = await page.evaluate(
    ({ note, top }) => ({
      note,
      view: top ? "top-down" : "angled",
      title: document.getElementById("scene-title").textContent,
      tick: document.getElementById("tick").textContent,
      settings: Object.fromEntries(
        [
          "scenario",
          "algorithm",
          "clock",
          "walkers",
          "horizon",
          "frames",
          "seed",
          "threads",
          "recording",
        ].map((id) => [id, document.getElementById(id).value]),
      ),
      editor: !document.getElementById("editor").hidden,
      selection: document.getElementById("selection-name").textContent,
    }),
    { note, top },
  );
  await writeFile(
    resolve(output, "capture-manifest.json"),
    JSON.stringify({ viewport, style: "futuristic", captures }, null, 2) + "\n",
  );
  console.log(`Captured ${name}: ${captures[name].tick}`);
}
async function view(wantTop) {
  if (top !== wantTop) {
    await page.locator("#view").click();
    top = wantTop;
  }
}
async function field(id, value) {
  if ((await page.locator(`#${id}`).inputValue()) === String(value)) return;
  await page.locator(`#${id}`).fill(String(value));
  await page.locator(`#${id}`).press("Tab");
  await ready();
}
async function reset() {
  await page.locator("#reset").click();
  await ready();
  assert.equal(await page.locator("#tick").textContent(), "TICK 000000");
}
async function step() {
  const before = await page.locator("#tick").textContent();
  await page.locator("#step").click();
  await page.waitForFunction(
    (before) => document.getElementById("tick").textContent !== before,
    before,
  );
  await ready();
}
async function editor(open = true) {
  if ((await page.locator("#editor").isVisible()) !== open)
    await page.locator(open ? "#edit" : "#close-editor").click();
}
async function loadExample(name) {
  const scene = JSON.parse(
    await readFile(
      resolve(repo, `docs/_static/control_lab/examples/${name}.json`),
    ),
  );
  await editor();
  await page.locator("#edit-json").click();
  await page.locator("#scene-json").fill(JSON.stringify(scene, null, 2));
  await page.locator("#apply-json").click();
  await ready();
  await view(true);
  await page.locator("#tool").selectOption("select");
  await graphicsReady();
  // Allow the camera and editor layout to settle before projecting pointer edits.
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  return scene;
}
// Project a known scene point using the renderer's documented whole-arena
// orthographic framing. All edits still go through pointer events and the UI.
async function worldClick(scene, point, modifiers) {
  assert(top, "Scene editing capture uses the top-down whole-arena view");
  const box = await page.locator("#world").boundingBox();
  const span =
    Math.max(scene.size[1], scene.size[0] / (box.width / box.height)) * 0.66;
  const scale = box.height / (2 * span);
  for (const key of modifiers || []) await page.keyboard.down(key);
  const x = box.x + box.width / 2 + (point[0] - scene.size[0] / 2) * scale;
  const y = box.y + box.height / 2 - (point[1] - scene.size[1] / 2) * scale;
  assert.equal(
    await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.id, {
      x,
      y,
    }),
    "world",
    `World point ${point} is covered by another UI element`,
  );
  await page.mouse.click(x, y);
  await page.waitForTimeout(200);
  for (const key of modifiers || []) await page.keyboard.up(key);
}
async function readScene() {
  await page.locator("#edit-json").click();
  const scene = JSON.parse(await page.locator("#scene-json").inputValue());
  await page
    .locator('#json-dialog button[aria-label="Close JSON editor"]')
    .click();
  return scene;
}
try {
  console.log("Loading Lab and its authored models…");
  if (group === "all" || group === "tasks") {
    const catalog = JSON.parse(
      await readFile(
        resolve(repo, "fractal-gas-web/web/lab/scenario-catalog.json"),
      ),
    );
    for (const task of catalog) {
      if (only && only !== task.id) continue;
      console.log(`Preparing ${task.id}…`);
      await openLab();
      await page.locator("#scenario").selectOption(task.id);
      await ready();
      await view(false);
      await reset();
      await capture(
        `${task.id}-overview`,
        "Shipped task at its initial state; FMC 128/16/6, seed 7, one thread.",
      );
      if (task.id === "ants") {
        await field("ants-vehicle-count", 4);
        await page.locator("#ants-vehicle-type").selectOption("drone");
        await ready();
      }
      if (task.id === "rocket") {
        await page.locator("#recording").selectOption("2");
        await ready();
        await page.locator("#archive-all").check();
      }
      await view(true);
      await step();
      await capture(
        `${task.id}-detail`,
        task.id === "ants"
          ? "Four drones after one decision; the original default is 48 harvesters."
          : "Top-down task view after one completed decision; paths describe planning, not task completion.",
      );
      if (task.id === "rocket") {
        await page.locator("#replay").click();
        await page.waitForFunction(
          () => document.getElementById("run-state").textContent === "REPLAY",
        );
        await capture(
          "rocket-replay",
          "Replay branch restores a selected search branch; compare with executed world replay.",
        );
        await page.locator("#recording").selectOption("1");
        await ready();
        await page.locator("#archive-all").uncheck();
      }
    }
  }
  if (["all", "tasks", "tracks"].includes(group)) {
    const catalog = JSON.parse(
      await readFile(
        resolve(repo, "fractal-gas-web/web/lab/scenario-catalog.json"),
      ),
    );
    for (const track of catalog.find((task) => task.id === "racing").tracks) {
      if (only && only !== track.id) continue;
      console.log(`Preparing track ${track.id}…`);
      await openLab();
      await page.locator("#scenario").selectOption("racing");
      await ready();
      if (track.id !== "racing") {
        await page.locator("#track").selectOption(track.id);
        await ready();
      }
      await view(true);
      await capture(
        `track-${track.id}`,
        "Circuit at reset with the real track preview, checkpoint count, and physical layout.",
      );
    }
  }
  if (group === "all" || group === "editor") {
    let scene;
    if (!only || only === "foraging") {
      await openLab();
      scene = await loadExample("foraging-starter");
      await capture(
        "editor-overview",
        "Foraging workshop starter with scene editor open in top-down view.",
      );
      await worldClick(scene, [8, 8]);
      assert.match(
        await page.locator("#selection-name").textContent(),
        /bodies \/ 0/,
      );
      await page
        .locator('#entity-properties input[data-path="mass"]')
        .fill("2");
      await page
        .locator('#entity-properties input[data-path="drag"]')
        .fill("0.4");
      await page.locator("#apply-properties").scrollIntoViewIfNeeded();
      await capture(
        "editor-properties",
        "Mass 2 kg and drag 0.4 entered for the selected drone, before Apply properties.",
      );
      await page.locator("#apply-properties").click();
      await ready();
      assert.equal(await page.locator("#selection-name").textContent(), "None");
      assert.equal((await readScene()).bodies[0].mass, 2);
      await worldClick(scene, [16, 8]);
      await worldClick(scene, [24, 8], ["Shift"]);
      assert.match(
        await page.locator("#selection-name").textContent(),
        /2 selected/,
      );
      await page.locator("#entity").scrollIntoViewIfNeeded();
      await capture(
        "editor-multiselect",
        "Two pickup entities selected with Shift-click; entity fields show the last selection.",
      );
      await page.locator("#duplicate-entities").click();
      await ready();
      assert.equal((await readScene()).pickups.length, 5);
      await page.locator("#undo").click();
      await ready();
      assert.equal((await readScene()).pickups.length, 3);
      await page.locator("#redo").click();
      await ready();
      assert.equal((await readScene()).pickups.length, 5);
    }
    if (!only || only === "cargo") {
      await openLab();
      scene = await loadExample("cargo-starter");
      await page.locator("#tool").selectOption("tether");
      await worldClick(scene, [8, 15]);
      assert.match(await page.locator("#status").textContent(), /second body/i);
      await worldClick(scene, [13, 15]);
      await ready();
      assert.equal((await readScene()).tethers.length, 1);
      await page.locator("#tool").scrollIntoViewIfNeeded();
      await capture(
        "editor-tether",
        "Cargo workshop after connecting the tug and cargo using the two-click tether tool.",
      );
      await page.locator("#tool").selectOption("holes");
      for (const point of [
        [19, 3],
        [22, 3],
        [22, 6],
        [19, 6],
      ])
        await worldClick(scene, point);
      await page.locator("#finish-polygon").click();
      await ready();
      assert.equal((await readScene()).holes.length, 1);
      await page.locator("#tool").scrollIntoViewIfNeeded();
      await capture(
        "editor-polygon",
        "A rectangular hole compiled from four vertices; its wall excludes the interior from playable space.",
      );
      await page.locator("#tool").selectOption("select");
      await worldClick(scene, [8, 15]);
      await page.locator("#template-name").fill("workshop_tug");
      await page.locator("#save-template").scrollIntoViewIfNeeded();
      await capture(
        "editor-template",
        "The selected tug can be saved as the scene-local workshop_tug agent type.",
      );
      await page.locator("#save-template").click();
      await ready();
      assert.ok((await readScene()).agent_types.workshop_tug);
    }
    if (!only || only === "kart") {
      await openLab();
      scene = await loadExample("kart-finished");
      await worldClick(scene, [8, 12]);
      await page.locator("#editor details > summary").click();
      await page.locator("#apply-action").scrollIntoViewIfNeeded();
      const sliders = page.locator("#action-channels input");
      assert.equal(await sliders.count(), 3);
      await sliders.nth(0).fill("0.5");
      await capture(
        "editor-actuators",
        "Kart throttle set to 0.50, steering and brake neutral; Apply action advances one physics frame.",
      );
      await page.locator("#apply-action").click();
      await page.waitForFunction(
        () => document.getElementById("tick").textContent === "TICK 000001",
      );
      await page.locator("#edit-json").click();
      await capture(
        "editor-json",
        "Complete scene JSON for the downloadable finished kart workshop.",
      );
      await page
        .locator("#scene-json")
        .fill('{\n  "version": 1,\n  "name": "Broken JSON"\n  "bodies": []\n}');
      await page.locator("#apply-json").click();
      assert.notEqual(await page.locator("#json-error").textContent(), "");
      await capture(
        "editor-error",
        "A missing comma is reported in the JSON dialog; correct the text before compiling again.",
      );
      await page.locator("#scene-json").fill(JSON.stringify(scene, null, 2));
      await page.locator("#apply-json").click();
      await ready();
      const download = page.waitForEvent("download");
      await page.locator("#export-scene").click();
      const exported = await download;
      const exportedPath = await exported.path();
      assert.deepEqual(JSON.parse(await readFile(exportedPath)), scene);
      const chooser = page.waitForEvent("filechooser");
      await page.locator("#import-scene").click();
      await (await chooser).setFiles(exportedPath);
      await ready();
      assert.deepEqual(await readScene(), scene);
    }
    console.log(
      "Selected editor workflow checks passed.",
    );
  }
  assert.deepEqual(errors, []);
} catch (error) {
  console.error(error);
  console.error(
    await page
      .locator("body")
      .innerText({ timeout: 5000 })
      .catch(() => "Page unresponsive"),
  );
  await page
    .screenshot({ path: resolve(output, "capture-error.png"), timeout: 10000 })
    .catch(() => {});
  process.exitCode = 1;
} finally {
  await browser?.close();
}
