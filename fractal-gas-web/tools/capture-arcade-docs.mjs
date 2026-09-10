// Capture the user-guide screenshots from the real Arcade browser application.
// Build the arcade WASM first, then run serve.py and invoke:
//   ARCADE_DOC_URL=http://127.0.0.1:8091/web/arcade.html npm run capture:arcade-docs
// The local plaintext ROM fixtures are used when available; encrypted ROMs are
// never created or modified by this capture workflow.

import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { chromium } from "playwright";

const repo = fileURLToPath(new URL("../../", import.meta.url));
const base = process.env.ARCADE_DOC_URL || "http://127.0.0.1:8091/web/arcade.html";
const output =
  process.env.ARCADE_SCREENSHOTS || resolve(repo, "docs/_static/arcade_lab");
const viewport = { width: 1536, height: 1100 };

await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const captures = {};
const failures = [];
let page;

async function openArcade() {
  if (page) await page.close();
  page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  page.setDefaultTimeout(120000);
  page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    failures.push(`requestfailed: ${request.url()} ${request.failure()?.errorText}`);
  });
  await page.goto(base);
  await ready();
}

async function ready() {
  await page.waitForFunction(
    () => !document.getElementById("btn-start").disabled,
  );
  await page.waitForFunction(
    () => document.getElementById("status").textContent.startsWith("Ready"),
  );
}

async function setField(id, value) {
  const field = page.locator(`#${id}`);
  if ((await field.inputValue()) === String(value)) return;
  await field.fill(String(value));
  await field.press("Tab");
  if (["param-n", "param-seed", "param-max-walkers"].includes(id)) {
    await ready();
  }
}

async function configureSmallRun() {
  await setField("param-n", 4);
  await setField("param-dt-min", 2);
  await setField("param-dt-max", 2);
  await ready();
}

// A running swarm keeps the main thread busy for hundreds of milliseconds at
// a time, so trusted input events are not acknowledged and a Playwright click
// on Pause times out. Dispatch the click from inside the page instead.
async function pauseRun() {
  const paused = await page.evaluate(() => {
    const pause = document.getElementById("btn-pause");
    if (pause.disabled) return false; // the run already ended on its own
    pause.click();
    return true;
  });
  if (!paused) return;
  await page.waitForFunction(
    () => document.getElementById("status").textContent === "Paused" ||
      !document.getElementById("run-ended").hidden,
  );
}

async function runIterations(target) {
  await page.locator("#btn-start").click();
  await page.waitForFunction(
    (target) =>
      Number(document.getElementById("stat-iteration").textContent) >= target ||
      !document.getElementById("run-ended").hidden,
    target,
  );
  await pauseRun();
}

async function runPlayedFrames(target) {
  await page.locator("#btn-start").click();
  await page.waitForFunction(
    (target) =>
      Number(document.getElementById("stat-played-frames").textContent.replaceAll(",", "")) >= target ||
      !document.getElementById("run-ended").hidden,
    target,
  );
  await pauseRun();
}

async function selectConsole(id) {
  await page.locator(`[data-console="${id}"]`).click();
  await ready();
}

async function selectAlgorithm(id) {
  await page.locator(`[data-algo="${id}"]`).click();
  await ready();
}

async function capture(name, note) {
  await page.mouse.move(10, 10);
  await page.evaluate(() => document.activeElement?.blur());
  await page.screenshot({
    path: resolve(output, `${name}.png`),
    fullPage: false,
    animations: "disabled",
    timeout: 60000,
  });
  captures[name] = await page.evaluate(({ note }) => {
    const value = (id) => document.getElementById(id)?.value ?? null;
    const active = (selector) =>
      document.querySelector(`${selector}.active`)?.textContent?.trim() ?? null;
    return {
      note,
      viewport: [window.innerWidth, window.innerHeight],
      console: active("#console-select button"),
      algorithm: active("#algo-select button"),
      observation: active("#obs-mode button"),
      atariGame: value("atari-game"),
      world: value("param-world"),
      stage: value("param-stage"),
      zone: value("param-zone"),
      act: value("param-act"),
      walkers: value("param-n"),
      seed: value("param-seed"),
      elite: value("param-elite"),
      dtMin: value("param-dt-min"),
      dtMax: value("param-dt-max"),
      horizon: value("param-horizon"),
      consensus: document.getElementById("param-consensus")?.checked ?? null,
      maxHorizon: value("param-max-horizon"),
      visitReward: active("#visit-reward-select button"),
      iteration: document.getElementById("stat-iteration")?.textContent ?? null,
      playedFrames: document.getElementById("stat-played-frames")?.textContent ?? null,
      status: document.getElementById("status")?.textContent ?? null,
    };
  }, { note });
  await writeFile(
    resolve(output, "capture-manifest.json"),
    JSON.stringify({ source: "capture-arcade-docs.mjs", viewport, captures }, null, 2) + "\n",
  );
  console.log(`Captured ${name}.png`);
}

try {
  await openArcade();
  await capture(
    "arcade-overview",
    "Mario Wave workspace at reset with the shared sidebar, screen, map, and plots.",
  );

  await configureSmallRun();
  await runIterations(5);
  await capture(
    "mario-map",
    "Mario Wave after five short iterations, showing the level-map swarm overlay.",
  );

  await openArcade();
  await configureSmallRun();
  await selectConsole(1);
  await page.locator("#atari-game").selectOption("breakout");
  await ready();
  await runIterations(2);
  await capture(
    "atari-breakout",
    "Generic Atari Breakout with the game picker and score-driven workspace visible.",
  );

  await openArcade();
  await configureSmallRun();
  await selectConsole(2);
  await runIterations(12);
  await capture(
    "sonic-fog",
    "Sonic after short exploration, with the swarm-built fog-of-war map revealed.",
  );

  await openArcade();
  await configureSmallRun();
  await selectConsole(3);
  await runIterations(12);
  await capture(
    "montezuma-pyramid",
    "Montezuma after short exploration, with discovered rooms placed on the pyramid map.",
  );

  await openArcade();
  await configureSmallRun();
  // The planning panel (and its horizon field) only exists once a planner
  // algorithm is selected, so pick FMC before touching the horizon.
  await selectAlgorithm(2);
  await setField("param-horizon", 3);
  await runPlayedFrames(4);
  await capture(
    "arcade-planner",
    "FMC planning session showing the committed game, search controls, and played-game stats.",
  );

  await openArcade();
  await configureSmallRun();
  await selectAlgorithm(1);
  await runIterations(8);
  await page.waitForFunction(
    () => !document.getElementById("map-visits").hidden,
  );
  await page.locator("#map-visits").click();
  await page.waitForFunction(
    () => document.getElementById("map-title").textContent.includes("visits"),
  );
  await capture(
    "graph-visits",
    "Graph with Coords and visit reward enabled, showing the accumulated visit heatmap.",
  );
} finally {
  if (page) await page.close();
  await browser.close();
}

assert.deepEqual(failures, [], failures.join("\n"));
