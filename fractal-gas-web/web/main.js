// Main-thread UI: sidebar controls, canvas rendering of the best walker,
// stat readouts and the five plots (mirroring the Panel dashboard panes).

import { LinePlot } from "./plots.js";

const $ = (id) => document.getElementById(id);

const statusEl = $("status");
const screenCanvas = $("screen");
const screenCtx = screenCanvas.getContext("2d");

const plots = {
  reward: new LinePlot($("plot-reward"), {
    title: "Cumulative reward",
    series: [
      { name: "max", color: "#7ee787" },
      { name: "mean", color: "#58a6ff" },
    ],
  }),
  vr: new LinePlot($("plot-vr"), {
    title: "Virtual reward",
    series: [
      { name: "max", color: "#d2a8ff" },
      { name: "mean", color: "#bc8cff" },
    ],
  }),
  clones: new LinePlot($("plot-clones"), {
    title: "Cloned walkers (%)",
    series: [{ name: "clone %", color: "#ffa657" }],
  }),
  alive: new LinePlot($("plot-alive"), {
    title: "Alive walkers",
    series: [{ name: "alive", color: "#7ee787" }],
  }),
  dt: new LinePlot($("plot-dt"), {
    title: "Mean frame skip (dt)",
    series: [{ name: "mean dt", color: "#79c0ff" }],
  }),
};

// ---------------------------------------------------------------------------
// Level map + swarm overlay (Mario only). A full-level map image
// (web/maps/mario-W-S.gif, 1 map pixel = 1 world pixel) is drawn at the
// aspect-true zoom set by the viewport height (drag the resize handle),
// with every walker's RAM x-position overlaid as a dot — the whole gas
// spreading through the level, best walker highlighted. When the map is
// zoomed past the panel width a horizontal scrollbar appears, auto-scrolled
// to follow the best walker unless the user scrolled recently.
// ---------------------------------------------------------------------------
const mapCanvas = $("level-map");
const mapResize = $("map-resize");
const mapCtx = mapCanvas.getContext("2d");
const mapImages = new Map(); // "world-stage" -> {img, ok, failed}
let lastSwarm = null; // per-walker arrays from the latest step stats
// Default viewport height: 2.5x the aspect-fit height, so the strip is
// actually readable. Applied on first load and on level change; a manual
// drag of the resize handle sticks for the rest of the run.
const MAP_DEFAULT_STRETCH = 2.5;
let mapSizedForKey = null; // "world-stage" the default height was applied for
// Auto-follow: keep the best walker centered in the scrolled viewport,
// yielding to a manual scroll for a few seconds so browsing is possible.
const MAP_FOLLOW_GRACE_MS = 4000;
let mapExpectedScroll = -1; // scrollLeft we set programmatically
let mapUserScrolledAt = -Infinity;
mapResize.addEventListener("scroll", () => {
  if (Math.abs(mapResize.scrollLeft - mapExpectedScroll) > 2) {
    mapUserScrolledAt = performance.now();
  }
});

function getMapImage(world, stage) {
  const key = `${world}-${stage}`;
  let entry = mapImages.get(key);
  if (!entry) {
    const img = new Image();
    entry = { img, ok: false, failed: false };
    img.onload = () => { entry.ok = true; drawMap(); };
    img.onerror = () => { entry.failed = true; drawMap(); };
    img.src = `maps/mario-${world}-${stage}.gif`;
    mapImages.set(key, entry);
  }
  return entry;
}

// ---------------------------------------------------------------------------
// Sonic fog-of-war: the map is BUILT by the swarm. Each walker step ships a
// 40x28 RGB tile (its 320x224 frame downsampled 8x) plus its camera
// position; tiles are stitched into a per-level offscreen canvas at 1/8
// level scale. Unexplored areas stay dark until some walker sees them.
// ---------------------------------------------------------------------------
const FOG_SCALE = 8;
const TILE_W = 40, TILE_H = 28;
// Display scale cap at "fit" (css px per fog px) and the +/- zoom state.
const SONIC_FIT_MAX_SCALE = 3;
const SONIC_ZOOM_STEP = 1.5;
const SONIC_ZOOM_MAX = 16;
let sonicMapZoom = 1;
const fogCanvases = new Map(); // "zone-act" -> {canvas, ctx}

function getFogCanvas(zone, act, needW, needH) {
  const key = `${zone}-${act}`;
  let entry = fogCanvases.get(key);
  if (!entry) {
    const canvas = document.createElement("canvas");
    canvas.width = 1408;  // GHZ-sized default (~11k px / 8), grows on demand
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0a0b0f";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Explored bounding box (fog-canvas pixels), so the display can crop
    // to what the swarm has actually revealed.
    entry = { canvas, ctx, minX: Infinity, minY: Infinity, maxX: 0, maxY: 0 };
    fogCanvases.set(key, entry);
  }
  if (needW > entry.canvas.width || needH > entry.canvas.height) {
    const grown = document.createElement("canvas");
    grown.width = Math.max(entry.canvas.width, Math.ceil(needW / 256) * 256);
    grown.height = Math.max(entry.canvas.height, Math.ceil(needH / 256) * 256);
    const gctx = grown.getContext("2d");
    gctx.fillStyle = "#0a0b0f";
    gctx.fillRect(0, 0, grown.width, grown.height);
    gctx.drawImage(entry.canvas, 0, 0);
    entry.canvas = grown;
    entry.ctx = gctx;
  }
  return entry;
}

// The Sonic HUD is drawn at fixed screen positions (SCORE/TIME/RINGS
// top-left, lives bottom-left), so those tile pixels are screen furniture,
// not level. Mask them out (alpha 0) when stitching: walkers whose cameras
// sit elsewhere see the same world area without the HUD on it and fill the
// hole with clean pixels — the stitched map self-heals.
const fogHudMask = (() => {
  const mask = new Uint8Array(TILE_W * TILE_H);
  for (let y = 0; y < TILE_H; y++) {
    for (let x = 0; x < TILE_W; x++) {
      const hudTop = x <= 15 && y <= 8;      // SCORE / TIME / RINGS block
      const hudLives = x <= 11 && y >= 23;   // lives counter
      if (hudTop || hudLives) mask[y * TILE_W + x] = 1;
    }
  }
  return mask;
})();
const fogTileImage = new ImageData(TILE_W, TILE_H);
const fogTileScratch = document.createElement("canvas");
fogTileScratch.width = TILE_W;
fogTileScratch.height = TILE_H;
const fogTileScratchCtx = fogTileScratch.getContext("2d");
function applyFogTiles(swarm, tilesBuf) {
  const tiles = new Uint8Array(tilesBuf);
  const rgba = fogTileImage.data;
  for (let i = 0; i < swarm.xs.length; i++) {
    const camX = Math.floor(swarm.camXs[i] / FOG_SCALE);
    const camY = Math.floor(swarm.camYs[i] / FOG_SCALE);
    const entry = getFogCanvas(swarm.ws[i], swarm.ss[i],
                               camX + TILE_W, camY + TILE_H);
    const base = i * TILE_W * TILE_H * 3;
    for (let px = 0; px < TILE_W * TILE_H; px++) {
      rgba[px * 4 + 0] = tiles[base + px * 3 + 0];
      rgba[px * 4 + 1] = tiles[base + px * 3 + 1];
      rgba[px * 4 + 2] = tiles[base + px * 3 + 2];
      rgba[px * 4 + 3] = fogHudMask[px] ? 0 : 255;
    }
    // putImageData ignores compositing (it would stamp the transparent HUD
    // holes over already-clean pixels), so go through a scratch canvas and
    // drawImage, which is source-over.
    fogTileScratchCtx.putImageData(fogTileImage, 0, 0);
    entry.ctx.drawImage(fogTileScratch, camX, camY);
    entry.minX = Math.min(entry.minX, camX);
    entry.minY = Math.min(entry.minY, camY);
    entry.maxX = Math.max(entry.maxX, camX + TILE_W);
    entry.maxY = Math.max(entry.maxY, camY + TILE_H);
  }
}

function clearFog() {
  fogCanvases.clear();
}

// The maps crop the NES 240px frame to 224px (8px overscan top/bottom), so
// on-screen y-pixels map to image rows with an 8px shift.
const MAP_Y_CROP = 8;
// ram[0x3B8] references the top of the 32px (big-Mario) sprite box: measured
// standing on 1-1's floor (map ground top y=200) it reads 176, so +32 lands
// on the feet row. Anchor dots slightly above the feet so a grounded walker
// sits on the ground line instead of half-sunk into it.
const MARIO_FEET_OFFSET = 32;
function marioMapY(ramY, imgH) {
  return Math.min(
    Math.max(ramY + MARIO_FEET_OFFSET - MAP_Y_CROP - 4, 0), imgH);
}

function drawMap() {
  if (consoleId === 1) return;
  if (consoleId === 2) return drawSonicMap();
  // Displayed level: the best walker's current level, or the selected start
  // level before the run produces stats (so the big picture shows up front).
  const world = lastSwarm ? lastSwarm.world : parseInt($("param-world").value, 10) || 1;
  const stage = lastSwarm ? lastSwarm.level : parseInt($("param-stage").value, 10) || 1;
  const entry = getMapImage(world, stage);

  const panelWidth = mapResize.clientWidth || 900;
  const dpr = window.devicePixelRatio || 1;
  const imgW = entry.ok ? entry.img.naturalWidth : 3400;
  const imgH = entry.ok ? entry.img.naturalHeight : 224;

  // Apply the default zoomed height on first draw of each level (once the
  // image's real aspect is known); a manual resize of the wrapper is left
  // alone afterwards.
  const key = `${world}-${stage}`;
  if (entry.ok && mapSizedForKey !== key) {
    mapSizedForKey = key;
    mapResize.style.height =
      Math.round(imgH * (panelWidth / imgW) * MAP_DEFAULT_STRETCH) + "px";
  }

  // Aspect-true zoom: the viewport height sets a uniform scale; the canvas
  // grows as wide as the level needs and the wrapper scrolls horizontally.
  const cssHeight = mapResize.clientHeight || 150;
  const scale = cssHeight / imgH;
  const cssWidth = Math.max(1, Math.round(imgW * scale));
  mapCanvas.width = Math.round(cssWidth * dpr);
  mapCanvas.height = Math.round(cssHeight * dpr);
  mapCanvas.style.width = cssWidth + "px";
  mapCanvas.style.height = cssHeight + "px";

  // Draw in css-pixel space so dots/lines keep fixed on-screen sizes.
  mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  mapCtx.clearRect(0, 0, cssWidth, cssHeight);
  if (entry.ok) {
    mapCtx.drawImage(entry.img, 0, 0, cssWidth, cssHeight);
  } else {
    mapCtx.fillStyle = "#12141a";
    mapCtx.fillRect(0, 0, cssWidth, cssHeight);
    mapCtx.fillStyle = "#6b7183";
    mapCtx.font = "14px system-ui";
    mapCtx.fillText(
      entry.failed ? `map for ${world}-${stage} unavailable` : "loading map...",
      12, cssHeight / 2,
    );
  }
  if (!lastSwarm) return;

  const { xs, ys, ws, ss, alive, bestIdx } = lastSwarm;
  for (let i = 0; i < xs.length; i++) {
    if (i === bestIdx || ws[i] !== world || ss[i] !== stage) continue;
    const x = xs[i] * scale;
    const y = marioMapY(ys[i], imgH) * scale;
    mapCtx.beginPath();
    mapCtx.arc(x, y, 3, 0, 2 * Math.PI);
    // Magenta reads on both the sky and the ground tiles of every level.
    mapCtx.fillStyle = alive[i] ? "rgba(224, 64, 251, 0.75)" : "rgba(139, 148, 158, 0.5)";
    mapCtx.fill();
    mapCtx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    mapCtx.lineWidth = 0.75;
    mapCtx.stroke();
  }
  // Best walker: gold-ringed dot + full-height guide line, drawn last.
  if (ws[bestIdx] === world && ss[bestIdx] === stage) {
    const x = xs[bestIdx] * scale;
    const y = marioMapY(ys[bestIdx], imgH) * scale;
    mapCtx.strokeStyle = "rgba(255, 210, 31, 0.35)";
    mapCtx.lineWidth = 1.5;
    mapCtx.beginPath();
    mapCtx.moveTo(x, 0);
    mapCtx.lineTo(x, cssHeight);
    mapCtx.stroke();
    mapCtx.beginPath();
    mapCtx.arc(x, y, 4, 0, 2 * Math.PI);
    mapCtx.fillStyle = "#ff5252";
    mapCtx.fill();
    mapCtx.strokeStyle = "#ffd21f";
    mapCtx.lineWidth = 2;
    mapCtx.stroke();

    // Auto-follow: keep the marker centered when the map overflows, unless
    // the user scrolled by hand within the grace period.
    if (cssWidth > panelWidth &&
        performance.now() - mapUserScrolledAt > MAP_FOLLOW_GRACE_MS) {
      const target = Math.round(
        Math.min(Math.max(x - panelWidth / 2, 0), cssWidth - panelWidth));
      mapExpectedScroll = target;
      mapResize.scrollLeft = target;
    }
  }
}

// Sonic branch of drawMap: draw the fog canvas of the displayed level
// (best walker's zone/act, or the selected start level pre-run) scaled to
// the panel, with the same walker-dot overlay at 1/8 level scale.
function drawSonicMap() {
  let zone, act;
  if (lastSwarm) {
    zone = lastSwarm.world;
    act = lastSwarm.level;
  } else {
    const lp = levelParams();
    zone = lp.world;
    act = lp.stage;
  }
  const entry = getFogCanvas(zone, act, 0, 0);

  // Crop to the explored bounding box (small margin) so the reveal fills
  // the whole viewport instead of leaving black around a compressed strip.
  const MARGIN = 6;
  let srcX = 0, srcY = 0,
      srcW = entry.canvas.width, srcH = entry.canvas.height;
  if (entry.maxX > entry.minX) {
    srcX = Math.max(0, entry.minX - MARGIN);
    srcY = Math.max(0, entry.minY - MARGIN);
    srcW = Math.min(entry.canvas.width, entry.maxX + MARGIN) - srcX;
    srcH = Math.min(entry.canvas.height, entry.maxY + MARGIN) - srcY;
  }

  const vpW = mapResize.clientWidth || 900;
  const dpr = window.devicePixelRatio || 1;
  // Default viewport height per level; a manual drag of the resize handle
  // sticks afterwards.
  if (mapSizedForKey !== `sonic-${zone}-${act}`) {
    mapSizedForKey = `sonic-${zone}-${act}`;
    mapResize.style.height = "280px";
  }
  const vpH = mapResize.clientHeight || 280;
  // Uniform aspect-true scale: fit the explored crop inside the viewport
  // (letterboxed on the dark fog background), capped so a barely-explored
  // map isn't blown up into mush, then multiplied by the +/- zoom. When
  // zoomed past the viewport the wrapper scrolls both ways.
  const fitScale = Math.min(vpW / srcW, vpH / srcH, SONIC_FIT_MAX_SCALE);
  const scale = fitScale * sonicMapZoom;
  const cssWidth = Math.max(1, Math.round(srcW * scale));
  const cssHeight = Math.max(1, Math.round(srcH * scale));
  mapCanvas.width = Math.round(cssWidth * dpr);
  mapCanvas.height = Math.round(cssHeight * dpr);
  mapCanvas.style.width = cssWidth + "px";
  mapCanvas.style.height = cssHeight + "px";
  mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // The fog tiles are 8x-downsampled frames: smooth interpolation reads far
  // better than nearest-neighbour blocks when scaled up.
  mapCtx.imageSmoothingEnabled = true;
  mapCtx.imageSmoothingQuality = "high";
  mapCtx.fillStyle = "#0a0b0f";
  mapCtx.fillRect(0, 0, cssWidth, cssHeight);
  mapCtx.drawImage(entry.canvas, srcX, srcY, srcW, srcH,
                   0, 0, cssWidth, cssHeight);
  if (!lastSwarm) return;

  const { xs, ys, ws, ss, alive, bestIdx } = lastSwarm;
  for (let i = 0; i < xs.length; i++) {
    if (i === bestIdx || ws[i] !== zone || ss[i] !== act) continue;
    const x = (xs[i] / FOG_SCALE - srcX) * scale;
    const y = (ys[i] / FOG_SCALE - srcY) * scale;
    mapCtx.beginPath();
    mapCtx.arc(x, y, 3, 0, 2 * Math.PI);
    mapCtx.fillStyle = alive[i] ? "rgba(224, 64, 251, 0.75)" : "rgba(139, 148, 158, 0.5)";
    mapCtx.fill();
    mapCtx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    mapCtx.lineWidth = 0.75;
    mapCtx.stroke();
  }
  if (ws[bestIdx] === zone && ss[bestIdx] === act) {
    const x = (xs[bestIdx] / FOG_SCALE - srcX) * scale;
    const y = (ys[bestIdx] / FOG_SCALE - srcY) * scale;
    mapCtx.strokeStyle = "rgba(255, 210, 31, 0.35)";
    mapCtx.lineWidth = 1.5;
    mapCtx.beginPath();
    mapCtx.moveTo(x, 0);
    mapCtx.lineTo(x, cssHeight);
    mapCtx.stroke();
    mapCtx.beginPath();
    mapCtx.arc(x, y, 4, 0, 2 * Math.PI);
    mapCtx.fillStyle = "#ff5252";
    mapCtx.fill();
    mapCtx.strokeStyle = "#ffd21f";
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
  }
}

// +/- zoom buttons (Sonic fog map): multiply the fit scale, keeping the
// point at the middle of the viewport in place across the zoom change.
function setSonicZoom(zoom) {
  zoom = Math.min(SONIC_ZOOM_MAX, Math.max(1, zoom));
  const ratio = zoom / sonicMapZoom;
  const cx = mapResize.scrollLeft + mapResize.clientWidth / 2;
  const cy = mapResize.scrollTop + mapResize.clientHeight / 2;
  sonicMapZoom = zoom;
  drawMap();
  mapResize.scrollLeft = cx * ratio - mapResize.clientWidth / 2;
  mapResize.scrollTop = cy * ratio - mapResize.clientHeight / 2;
}
$("map-zoom-in").addEventListener("click",
  () => setSonicZoom(sonicMapZoom * SONIC_ZOOM_STEP));
$("map-zoom-out").addEventListener("click",
  () => setSonicZoom(sonicMapZoom / SONIC_ZOOM_STEP));
$("map-zoom-fit").addEventListener("click", () => {
  sonicMapZoom = 1;
  drawMap();
});

window.addEventListener("resize", drawMap);
// Redraw live while the user drags the map's resize handle.
new ResizeObserver(() => drawMap()).observe(mapResize);

let worker = null;
let romBuffer = null;
let auxBuffer = null; // Genesis savestate
let initialized = false;
let running = false;
// Per-console observation defaults: the fast Coords tuples for Mario and
// Sonic; Atari has no coords tuple (its mode 3 aliases the 128-byte RAM),
// so it defaults to RAM.
const OBS_DEFAULTS = { 0: 3, 1: 0, 2: 3 };
let obsMode = OBS_DEFAULTS[0]; // 0=RAM, 1=RGB, 2=Gray, 3=Coords
let consoleId = 0; // 0=NES Mario, 1=Atari, 2=Genesis
const genesisGame = 1; // Genesis game: Sonic
let atariGame = "ms_pacman"; // ALE rom id, served from web/roms/atari/
let sonicRomBuffer = null; // user-supplied

let ipsWindow = [];

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}


// world/stage in the params double as the start level: Mario's 1-based
// world/stage, or Sonic's internal zone id / 0-based act (Scrap Brain act 3
// is internally Labyrinth act 4).
function levelParams() {
  if (consoleId === 2) {
    let zone = parseInt($("param-zone").value, 10) || 0;
    let act = (parseInt($("param-act").value, 10) || 1) - 1;
    if (zone === 5 && act === 2) { zone = 1; act = 3; }
    return { world: zone, stage: act };
  }
  return {
    world: parseInt($("param-world").value, 10) || 1,
    stage: parseInt($("param-stage").value, 10) || 1,
  };
}

function readParams() {
  return {
    n: parseInt($("param-n").value, 10) || 48,
    distCoef: parseFloat($("param-dist-coef").value),
    rewardCoef: parseFloat($("param-reward-coef").value),
    // Always accumulate: the shaped one-time bonuses (Mario's flag, Sonic's
    // act completion) only work as a persistent fitness signal.
    useCumulativeReward: true,
    dtMin: parseInt($("param-dt-min").value, 10) || 6,
    dtMax: parseInt($("param-dt-max").value, 10) || 30,
    nElite: parseInt($("param-elite").value, 10) || 0,
    seed: parseInt($("param-seed").value, 10) || 0,
    nThreads: Math.min(Math.max((navigator.hardwareConcurrency || 4) - 1, 1), 8),
    obsMode,
    ...levelParams(),
    console: consoleId,
    game: genesisGame,
  };
}

function updateButtons() {
  $("btn-start").disabled = !initialized || running;
  $("btn-pause").disabled = !initialized || !running;
  $("btn-reset").disabled = !initialized;
}

function clearPlots() {
  for (const p of Object.values(plots)) p.clear();
  ipsWindow = [];
}

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (event) => {
    const msg = event.data;
    switch (msg.type) {
      case "ready":
        initialized = true;
        running = false;
        clearPlots();
        clearFog();
        lastSwarm = null;
        mapSizedForKey = null;
        drawMap();
        $("run-ended").hidden = true;
        setStatus("Ready - press Start", "ok");
        updateButtons();
        break;
      case "resetDone":
        running = false;
        clearPlots();
        lastSwarm = null;
        drawMap();
        screenCtx.clearRect(0, 0, screenCanvas.width, screenCanvas.height);
        $("run-ended").hidden = true;
        setStatus("Reset - press Start", "ok");
        updateButtons();
        break;
      case "step":
        onStep(msg);
        break;
      case "allDead": {
        running = false;
        updateButtons();
        const detail =
          `All walkers dead at iteration ${msg.iteration}. ` +
          "Press Reset to start over (a higher distance coef keeps the " +
          "swarm more diverse and harder to wipe out).";
        $("run-ended-detail").textContent = detail;
        $("run-ended").hidden = false;
        setStatus("Run stopped - all walkers dead", "error");
        break;
      }
      case "error":
        running = false;
        initialized = false;
        setStatus("Error: " + msg.message, "error");
        updateButtons();
        break;
    }
  };
  worker.onerror = (err) => {
    setStatus("Worker error: " + err.message, "error");
  };
  return worker;
}

function onStep(msg) {
  const s = msg.stats;
  if (!s) return;

  if (msg.frame && msg.frameWidth > 0) {
    if (screenCanvas.width !== msg.frameWidth ||
        screenCanvas.height !== msg.frameHeight) {
      screenCanvas.width = msg.frameWidth;
      screenCanvas.height = msg.frameHeight;
    }
    const image = new ImageData(
      new Uint8ClampedArray(msg.frame),
      msg.frameWidth,
      msg.frameHeight,
    );
    screenCtx.putImageData(image, 0, 0);
  }

  if (s.walkerX) {
    lastSwarm = {
      xs: s.walkerX,
      ys: s.walkerY,
      ws: s.walkerWorld,
      ss: s.walkerStage,
      camXs: s.walkerCamX,
      camYs: s.walkerCamY,
      alive: s.walkerAlive,
      bestIdx: s.bestWalkerIdx,
      world: s.world,
      level: s.level,
    };
    if (consoleId === 2 && msg.walkerTiles) {
      applyFogTiles(lastSwarm, msg.walkerTiles);
    }
    drawMap();
  }

  const n = parseInt($("param-n").value, 10) || 48;
  plots.reward.append([s.maxReward, s.meanReward]);
  plots.vr.append([s.maxVirtualReward, s.meanVirtualReward]);
  plots.clones.append([(100 * s.numCloned) / n]);
  plots.alive.append([s.aliveCount]);
  plots.dt.append([s.meanDt]);

  $("stat-iteration").textContent = s.iteration;
  if (s.world !== undefined) {
    if (consoleId === 2) {
      const zoneNames = ["GHZ", "LZ", "MZ", "SLZ", "SYZ", "SBZ"];
      $("stat-world").textContent =
        `${zoneNames[s.world] ?? s.world} ${s.level + 1}`;
    } else {
      $("stat-world").textContent = `${s.world}-${s.level}`;
    }
  }
  $("stat-max-reward").textContent = s.maxReward.toFixed(1);
  $("stat-mean-reward").textContent = s.meanReward.toFixed(1);
  $("stat-alive").textContent = `${s.aliveCount} / ${n}`;
  $("stat-frames").textContent = Math.round(s.totalSteps * s.meanDt).toLocaleString();

  const now = performance.now();
  ipsWindow.push(now);
  while (ipsWindow.length > 0 && now - ipsWindow[0] > 3000) ipsWindow.shift();
  if (ipsWindow.length > 1) {
    const ips = ((ipsWindow.length - 1) / (now - ipsWindow[0])) * 1000;
    $("stat-ips").textContent = ips.toFixed(1);
  }
}

function initRun() {
  if (!romBuffer) return;
  if (!crossOriginIsolated) {
    setStatus(
      "Not cross-origin isolated - serve with serve.py (COOP/COEP headers " +
        "are required for threads)",
      "error",
    );
    return;
  }
  setStatus("Loading emulator + swarm...");
  initialized = false;
  running = false;
  updateButtons();
  // Copy so the source buffers survive repeated inits (transfer detaches).
  const rom = romBuffer.slice(0);
  const aux = auxBuffer ? auxBuffer.slice(0) : new ArrayBuffer(0);
  ensureWorker().postMessage(
    { type: "init", rom, aux, params: readParams() },
    [rom, aux],
  );
}

// Per-console assets: NES and Genesis ship bundled files; Atari needs a
// user-supplied ROM.
async function loadConsoleAssets() {
  romBuffer = null;
  auxBuffer = null;
  $("level-section").hidden = consoleId !== 0;
  $("atari-section").hidden = consoleId !== 1;
  $("sonic-section").hidden = consoleId !== 2;
  $("map-panel").hidden = consoleId === 1;  // maps for Mario + Sonic
  $("map-zoom").hidden = consoleId !== 2;   // +/- zoom is fog-map only
  sonicMapZoom = 1;
  document.querySelector(".map-credit").innerHTML = consoleId === 2
    ? "Fog of war: the map is revealed by the swarm as it explores. " +
      "Magenta dots: alive walkers &middot; grey: dead &middot; gold ring: best walker."
    : 'Magenta dots: alive walkers &middot; grey: dead &middot; gold ring: ' +
      'best walker. Level maps from <a href="https://ian-albert.com/games/' +
      'super_mario_bros_maps/" target="_blank" rel="noreferrer">ian-albert.com</a>.';
  lastSwarm = null;
  if (consoleId === 0) drawMap();
  try {
    if (consoleId === 0) {
      const resp = await fetch("test-rom.nes");
      if (!resp.ok) throw new Error("test-rom.nes not found");
      romBuffer = await resp.arrayBuffer();
    } else if (consoleId === 2) {
      // Sonic: bundled ROM (web/sonic.rom, gitignored) or user-supplied.
      // The core ships as web/retro_shim.{js,wasm}; each core worker
      // fetches it itself, and the game boots from power-on into the
      // selected zone/act.
      if (sonicRomBuffer) {
        romBuffer = sonicRomBuffer.slice(0);
      } else {
        const rom = await fetch("sonic.rom");
        if (!rom.ok) {
          $("sonic-rom-row").hidden = false;
          setStatus("Pick a Sonic The Hedgehog (Genesis) ROM to start");
          return;
        }
        romBuffer = await rom.arrayBuffer();
      }
    } else {
      const resp = await fetch(`roms/atari/${atariGame}.bin`);
      if (!resp.ok) throw new Error(`roms/atari/${atariGame}.bin not found`);
      romBuffer = await resp.arrayBuffer();
    }
    initRun();
  } catch (err) {
    setStatus(String(err), "error");
  }
}

// Start-level selectors: structural change -> restart the run.
for (const id of ["param-world", "param-stage"]) {
  $(id).addEventListener("change", () => {
    lastSwarm = null;
    drawMap(); // show the new level's map right away
    if (romBuffer) initRun();
  });
}

$("btn-start").addEventListener("click", () => {
  running = true;
  $("run-ended").hidden = true;
  updateButtons();
  setStatus("Running", "ok");
  worker.postMessage({ type: "start" });
});

$("btn-pause").addEventListener("click", () => {
  running = false;
  updateButtons();
  setStatus("Paused");
  worker.postMessage({ type: "pause" });
});

$("btn-reset").addEventListener("click", () => {
  worker.postMessage({ type: "reset" });
});

// Live-tunable parameters -> setParams; structural ones -> re-init.
for (const id of ["param-dist-coef", "param-reward-coef"]) {
  $(id).addEventListener("input", () => {
    $("dist-coef-value").textContent = parseFloat($("param-dist-coef").value).toFixed(2);
    $("reward-coef-value").textContent = parseFloat($("param-reward-coef").value).toFixed(2);
    if (initialized) worker.postMessage({ type: "setParams", params: readParams() });
  });
}
for (const id of ["param-dt-min", "param-dt-max", "param-elite"]) {
  $(id).addEventListener("change", () => {
    if (initialized) worker.postMessage({ type: "setParams", params: readParams() });
  });
}
for (const id of ["param-n", "param-seed"]) {
  $(id).addEventListener("change", () => {
    if (romBuffer) initRun();
  });
}

// Observation mode toggles: structural change -> restart the run.
function setObsMode(mode) {
  obsMode = mode;
  for (const b of $("obs-mode").querySelectorAll("button")) {
    b.classList.toggle("active", parseInt(b.dataset.mode, 10) === mode);
  }
}
for (const btn of $("obs-mode").querySelectorAll("button")) {
  btn.addEventListener("click", () => {
    const mode = parseInt(btn.dataset.mode, 10);
    if (mode === obsMode) return;
    setObsMode(mode);
    if (romBuffer) initRun();
  });
}

updateButtons();


// --- Panel UX: collapsible sidebar, resizable/minimizable plots panel -------
const sidebarEl = $("sidebar");
const plotsPanel = $("plots-panel");
const plotsResizer = $("plots-resizer");

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

function resizePlots() {
  if (plotsPanel.classList.contains("collapsed")) return;
  const w = Math.max(180, $("plots-body").clientWidth - 4);
  for (const plot of Object.values(plots)) {
    if (plot.canvas.width !== w) plot.canvas.width = w;
    plot.render();
  }
}

function setSidebarCollapsed(collapsed) {
  sidebarEl.classList.toggle("collapsed", collapsed);
  $("sidebar-toggle").textContent = collapsed ? "\u00bb" : "\u00ab";
  $("sidebar-toggle").title = collapsed ? "Expand controls" : "Collapse controls";
  store.set("fgSidebarCollapsed", collapsed ? "1" : "0");
}

function setPlotsCollapsed(collapsed) {
  plotsPanel.classList.toggle("collapsed", collapsed);
  $("plots-toggle").textContent = collapsed ? "\u00ab" : "\u00bb";
  $("plots-toggle").title = collapsed ? "Expand plots" : "Collapse plots";
  store.set("fgPlotsCollapsed", collapsed ? "1" : "0");
  if (!collapsed) requestAnimationFrame(resizePlots);
}

$("sidebar-toggle").addEventListener("click", () =>
  setSidebarCollapsed(!sidebarEl.classList.contains("collapsed")));
$("plots-toggle").addEventListener("click", () =>
  setPlotsCollapsed(!plotsPanel.classList.contains("collapsed")));

plotsResizer.addEventListener("pointerdown", (down) => {
  down.preventDefault();
  plotsResizer.setPointerCapture(down.pointerId);
  plotsResizer.classList.add("dragging");
  const startWidth = plotsPanel.getBoundingClientRect().width;
  const startX = down.clientX;
  const onMove = (move) => {
    const w = Math.min(900, Math.max(220, startWidth + (startX - move.clientX)));
    plotsPanel.style.width = w + "px";
    resizePlots();
  };
  const onUp = () => {
    plotsResizer.classList.remove("dragging");
    plotsResizer.removeEventListener("pointermove", onMove);
    plotsResizer.removeEventListener("pointerup", onUp);
    store.set("fgPlotsWidth", String(Math.round(
      plotsPanel.getBoundingClientRect().width)));
  };
  plotsResizer.addEventListener("pointermove", onMove);
  plotsResizer.addEventListener("pointerup", onUp);
});

let resizeTimer = 0;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(resizePlots, 100);
});

// Restore persisted panel state.
{
  const w = parseInt(store.get("fgPlotsWidth") || "", 10);
  if (w >= 220 && w <= 900) plotsPanel.style.width = w + "px";
  if (store.get("fgSidebarCollapsed") === "1") setSidebarCollapsed(true);
  if (store.get("fgPlotsCollapsed") === "1") setPlotsCollapsed(true);
  requestAnimationFrame(resizePlots);
}

// Console selector: structural change -> reload assets and restart.
for (const btn of $("console-select").querySelectorAll("button")) {
  btn.addEventListener("click", () => {
    const id = parseInt(btn.dataset.console, 10);
    if (id === consoleId) return;
    consoleId = id;
    for (const b of $("console-select").querySelectorAll("button")) {
      b.classList.toggle("active", b === btn);
    }
    setObsMode(OBS_DEFAULTS[id]);
    loadConsoleAssets();
  });
}

// Atari game picker: every ALE-supported rom bundled under web/roms/atari/.
const ATARI_GAMES = [
  "adventure", "air_raid", "alien", "amidar", "assault", "asterix",
  "asteroids", "atlantis", "atlantis2", "backgammon", "bank_heist",
  "basic_math", "battle_zone", "beam_rider", "berzerk", "blackjack",
  "bowling", "boxing", "breakout", "carnival", "casino", "centipede",
  "chopper_command", "combat", "crazy_climber", "crossbow", "darkchambers",
  "defender", "demon_attack", "donkey_kong", "double_dunk", "earthworld",
  "elevator_action", "enduro", "entombed", "et", "fishing_derby",
  "flag_capture", "freeway", "frogger", "frostbite", "galaxian", "gopher",
  "gravitar", "hangman", "haunted_house", "hero", "human_cannonball",
  "ice_hockey", "jamesbond", "journey_escape", "joust", "kaboom",
  "kangaroo", "keystone_kapers", "king_kong", "klax", "koolaid", "krull",
  "kung_fu_master", "laser_gates", "lost_luggage", "mario_bros",
  "maze_craze", "miniature_golf", "montezuma_revenge", "mr_do", "ms_pacman",
  "name_this_game", "othello", "pacman", "phoenix", "pitfall", "pitfall2",
  "pong", "pooyan", "private_eye", "qbert", "riverraid", "road_runner",
  "robotank", "seaquest", "sir_lancelot", "skiing", "solaris",
  "space_invaders", "space_war", "star_gunner", "superman", "surround",
  "tennis", "tetris", "tic_tac_toe_3d", "time_pilot", "trondead",
  "turmoil", "tutankham", "up_n_down", "venture", "video_checkers",
  "video_chess", "video_cube", "video_pinball", "warlords", "wizard_of_wor",
  "word_zapper", "yars_revenge", "zaxxon",
];

const atariSelect = $("atari-game");
for (const id of ATARI_GAMES) {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = id.split("_").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
  opt.selected = id === atariGame;
  atariSelect.appendChild(opt);
}
atariSelect.addEventListener("change", () => {
  atariGame = atariSelect.value;
  if (consoleId === 1) loadConsoleAssets();
});

$("genesis-rom-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  sonicRomBuffer = await file.arrayBuffer();
  if (consoleId === 2) loadConsoleAssets();
});

// Sonic start-level selectors: structural change -> restart the run.
for (const id of ["param-zone", "param-act"]) {
  $(id).addEventListener("change", () => {
    if (consoleId === 2 && romBuffer) initRun();
  });
}

// Startup: load the default console's bundled assets.
loadConsoleAssets();

// Headless test hook: #autorun-sonic selects the Sonic console and starts
// the run as soon as it initializes (used by the screenshot autotest).
if (location.hash === "#autorun-sonic") {
  document.querySelector('#console-select [data-console="2"]').click();
  const auto = setInterval(() => {
    const btn = $("btn-start");
    if (!btn.disabled) {
      clearInterval(auto);
      btn.click();
      // After the swarm has explored a while, upload the fog map canvas
      // for the headless visual check.
      let shots = 0;
      const snap = setInterval(() => {
        shots++;
        mapCanvas.toBlob((blob) => {
          if (blob) fetch(`/debug-upload/sonic-fog-${shots}.png`,
                          { method: "POST", body: blob }).catch(() => {});
        });
        if (shots >= 3) clearInterval(snap);
      }, 30000);
    }
  }, 500);
}
