// Main-thread UI: sidebar controls, canvas rendering of the best walker,
// stat readouts and the five plots (mirroring the Panel dashboard panes).

import { LinePlot } from "./plots.js";
import { initHelp } from "./help.js";
import { buildRewardPanel, readRewardWeights, showRewardPanel } from "./reward_terms.js";

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
  tree: new LinePlot($("plot-tree"), {
    title: "Graph size",
    series: [
      { name: "nodes", color: "#79c0ff" },
      { name: "leaves", color: "#7ee787" },
      { name: "stepped", color: "#ffa657" },
    ],
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
let mapZoom = 1;
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
// `tiles` describes the tiles of the LAST env batch (in Graph mode only the
// walkers that stepped), each with its own camera/level, so they never need
// to line up with the per-walker arrays.
function applyFogTiles(tilesInfo, tilesBuf) {
  const tiles = new Uint8Array(tilesBuf);
  const rgba = fogTileImage.data;
  const count = Math.min(tilesInfo.count, Math.floor(tiles.length / (TILE_W * TILE_H * 3)));
  for (let i = 0; i < count; i++) {
    const camX = Math.floor(tilesInfo.camXs[i] / FOG_SCALE);
    const camY = Math.floor(tilesInfo.camYs[i] / FOG_SCALE);
    const entry = getFogCanvas(tilesInfo.ws[i], tilesInfo.ss[i],
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

// ---------------------------------------------------------------------------
// Swarm overlay shared by the three maps. `project(i)` maps walker i to css
// pixels on the current map, or returns null when it is not on the displayed
// level. In Graph mode every node is joined to its parent by a thin line
// (the graph of visited states) and interior nodes are drawn smaller than
// the leaves; the wave has parent = self and every walker is a "leaf".
// ---------------------------------------------------------------------------
function drawSwarmOverlay(project, cssHeight, guideLine) {
  const { xs, parents, leaf, alive, bestIdx } = lastSwarm;
  const n = xs.length;
  const pts = new Array(n);
  for (let i = 0; i < n; i++) pts[i] = project(i);
  if (parents) {
    mapCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = parents[i];
      if (p === i || !pts[i] || !pts[p]) continue;
      mapCtx.moveTo(pts[p].x, pts[p].y);
      mapCtx.lineTo(pts[i].x, pts[i].y);
    }
    mapCtx.strokeStyle = "rgba(255, 255, 255, 0.22)";
    mapCtx.lineWidth = 0.75;
    mapCtx.stroke();
  }
  for (let i = 0; i < n; i++) {
    if (i === bestIdx || !pts[i]) continue;
    const isLeaf = !leaf || leaf[i];
    if (!isLeaf) {
      // Interior node of the graph: a small faint square.
      mapCtx.fillStyle = alive[i] ? "rgba(224, 64, 251, 0.45)" : "rgba(139, 148, 158, 0.35)";
      mapCtx.fillRect(pts[i].x - 1, pts[i].y - 1, 2, 2);
      continue;
    }
    mapCtx.beginPath();
    mapCtx.arc(pts[i].x, pts[i].y, 3, 0, 2 * Math.PI);
    // Magenta reads on both the sky and the ground tiles of every level.
    mapCtx.fillStyle = alive[i] ? "rgba(224, 64, 251, 0.75)" : "rgba(139, 148, 158, 0.5)";
    mapCtx.fill();
    mapCtx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    mapCtx.lineWidth = 0.75;
    mapCtx.stroke();
  }
  // Best walker: gold-ringed dot (+ optional full-height guide line), last.
  const b = pts[bestIdx];
  if (!b) return null;
  if (guideLine) {
    mapCtx.strokeStyle = "rgba(255, 210, 31, 0.35)";
    mapCtx.lineWidth = 1.5;
    mapCtx.beginPath();
    mapCtx.moveTo(b.x, 0);
    mapCtx.lineTo(b.x, cssHeight);
    mapCtx.stroke();
  }
  mapCtx.beginPath();
  mapCtx.arc(b.x, b.y, 4, 0, 2 * Math.PI);
  mapCtx.fillStyle = "#ff5252";
  mapCtx.fill();
  mapCtx.strokeStyle = "#ffd21f";
  mapCtx.lineWidth = 2;
  mapCtx.stroke();
  return b;
}

function drawMap() {
  if (consoleId === 1) return;
  if (consoleId === 2) return drawSonicMap();
  if (consoleId === 3) return drawPyramidMap();
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

  const { xs, ys, ws, ss } = lastSwarm;
  const best = drawSwarmOverlay((i) => {
    if (ws[i] !== world || ss[i] !== stage) return null;
    return { x: xs[i] * scale, y: marioMapY(ys[i], imgH) * scale };
  }, cssHeight, true);
  if (best) {
    // Auto-follow: keep the marker centered when the map overflows, unless
    // the user scrolled by hand within the grace period.
    if (cssWidth > panelWidth &&
        performance.now() - mapUserScrolledAt > MAP_FOLLOW_GRACE_MS) {
      const target = Math.round(
        Math.min(Math.max(best.x - panelWidth / 2, 0), cssWidth - panelWidth));
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
  const scale = fitScale * mapZoom;
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

  const { xs, ys, ws, ss } = lastSwarm;
  drawSwarmOverlay((i) => {
    if (ws[i] !== zone || ss[i] !== act) return null;
    return { x: (xs[i] / FOG_SCALE - srcX) * scale,
             y: (ys[i] / FOG_SCALE - srcY) * scale };
  }, cssHeight, true);
}

// ---------------------------------------------------------------------------
// Montezuma pyramid map: the level-1 temple is 24 rooms on a 9x4 grid (same
// PYRAMID table as src/montezuma_logic.hpp). Like the Sonic fog map, it is
// BUILT by the swarm: the worker ships each room's 160x160 image (HUD
// cropped) the first time a walker stands in it, and every walker's in-room
// position is overlaid on the room's cell. Unvisited rooms stay dark.
// ---------------------------------------------------------------------------
const PYRAMID = [
  [-1, -1, -1, 0, 1, 2, -1, -1, -1],
  [-1, -1, 3, 4, 5, 6, 7, -1, -1],
  [-1, 8, 9, 10, 11, 12, 13, 14, -1],
  [15, 16, 17, 18, 19, 20, 21, 22, 23],
];
const ROOM = 160;
const PYRAMID_W = PYRAMID[0].length * ROOM;  // 1440
const PYRAMID_H = PYRAMID.length * ROOM;     // 640
const ROOM_CELL = new Map(); // room -> {col, row}
for (let r = 0; r < PYRAMID.length; r++) {
  for (let c = 0; c < PYRAMID[r].length; c++) {
    if (PYRAMID[r][c] >= 0) ROOM_CELL.set(PYRAMID[r][c], { col: c, row: r });
  }
}
const pyramidLevels = new Map(); // level -> {canvas, ctx, rooms: Set}

function getPyramidCanvas(level) {
  let entry = pyramidLevels.get(level);
  if (entry) return entry;
  const canvas = document.createElement("canvas");
  canvas.width = PYRAMID_W;
  canvas.height = PYRAMID_H;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0a0b0f";
  ctx.fillRect(0, 0, PYRAMID_W, PYRAMID_H);
  // Unexplored rooms: a faint outline and the room number, so the shape of
  // the pyramid is visible before the swarm reveals it.
  for (const [room, { col, row }] of ROOM_CELL) {
    ctx.fillStyle = "#12141a";
    ctx.fillRect(col * ROOM + 1, row * ROOM + 1, ROOM - 2, ROOM - 2);
    ctx.strokeStyle = "#2c3040";
    ctx.lineWidth = 1;
    ctx.strokeRect(col * ROOM + 0.5, row * ROOM + 0.5, ROOM - 1, ROOM - 1);
    ctx.fillStyle = "#3a3f52";
    ctx.font = "bold 28px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(room), col * ROOM + ROOM / 2, row * ROOM + ROOM / 2);
  }
  entry = { canvas, ctx, rooms: new Set() };
  pyramidLevels.set(level, entry);
  return entry;
}

function applyRoomFrames(frames) {
  for (const f of frames) {
    const cell = ROOM_CELL.get(f.room);
    if (!cell) continue;  // rooms beyond the level-1 pyramid are not drawn
    const entry = getPyramidCanvas(f.level);
    const image = new ImageData(new Uint8ClampedArray(f.rgba), f.width, f.height);
    entry.ctx.putImageData(image, cell.col * ROOM, cell.row * ROOM);
    entry.rooms.add(f.room);
  }
}

function clearPyramid() {
  pyramidLevels.clear();
}

// Montezuma branch of drawMap: the displayed level's pyramid (best walker's
// level, or level 0 pre-run) aspect-fit into the viewport times the +/-
// zoom, with the walker-dot overlay on each room's cell.
function drawPyramidMap() {
  const level = lastSwarm ? lastSwarm.level : 0;
  const entry = getPyramidCanvas(level);
  const vpW = mapResize.clientWidth || 900;
  const dpr = window.devicePixelRatio || 1;
  // Default viewport height: the full pyramid at panel width.
  if (mapSizedForKey !== `pyramid-${level}`) {
    mapSizedForKey = `pyramid-${level}`;
    mapResize.style.height = Math.round(vpW * PYRAMID_H / PYRAMID_W) + "px";
  }
  const vpH = mapResize.clientHeight || 280;
  const fitScale = Math.min(vpW / PYRAMID_W, vpH / PYRAMID_H);
  const scale = fitScale * mapZoom;
  const cssWidth = Math.max(1, Math.round(PYRAMID_W * scale));
  const cssHeight = Math.max(1, Math.round(PYRAMID_H * scale));
  mapCanvas.width = Math.round(cssWidth * dpr);
  mapCanvas.height = Math.round(cssHeight * dpr);
  mapCanvas.style.width = cssWidth + "px";
  mapCanvas.style.height = cssHeight + "px";
  mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // Pixel art: keep the room images crisp when scaled.
  mapCtx.imageSmoothingEnabled = scale < 1;
  mapCtx.fillStyle = "#0a0b0f";
  mapCtx.fillRect(0, 0, cssWidth, cssHeight);
  mapCtx.drawImage(entry.canvas, 0, 0, cssWidth, cssHeight);
  if (!lastSwarm) return;

  const { xs, ys, ws, ss, bestIdx } = lastSwarm;
  const project = (i) => {
    if (ss[i] !== level) return null;
    const cell = ROOM_CELL.get(ws[i]);
    if (!cell) return null;
    return { x: (cell.col * ROOM + xs[i]) * scale,
             y: (cell.row * ROOM + ys[i]) * scale };
  };
  // Highlight the best walker's room, then the overlay on top.
  const bestCell = ss[bestIdx] === level ? ROOM_CELL.get(ws[bestIdx]) : null;
  if (bestCell) {
    mapCtx.strokeStyle = "rgba(255, 210, 31, 0.5)";
    mapCtx.lineWidth = 1.5;
    mapCtx.strokeRect(bestCell.col * ROOM * scale + 0.75, bestCell.row * ROOM * scale + 0.75,
                      ROOM * scale - 1.5, ROOM * scale - 1.5);
  }
  drawSwarmOverlay(project, cssHeight, false);
}

// +/- zoom buttons (Sonic fog map, Montezuma pyramid): multiply the fit
// scale, keeping the point at the middle of the viewport in place across
// the zoom change.
function setMapZoom(zoom) {
  zoom = Math.min(SONIC_ZOOM_MAX, Math.max(1, zoom));
  const ratio = zoom / mapZoom;
  const cx = mapResize.scrollLeft + mapResize.clientWidth / 2;
  const cy = mapResize.scrollTop + mapResize.clientHeight / 2;
  mapZoom = zoom;
  drawMap();
  mapResize.scrollLeft = cx * ratio - mapResize.clientWidth / 2;
  mapResize.scrollTop = cy * ratio - mapResize.clientHeight / 2;
}
$("map-zoom-in").addEventListener("click",
  () => setMapZoom(mapZoom * SONIC_ZOOM_STEP));
$("map-zoom-out").addEventListener("click",
  () => setMapZoom(mapZoom / SONIC_ZOOM_STEP));
$("map-zoom-fit").addEventListener("click", () => {
  mapZoom = 1;
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
// Per-console observation defaults: the fast Coords tuples for Mario, Sonic
// and Montezuma; generic Atari has no coords tuple (its mode 3 aliases the
// 128-byte RAM), so it defaults to RAM.
const OBS_DEFAULTS = { 0: 3, 1: 0, 2: 3, 3: 3 };
let obsMode = OBS_DEFAULTS[0]; // 0=RAM, 1=RGB, 2=Gray, 3=Coords
// UI console ids: 0=NES Mario, 1=Atari (any ALE game), 2=Genesis Sonic,
// 3=Montezuma's Revenge (C++ sees Atari console 1 with game 1).
let consoleId = 0;
const genesisGame = 1; // Genesis game: Sonic
// Swarm algorithm: 0 = Wave (FractalGas, fixed swarm), 1 = Graph
// (FractalTree, growing tree of states). Structural: restarts the run.
let algorithm = 0;
// Graph population cap defaults per UI console (every node keeps a full
// emulator state: Genesis blobs are MB-sized).
const MAX_WALKERS_DEFAULT = { 0: 4000, 1: 20000, 2: 150, 3: 20000 };
const MONTEZUMA_GAME = 1; // Atari game id with dedicated logic (wasm_bindings)
let atariGame = "ms_pacman"; // ALE rom id, served from web/roms/atari/

// ROM vault. The ROMs are copyrighted, so hosted deployments (GitHub Pages)
// ship them only as password-encrypted blobs under web/roms-enc/ (built by
// tools/encrypt-rom.mjs --all with ONE password for every game). Loading
// order per ROM: a plaintext copy served next to the page (local dev, all
// gitignored) -> this browser's IndexedDB cache (decrypted or uploaded on an
// earlier visit) -> the encrypted blob, decrypted with the password (kept
// in memory and in IndexedDB after the first unlock, so every other game
// then unlocks silently) -> for Sonic only, a one-time upload.
const ROM_DB = "fg-roms";
const ROM_STORE = "roms";
function romDbOpen() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(ROM_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(ROM_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function romDbGet(key) {
  try {
    const db = await romDbOpen();
    return await new Promise((resolve) => {
      const req = db.transaction(ROM_STORE).objectStore(ROM_STORE).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch (e) {
    return null;
  }
}
async function romDbPut(key, buf) {
  try {
    const db = await romDbOpen();
    db.transaction(ROM_STORE, "readwrite").objectStore(ROM_STORE)
      .put(buf, key);
  } catch (e) { /* private mode etc. — upload still works per-visit */ }
}
async function romDbDelete(key) {
  try {
    const db = await romDbOpen();
    db.transaction(ROM_STORE, "readwrite").objectStore(ROM_STORE).delete(key);
  } catch (e) { /* ignore */ }
}

// Password-encrypted ROM blobs. The password lives solely in the user's
// head and the ROM is decrypted here in the browser with WebCrypto. Blob
// layout, matching tools/encrypt-rom.mjs: salt(16) | iv(12) | AES-256-GCM
// ciphertext-with-tag. Key = PBKDF2-SHA256(password, salt, 250k). The vault
// shares one salt across every ROM, so the (slow) key derivation runs once
// per session: derived keys are cached by salt.
const ROM_PBKDF2_ITERS = 250000;
const romKeyCache = new Map(); // salt hex -> CryptoKey
async function deriveRomKey(password, salt) {
  const hex = Array.from(salt, (b) => b.toString(16).padStart(2, "0")).join("");
  const cacheKey = `${hex}:${password}`;
  let key = romKeyCache.get(cacheKey);
  if (key) return key;
  const keyMat = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(password), "PBKDF2", false,
    ["deriveKey"]);
  key = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: ROM_PBKDF2_ITERS, hash: "SHA-256" },
    keyMat, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  romKeyCache.set(cacheKey, key);
  return key;
}
async function decryptRom(encBuf, password) {
  const data = new Uint8Array(encBuf);
  const salt = data.slice(0, 16);
  const iv = data.slice(16, 28);
  const ct = data.slice(28);
  const key = await deriveRomKey(password, salt);
  // Throws on a wrong password (GCM tag mismatch).
  return await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
}

const ROM_ENC_DIR = "roms-enc";
const ROM_PASSWORD_KEY = "__password";
// id -> {plain, enc, name}. Atari games are "atari:<ale id>".
function romSpec(id) {
  if (id === "mario") {
    return { plain: "test-rom.nes", enc: `${ROM_ENC_DIR}/mario.nes.enc`,
             name: "Super Mario Bros" };
  }
  if (id === "sonic") {
    return { plain: "sonic.rom", enc: `${ROM_ENC_DIR}/sonic.rom.enc`,
             name: "Sonic The Hedgehog" };
  }
  if (id.startsWith("atari:")) {
    const game = id.slice("atari:".length);
    return { plain: `roms/atari/${game}.bin`,
             enc: `${ROM_ENC_DIR}/atari/${game}.bin.enc`,
             name: atariGameName(game) };
  }
  throw new Error(`unknown ROM ${id}`);
}
const romCache = new Map(); // id -> ArrayBuffer for this page load
let romPassword = null;     // in memory after the first successful unlock
let pendingRom = null;      // {id, enc} waiting for the password
async function fetchBuffer(path) {
  const resp = await fetch(path).catch(() => null);
  return resp && resp.ok ? await resp.arrayBuffer() : null;
}
// Resolves to {buf} or {buf: null, reason: "locked" | "missing"}. "locked"
// means an encrypted copy exists and the unlock prompt should be shown.
async function loadRom(id) {
  if (romCache.has(id)) return { buf: romCache.get(id) };
  const spec = romSpec(id);
  let buf = await fetchBuffer(spec.plain);
  if (!buf) buf = await romDbGet(id);
  if (!buf) {
    const enc = await fetchBuffer(spec.enc);
    if (!enc) return { buf: null, reason: "missing" };
    const pass = romPassword ?? await romDbGet(ROM_PASSWORD_KEY);
    if (typeof pass === "string" && pass) {
      try {
        buf = await decryptRom(enc, pass);
        romPassword = pass;
        romDbPut(id, buf);
      } catch (e) {
        // Stored password no longer matches the deployed vault: forget it.
        romPassword = null;
        romDbDelete(ROM_PASSWORD_KEY);
      }
    }
    if (!buf) {
      pendingRom = { id, enc };
      return { buf: null, reason: "locked" };
    }
  }
  romCache.set(id, buf);
  return { buf };
}

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
    // Montezuma is the Atari console with the dedicated game id.
    console: consoleId === 3 ? 1 : consoleId,
    game: consoleId === 3 ? MONTEZUMA_GAME : (consoleId === 2 ? genesisGame : 0),
    algorithm,
    maxWalkers: parseInt($("param-max-walkers").value, 10) || 0,
    eraseCoef: parseFloat($("param-erase-coef").value),
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
        $("stat-world").textContent = "\u2013";
        $("stat-tree").textContent = "\u2013";
        if (msg.algorithm === 1 && msg.maxWalkers) {
          const asked = parseInt($("param-max-walkers").value, 10) || 0;
          $("max-walkers-hint").textContent = msg.maxWalkers < asked
            ? `capped to ${msg.maxWalkers} by the browser's wasm memory`
            : `effective cap ${msg.maxWalkers}`;
        }
        clearPlots();
        clearFog();
        clearPyramid();
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
        clearPyramid();
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
      // Graph structure (Wave: parent = self, every walker a leaf).
      parents: s.algorithm === 1 ? s.walkerParent : null,
      leaf: s.algorithm === 1 ? s.walkerLeaf : null,
      alive: s.walkerAlive,
      bestIdx: s.bestWalkerIdx,
      world: s.world,
      level: s.level,
    };
    if (consoleId === 2 && msg.walkerTiles && s.tileCount) {
      applyFogTiles({ count: s.tileCount, camXs: s.tileCamX, camYs: s.tileCamY,
                      ws: s.tileZone, ss: s.tileAct }, msg.walkerTiles);
    }
    if (consoleId === 3 && msg.roomFrames && msg.roomFrames.length) {
      applyRoomFrames(msg.roomFrames);
    }
    drawMap();
  }

  // Live population: the Graph grows, so never trust the input box.
  const n = s.walkerCount || parseInt($("param-n").value, 10) || 48;
  plots.reward.append([s.maxReward, s.meanReward]);
  plots.vr.append([s.maxVirtualReward, s.meanVirtualReward]);
  plots.clones.append([(100 * s.numCloned) / n]);
  plots.alive.append([s.aliveCount]);
  plots.dt.append([s.meanDt]);
  if (s.algorithm === 1) {
    plots.tree.append([s.walkerCount, s.nLeaves, s.numStepped]);
    $("stat-tree").textContent = `${s.walkerCount} / ${s.nLeaves} / ${s.numStepped}`;
  }

  $("stat-iteration").textContent = s.iteration;
  if (s.world !== undefined) {
    if (consoleId === 2) {
      const zoneNames = ["GHZ", "LZ", "MZ", "SLZ", "SYZ", "SBZ"];
      $("stat-world").textContent =
        `${zoneNames[s.world] ?? s.world} ${s.level + 1}`;
    } else if (consoleId === 3) {
      $("stat-world").textContent =
        `Room ${s.world} \u00b7 L${s.level + 1} \u00b7 ${s.lives} lives`;
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
    { type: "init", rom, aux, params: readParams(),
      rewardWeights: readRewardWeights(consoleId) },
    [rom, aux],
  );
}

// Per-console assets: pick the console's ROM from the vault and start.
const CONSOLE_ROM = {
  0: () => "mario",
  1: () => `atari:${atariGame}`,
  2: () => "sonic",
  3: () => "atari:montezuma_revenge",
};
let loadToken = 0; // drops the result of a superseded load (fast switching)
async function loadConsoleAssets() {
  const token = ++loadToken;
  romBuffer = null;
  auxBuffer = null;
  $("level-section").hidden = consoleId !== 0;
  $("atari-section").hidden = consoleId !== 1;
  $("sonic-section").hidden = consoleId !== 2;
  $("sonic-rom-row").hidden = true;
  $("rom-unlock-section").hidden = true;
  $("map-panel").hidden = consoleId === 1;  // maps for Mario, Sonic, Montezuma
  showRewardPanel($("reward-terms-section"), consoleId);
  $("map-zoom").hidden = consoleId !== 2 && consoleId !== 3;  // built maps only
  mapZoom = 1;
  $("param-max-walkers").value = MAX_WALKERS_DEFAULT[consoleId] ?? 4000;
  applyAlgoUi();
  $("map-title").innerHTML = consoleId === 3
    ? "Pyramid map &mdash; swarm" : "Level map &mdash; swarm";
  document.querySelector(".map-credit").innerHTML = consoleId === 2
    ? "Fog of war: the map is revealed by the swarm as it explores. " +
      "Magenta dots: alive walkers &middot; grey: dead &middot; gold ring: best walker."
    : consoleId === 3
    ? "The temple's 24 rooms are revealed as walkers enter them (room " +
      "numbers mark the unexplored ones). Magenta dots: alive walkers " +
      "&middot; grey: dead &middot; gold ring and frame: best walker and its room."
    : 'Magenta dots: alive walkers &middot; grey: dead &middot; gold ring: ' +
      'best walker. Level maps from <a href="https://ian-albert.com/games/' +
      'super_mario_bros_maps/" target="_blank" rel="noreferrer">ian-albert.com</a>.';
  document.querySelector(".map-credit").innerHTML +=
    " In Graph mode thin lines join every node to its parent; small squares are interior nodes.";
  lastSwarm = null;
  mapSizedForKey = null;
  if (consoleId === 0 || consoleId === 3) drawMap();
  try {
    const id = CONSOLE_ROM[consoleId]();
    const spec = romSpec(id);
    const { buf, reason } = await loadRom(id);
    if (token !== loadToken) return;
    if (!buf) {
      if (reason === "locked") {
        $("rom-unlock-section").hidden = false;
        $("rom-pass-hint").textContent =
          `${spec.name} is bundled encrypted; one password unlocks every game.`;
        if (consoleId === 2) $("sonic-rom-row").hidden = false;  // upload still allowed
        setStatus(`Enter the ROM password to unlock ${spec.name}` +
                  (consoleId === 2 ? " (or upload a ROM)" : ""));
        $("rom-pass-input").focus();
      } else if (consoleId === 2) {
        $("sonic-rom-row").hidden = false;
        setStatus("Pick a Sonic The Hedgehog (Genesis) ROM to start " +
                  "(kept in your browser for future visits)");
      } else {
        throw new Error(`${spec.plain} not found (and no encrypted copy at ${spec.enc})`);
      }
      return;
    }
    romBuffer = buf;
    initRun();
  } catch (err) {
    if (token === loadToken) setStatus(String(err), "error");
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
for (const id of ["param-n", "param-seed", "param-max-walkers"]) {
  $(id).addEventListener("change", () => {
    if (romBuffer) initRun();
  });
}
$("param-erase-coef").addEventListener("input", () => {
  $("erase-coef-value").textContent = parseFloat($("param-erase-coef").value).toFixed(2);
  if (initialized) worker.postMessage({ type: "setParams", params: readParams() });
});

// Algorithm toggle (Wave / Graph): structural change -> restart the run.
// The Graph reinterprets N as its start walkers / minimum leaves, has no
// elite buffer, and adds a population cap and (in Coords mode on a game
// with a map) the visit-count erase coefficient.
function applyAlgoUi() {
  const graph = algorithm === 1;
  for (const b of $("algo-select").querySelectorAll("button")) {
    b.classList.toggle("active", parseInt(b.dataset.algo, 10) === algorithm);
  }
  $("param-n-label").textContent = graph ? "Leaves (start = min leaves)" : "Walkers (N)";
  $("param-elite-row").hidden = graph;
  $("param-max-walkers-row").hidden = !graph;
  const visits = graph && obsMode === 3 && consoleId !== 1;
  $("param-erase-row").hidden = !visits;
  $("graph-visits-hint").hidden = !graph || visits;
  for (const el of document.querySelectorAll(".graph-stat, .graph-plot")) el.hidden = !graph;
  if (!graph) $("max-walkers-hint").textContent = "";
}
function setAlgorithm(algo) {
  algorithm = algo;
  $("param-max-walkers").value = MAX_WALKERS_DEFAULT[consoleId] ?? 4000;
  applyAlgoUi();
}
for (const btn of $("algo-select").querySelectorAll("button")) {
  btn.addEventListener("click", () => {
    const algo = parseInt(btn.dataset.algo, 10);
    if (algo === algorithm) return;
    setAlgorithm(algo);
    if (romBuffer) initRun();
  });
}

// Observation mode toggles: structural change -> restart the run.
function setObsMode(mode) {
  obsMode = mode;
  for (const b of $("obs-mode").querySelectorAll("button")) {
    b.classList.toggle("active", parseInt(b.dataset.mode, 10) === mode);
  }
  applyAlgoUi();
}
for (const btn of $("obs-mode").querySelectorAll("button")) {
  btn.addEventListener("click", () => {
    const mode = parseInt(btn.dataset.mode, 10);
    if (mode === obsMode) return;
    setObsMode(mode);
    if (romBuffer) initRun();
  });
}

// Reward term weights: live-tunable like the fitness coefficients. Built
// before initHelp() so the generated labels get their help icons.
buildRewardPanel($("reward-terms-section"), (termConsole) => {
  if (initialized && termConsole === consoleId) {
    worker.postMessage({ type: "setRewardWeights",
                         weights: readRewardWeights(consoleId) });
  }
});
showRewardPanel($("reward-terms-section"), consoleId);

updateButtons();
initHelp();


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

// Atari game picker: every ALE-supported rom bundled under web/roms/atari/
// (Montezuma's Revenge also has its own console button with dedicated logic).
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

function atariGameName(id) {
  return id.split("_").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
}
const atariSelect = $("atari-game");
for (const id of ATARI_GAMES) {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = atariGameName(id);
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
  const buf = await file.arrayBuffer();
  romCache.set("sonic", buf);
  romDbPut("sonic", buf);
  $("sonic-rom-row").hidden = true;
  $("rom-unlock-section").hidden = true;
  if (consoleId === 2) loadConsoleAssets();
});

// One password for the whole vault: decrypt the ROM that is waiting, then
// remember the password (memory + IndexedDB) so every other game unlocks
// without asking again.
async function unlockRoms() {
  const pass = $("rom-pass-input").value;
  if (!pass || !pendingRom) return;
  const hint = $("rom-pass-hint");
  try {
    hint.textContent = "Decrypting\u2026";
    const buf = await decryptRom(pendingRom.enc, pass);
    romPassword = pass;
    romDbPut(ROM_PASSWORD_KEY, pass);
    romCache.set(pendingRom.id, buf);
    romDbPut(pendingRom.id, buf);
    pendingRom = null;
    $("rom-pass-input").value = "";
    $("rom-unlock-section").hidden = true;
    $("sonic-rom-row").hidden = true;
    loadConsoleAssets();
  } catch (err) {
    // GCM tag mismatch => wrong password.
    hint.textContent = "Wrong password \u2014 try again.";
    romPassword = null;
  }
}
$("rom-pass-unlock").addEventListener("click", unlockRoms);
$("rom-pass-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") unlockRoms();
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
