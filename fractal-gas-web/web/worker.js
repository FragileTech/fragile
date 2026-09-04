// Web Worker owning the wasm module and the run loop. fg.step() blocks while
// the pthread pool works, so it must live here — never on the main thread.

let fg = null;
let running = false;
let stepScheduled = false;
let currentConsole = 0;
let currentGame = 0;
// Map "visits" heatmap: ship the Graph's visit-count blocks with each step
// only while the UI shows them (the flag outlives init, so the toggle
// survives restarts).
let visitOverlay = false;

function visitBlocks() {
  if (!visitOverlay || !fg || !fg.countingVisits()) return null;
  return fg.getVisitBlocks();
}

async function loadModule() {
  if (fg) return fg;
  const { default: createFractalGasModule } = await import("./fractal_gas.js");
  fg = await createFractalGasModule();
  return fg;
}

// --- Genesis core-worker farm -----------------------------------------------
// The Genesis backend runs one statically-linked GPGX shim module per plain
// Web Worker (see core-worker.js / src/retro_farm_env.cpp). Spawning nested
// workers requires THIS worker's event loop to be alive, and fg.init blocks
// it — so the farm is spawned and awaited HERE, before fg.init, and the
// pre-spawned farm (region pointer, worker count, blob size) is handed to
// C++ through the params. The job protocol afterwards is pure Atomics and
// needs no event loop. Layout constants mirror RetroFarmEnv.
const FARM_HEADER = 128;
const FARM_BLOB_CAP = 0x200000;
const FARM_OBS_CAP = 0x100000;
const FARM_RGBA_CAP = 0x80000;
const FARM_TILE_CAP = 0x1000;  // fog-of-war tile slot
const FARM_REGION_SIZE =
  FARM_HEADER + FARM_BLOB_CAP + FARM_OBS_CAP + FARM_RGBA_CAP + FARM_TILE_CAP;

let farmWorkers = [];
let farmRegionsPtr = 0;

function farmShutdown() {
  for (const w of farmWorkers) w.terminate();
  farmWorkers = [];
  if (farmRegionsPtr && fg) fg._free(farmRegionsPtr);
  farmRegionsPtr = 0;
}

async function farmSpawn(rom, n, game, mode, zone, act) {
  farmShutdown();
  farmRegionsPtr = fg._malloc(n * FARM_REGION_SIZE);
  fg.HEAPU8.fill(0, farmRegionsPtr, farmRegionsPtr + n * FARM_REGION_SIZE);
  const romBytes = new Uint8Array(rom);
  const readiness = [];
  for (let i = 0; i < n; i++) {
    const w = new Worker(new URL("core-worker.js", self.location.href),
                         { type: "module" });
    readiness.push(new Promise((resolve, reject) => {
      w.onmessage = (e) => (e.data.ready ? resolve(e.data) : reject(
        new Error(e.data.error || "core worker failed")));
      w.onerror = (e) => reject(new Error("core worker: " + e.message));
    }));
    w.postMessage({
      sab: fg.HEAPU8.buffer,
      regionOffset: farmRegionsPtr + i * FARM_REGION_SIZE,
      game,
      mode,
      zone,
      act,
      rom: romBytes,
    });
    farmWorkers.push(w);
  }
  const infos = await Promise.all(readiness);
  return { regionsPtr: farmRegionsPtr, blobLen: infos[0].blobLen };
}

function post(type, payload, transfer) {
  self.postMessage({ type, ...payload }, transfer ?? []);
}

// --- Montezuma room capture ---------------------------------------------------
// The pyramid map is BUILT by the swarm: when an alive walker stands in a
// (level, room) pair, its frame is rendered, the 50-row HUD cropped away, and
// the 160x160 room image shipped to the UI. Two pitfalls of a naive "first
// frame wins" capture, both fixed here:
//   - Room transitions show a black (or solid blue 0,28,136) screen: a frame
//     counts only if it has at least a sprite's worth of lit pixels and is not
//     the blue fill. The level-1 lower rooms are DARK (no torch) and show only
//     sprites, so a fraction-of-the-room threshold would never accept them.
//   - The game cycles the whole room palette for ~1 s when an item is picked
//     up (and while dying): a frame taken then would freeze pink/purple walls
//     into the map. Rooms are therefore re-captured every MZ_RECAPTURE_STEPS
//     iterations while a walker is in them, and the UI only gets a frame when
//     it has MORE black background than the one it already shows — a real
//     Atari room is mostly black, a palette flash is not.
// Layout constants mirror src/montezuma_logic.hpp.
const MZ_HUD_ROWS = 50;
const MZ_ROOM_W = 160;
const MZ_ROOM_H = 160;
const MZ_MAX_CAPTURES_PER_STEP = 3;
const MZ_MIN_LIT_PIXELS = 40;          // Panama Joe alone is ~150 lit pixels
const MZ_TRANSITION_FILL_FRACTION = 0.9;  // the blue between-room screen
const MZ_RECAPTURE_STEPS = 30;
// key "level:room" -> { black: pixels of the frame shown, seen: iteration
// of the last evaluation }
let roomCaptures = new Map();

function isMontezuma() {
  return currentConsole === 1 && currentGame === 1;
}

// Crop the HUD and classify the room image: null when it is a transition
// screen, else { rgba, black } (black = count of black pixels).
function evaluateRoomFrame(view, frameW) {
  const start = MZ_HUD_ROWS * frameW * 4;
  const room = new Uint8ClampedArray(view.buffer, view.byteOffset + start,
                                     MZ_ROOM_W * MZ_ROOM_H * 4).slice();
  let lit = 0, black = 0, blue = 0;
  for (let p = 0; p < room.length; p += 4) {
    const r = room[p], g = room[p + 1], b = room[p + 2];
    if (r | g | b) {
      lit++;
      if (r === 0 && g === 28 && b === 136) blue++;
    } else {
      black++;
    }
  }
  const total = MZ_ROOM_W * MZ_ROOM_H;
  if (lit < MZ_MIN_LIT_PIXELS) return null;                    // black transition
  if (blue > MZ_TRANSITION_FILL_FRACTION * total) return null;  // blue transition
  return { rgba: room.buffer, black };
}

function captureRooms(stats) {
  if (!stats || !stats.walkerWorld) return [];
  const frameW = fg.frameWidth();
  const frameH = fg.frameHeight();
  if (frameW !== MZ_ROOM_W || frameH < MZ_HUD_ROWS + MZ_ROOM_H) return [];
  const frames = [];
  const rooms = stats.walkerWorld;
  const levels = stats.walkerStage;
  const alive = stats.walkerAlive;
  const iteration = stats.iteration || 0;
  const tried = new Set();  // one render per room per step
  for (let i = 0; i < rooms.length && frames.length < MZ_MAX_CAPTURES_PER_STEP; i++) {
    if (!alive[i]) continue;
    const key = `${levels[i]}:${rooms[i]}`;
    if (tried.has(key)) continue;
    const prev = roomCaptures.get(key);
    if (prev && iteration - prev.seen < MZ_RECAPTURE_STEPS) continue;
    tried.add(key);
    const view = fg.renderWalkerFrame(i);
    if (!view) continue;
    const evaluated = evaluateRoomFrame(view, frameW);
    if (!evaluated) continue;  // transition screen: try again next step
    if (prev && evaluated.black <= prev.black) {
      // Not better than the image shown (e.g. a palette flash): keep the old
      // one, check again later.
      prev.seen = iteration;
      continue;
    }
    roomCaptures.set(key, { black: evaluated.black, seen: iteration });
    frames.push({ level: levels[i], room: rooms[i],
                  width: MZ_ROOM_W, height: MZ_ROOM_H, rgba: evaluated.rgba });
  }
  return frames;
}

function stepOnce() {
  stepScheduled = false;
  if (!running || !fg) return;

  const stats = fg.step();
  const frameView = fg.getBestFrame();
  let frame = null;
  if (frameView) {
    // Copy out of wasm memory so the buffer can be transferred.
    frame = new Uint8ClampedArray(frameView).buffer;
  }
  let walkerTiles = null;
  if (currentConsole === 2) {
    const tiles = fg.getWalkerTiles();
    if (tiles) walkerTiles = new Uint8Array(tiles).buffer;
  }
  const roomFrames = isMontezuma() ? captureRooms(stats) : [];
  const visits = visitBlocks();
  const transfers = [];
  if (frame) transfers.push(frame);
  if (walkerTiles) transfers.push(walkerTiles);
  for (const rf of roomFrames) transfers.push(rf.rgba);
  if (visits) transfers.push(visits.keys.buffer, visits.sums.buffer);
  post(
    "step",
    {
      stats,
      frame,
      walkerTiles,
      roomFrames,
      visits,
      frameWidth: fg.frameWidth(),
      frameHeight: fg.frameHeight(),
    },
    transfers,
  );

  // stop_when_all_dead: with every walker dead the swarm can only clone
  // among dead states - stop the run and tell the UI.
  if (stats && stats.aliveCount === 0) {
    running = false;
    post("allDead", { iteration: stats.iteration });
    return;
  }

  if (running && !stepScheduled) {
    stepScheduled = true;
    setTimeout(stepOnce, 0); // yield so incoming messages are processed
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case "init": {
        running = false;
        await loadModule();
        // embind requires every FgParams field; default the algorithm
        // fields so callers that predate them (autotest pages) still work.
        const params = { algorithm: 0, maxWalkers: 0, eraseCoef: 0.05, aggBlock: 5,
                         visitReward: (msg.params.algorithm ?? 0) === 1, visitCoef: 1.0,
                         ...msg.params };
        currentConsole = params.console;
        currentGame = params.game;
        roomCaptures = new Map();
        if (params.console === 2) {
          // Pre-spawn the Genesis core-worker farm (see above).
          const n = Math.min(Math.max(params.nThreads, 1), 8);
          // For Sonic, params.world/stage carry the internal zone id and
          // 0-based act (mapped by main.js).
          const farm = await farmSpawn(msg.rom, n, params.game, params.obsMode,
                                       params.world, params.stage);
          params.farmPtr = farm.regionsPtr;
          params.farmWorkers = n;
          params.farmBlobLen = farm.blobLen;
        } else {
          farmShutdown();
          params.farmPtr = 0;
          params.farmWorkers = 0;
          params.farmBlobLen = 0;
        }
        const aux = msg.aux ? new Uint8Array(msg.aux) : new Uint8Array(0);
        const ok = fg.init(new Uint8Array(msg.rom), aux, params);
        if (ok) {
          if (msg.rewardWeights) fg.setRewardWeights(msg.rewardWeights);
          // Graph mode: the effective population cap after the wasm memory
          // clamp (may be below the requested max walkers).
          post("ready", { algorithm: fg.algorithm(), maxWalkers: fg.maxWalkers(),
                          countingVisits: fg.countingVisits() });
        } else {
          post("error", { message: fg.lastError() });
        }
        break;
      }
      case "start":
        if (fg) {
          running = true;
          if (!stepScheduled) {
            stepScheduled = true;
            setTimeout(stepOnce, 0);
          }
        }
        break;
      case "pause":
        running = false;
        break;
      case "reset":
        running = false;
        if (fg) {
          fg.reset();
          roomCaptures = new Map();
          post("resetDone", {});
        }
        break;
      case "setRewardWeights":
        if (fg) fg.setRewardWeights(msg.weights);
        break;
      case "setVisitOverlay": {
        visitOverlay = !!msg.on;
        // Show the current grid right away (e.g. toggled while paused).
        const visits = visitBlocks();
        if (visits) post("visits", { visits }, [visits.keys.buffer, visits.sums.buffer]);
        break;
      }
      case "setParams":
        // embind requires every FgParams field; the farm fields are only
        // meaningful at init, so zeros suffice here.
        if (fg) {
          fg.setParams({ farmPtr: 0, farmWorkers: 0, farmBlobLen: 0,
                         algorithm: 0, maxWalkers: 0, eraseCoef: 0.05, aggBlock: 5,
                         visitReward: (msg.params.algorithm ?? 0) === 1, visitCoef: 1.0,
                         ...msg.params });
        }
        break;
      default:
        break;
    }
  } catch (err) {
    running = false;
    post("error", { message: String(err) });
  }
};
