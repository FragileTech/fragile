import { resourcePlan, MAIN_INITIAL, PAGE } from "./arcade-resources.js";

// Web Worker owning the wasm module and the run loop. fg.step() blocks while
// the pthread pool works, so it must live here — never on the main thread.

let fg = null;
let mainMemory = null;
let allocation = null;
let playbackBytes = 0;
let lastInit = null;
let runtimeFailed = false;
let running = false;
let stepScheduled = false;
let stepTimer = null;
let currentConsole = 0;
let currentGame = 0;
// Map "visits" heatmap: ship the Graph's visit-count blocks with each step
// only while the UI shows them (the flag outlives init, so the toggle
// survives restarts).
let visitOverlay = false;

function plannerDefaults(params) {
  const p = { horizon: 32, consensusPrefix: true, maxHorizon: 0,
              freezePrefixAfter: 0, ...params };
  if (p.algorithm >= 2) {
    if (!Number.isInteger(p.horizon) || p.horizon < 1 || p.horizon > 4096)
      throw new Error("Search horizon must be an integer between 1 and 4096");
    if (typeof p.consensusPrefix !== "boolean")
      throw new Error("Stop at first bifurcation must be a boolean");
    if (p.algorithm === 3 && p.consensusPrefix &&
        (!Number.isInteger(p.maxHorizon) || p.maxHorizon < 0 || p.maxHorizon > 4096 ||
         (p.maxHorizon !== 0 && p.maxHorizon < p.horizon)))
      throw new Error("Maximum search horizon must be 0 or an integer between the normal horizon and 4096");
  }
  return p;
}

function visitBlocks() {
  if (!visitOverlay || !fg || !fg.countingVisits()) return null;
  return fg.getVisitBlocks();
}

async function loadModule(plan) {
  if (fg) return fg;
  mainMemory = new WebAssembly.Memory({ initial: MAIN_INITIAL / PAGE,
    maximum: plan.mainLimitBytes / PAGE, shared: true });
  const { default: createFractalGasModule } = await import("./fractal_gas.js");
  fg = await createFractalGasModule({ wasmMemory: mainMemory,
    arcadeThreadPoolSize: plan.farmWorkers ? 0 : plan.workers - 1,
    onAbort: () => { runtimeFailed = true; } });
  return fg;
}

function disposeRuntime() {
  clearTimeout(stepTimer);
  running = false;
  stepScheduled = false;
  farmShutdown();
  fg?.PThread?.terminateAllThreads();
  fg = null;
  mainMemory = null;
  allocation = null;
  playbackBytes = 0;
  runtimeFailed = false;
}

function memoryStatus() {
  if (!allocation || !mainMemory) return null;
  let emulatorBytes = 0;
  for (let i = 0; i < farmWorkers.length; i++) {
    // Fixed communication regions remain inside their original buffer even
    // after the surrounding memory grows. Read only each region's header.
    const h = new Int32Array(mainMemory.buffer, farmRegionsPtr + i * FARM_REGION_SIZE, 32);
    emulatorBytes += Atomics.load(h, 15) * PAGE;
  }
  const mainBytes = mainMemory.buffer.byteLength;
  return { ...allocation, mainBytes, emulatorBytes, playbackBytes, allocatedBytes: mainBytes + emulatorBytes + playbackBytes,
    graphPopulationCap: fg?.maxWalkers?.() ?? 0 };
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
let farmCleanup = [];
let farmRegionsPtr = 0;

function farmShutdown() {
  for (const cleanup of farmCleanup) cleanup();
  farmCleanup = [];
  for (const w of farmWorkers) w.terminate();
  farmWorkers = [];
  // The entire main runtime is discarded; do not enter an aborted allocator.
  farmRegionsPtr = 0;
}

async function farmSpawn(rom, n, game, mode, zone, act) {
  farmShutdown();
  farmRegionsPtr = fg._malloc(n * FARM_REGION_SIZE) >>> 0;
  if (!farmRegionsPtr) throw new Error("Main engine memory limit: cannot allocate Sonic worker regions");
  new Uint8Array(mainMemory.buffer, farmRegionsPtr, n * FARM_REGION_SIZE).fill(0);
  const romBytes = new Uint8Array(rom);
  const readiness = [];
  for (let i = 0; i < n; i++) {
    const w = new Worker(new URL("core-worker.js", self.location.href),
                         { type: "module" });
    const header = new Int32Array(mainMemory.buffer, farmRegionsPtr + i * FARM_REGION_SIZE, 32);
    readiness.push(new Promise((resolve, reject) => {
      let ready = false;
      const timeout = setTimeout(() => reject(new Error("Sonic worker initialization timed out")), 30000);
      farmCleanup.push(() => { clearTimeout(timeout); reject(new Error("Sonic worker shut down")); });
      const fail = message => {
        clearTimeout(timeout);
        Atomics.store(header, 0, -1);
        Atomics.notify(header, 0);
        if (!ready) reject(new Error(message));
        else {
          running = false;
          runtimeFailed = true;
          post("error", { message: failureMessage(message), requiresReset: true, recoverable: true, resources: memoryStatus() });
        }
      };
      w.onmessage = (e) => {
        clearTimeout(timeout);
        if (e.data.ready) { ready = true; resolve(e.data); }
        else fail(e.data.error || "Sonic worker failed");
      };
      w.onerror = (e) => fail("Sonic emulator worker: " + e.message);
    }));
    w.postMessage({
      sab: mainMemory.buffer,
      memoryLimitBytes: allocation.shimLimitBytes,
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

function failureMessage(err) {
  const message = String(err);
  if (/Sonic emulator/i.test(message) && /bad_alloc|memory limit|out of memory|OOM|Cannot enlarge memory|could not allocate memory/i.test(message))
    return `Sonic emulator memory allocation failed (per-worker limit ${((allocation?.shimLimitBytes ?? 0) / 1024 ** 2).toFixed(0)} MiB). Select fewer workers or a larger engine memory limit and reset. ${message}`;
  if (/bad_alloc|out of memory|OOM|Cannot enlarge memory|could not allocate memory/i.test(message))
    return `Main engine memory allocation failed (limit ${((allocation?.mainLimitBytes ?? MAIN_INITIAL) / 1024 ** 3).toFixed(2)} GiB). Reduce walkers or history, or increase the engine memory limit and reset. ${message}`;
  return message;
}

function stepOnce() {
  try { advanceOnce(); }
  catch (err) {
    stepScheduled = false;
    running = false;
    runtimeFailed = true;
    post("error", { message: failureMessage(err), recoverable: true, requiresReset: true, resources: memoryStatus() });
  }
}

function advanceOnce() {
  stepScheduled = false;
  if (!running || !fg) return;

  const stats = fg.step();
  if (stats?.error) throw new Error(stats.error);
  trajectoryBest = stats.bestWalkerIdx ?? 0;
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
      resources: memoryStatus(),
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
  if (stats?.gameDone) {
    running = false;
    post("gameDone", { playedFrames: stats.playedFrames });
    return;
  }
  if (stats && stats.algorithm < 2 && stats.aliveCount === 0) {
    running = false;
    post("allDead", { iteration: stats.iteration });
    return;
  }

  if (running && !stepScheduled) {
    stepScheduled = true;
    stepTimer = setTimeout(stepOnce, 0); // yield so incoming messages are processed
  }
}

let trajectoryBest = 0;

async function handleMessage(event) {
  const msg = event.data;
  try {
    switch (msg.type) {
      case "init": {
        disposeRuntime();
        trajectoryBest = 0;
        lastInit = msg;
        allocation = resourcePlan({ n: 32, ...msg.params }, msg.resources, self.navigator?.hardwareConcurrency || 4);
        await loadModule(allocation);
        // embind requires every FgParams field; default the algorithm
        // fields so callers that predate them (autotest pages) still work.
        const params = plannerDefaults({ n: 32, useCumulativeReward: true, algorithm: 0, maxWalkers: 0, eraseCoef: 0.05, aggBlock: 5,
                         visitReward: (msg.params.algorithm ?? 0) === 1, visitCoef: 1.0,
                         ...msg.params, nThreads: allocation.workers,
                         memoryLimitBytes: allocation.mainLimitBytes });
        currentConsole = params.console;
        currentGame = params.game;
        roomCaptures = new Map();
        if (params.console === 2) {
          // Pre-spawn the Genesis core-worker farm (see above).
          const n = allocation.workers;
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
          fg.setDistanceMetric(msg.params.distance_metric ?? "l2");
          if (msg.rewardWeights) fg.setRewardWeights(msg.rewardWeights);
          // Graph mode: the effective population cap after the wasm memory
          // clamp (may be below the requested max walkers).
          if (fg.algorithm() !== 1 && params.removal_policy &&
              !fg.setPopulation(params.n, params.removal_policy)) throw new Error(fg.lastError());
          post("ready", { population: fg.populationStatus(), algorithm: fg.algorithm(), maxWalkers: fg.maxWalkers(),
                          countingVisits: fg.countingVisits(), resources: memoryStatus() });
          if (msg.reset) post("resetDone", { resources: memoryStatus() });
        } else {
          throw new Error(fg.lastError());
        }
        break;
      }
      case "playbackMemory": {
        if (Number.isInteger(msg.bytes) && msg.bytes >= 0 && msg.bytes <= (allocation?.playbackLimitBytes ?? 0))
          playbackBytes = msg.bytes;
        break;
      }
      case "trajectorySelect": {
        try {
          const result = fg?.selectTrajectory(msg.walker < 0 ? trajectoryBest : msg.walker) ?? { error: "Start a run first" };
          const root = result.root?.buffer;
          const actions = result.actions?.buffer;
          post("trajectoryRecording", { ...result, root, actions, request: msg.request },
            root && actions ? [root, actions] : []);
        } catch (error) {
          post("trajectoryRecording", { error: "Playback capture: " + failureMessage(error), request: msg.request });
        }
        break;
      }
      case "start":
        if (fg && !runtimeFailed) {
          running = true;
          if (!stepScheduled) {
            stepScheduled = true;
            stepTimer = setTimeout(stepOnce, 0);
          }
        }
        break;
      case "pause":
        running = false;
        clearTimeout(stepTimer);
        stepScheduled = false;
        post("paused", {});
        break;
      case "reset":
        running = false;
        if (lastInit) {
          await handleMessage({ data: lastInit });
          if (fg && !runtimeFailed) post("resetDone", { resources: memoryStatus() });
        } else if (fg) {
          fg.reset();
          post("resetDone", {});
        }
        break;
      case "dispose":
        disposeRuntime();
        lastInit = null;
        post("disposed", {});
        break;
      case "setRewardWeights":
        if (fg) fg.setRewardWeights(msg.weights);
        if (lastInit) lastInit.rewardWeights = msg.weights;
        break;
      case "setVisitOverlay": {
        visitOverlay = !!msg.on;
        // Show the current grid right away (e.g. toggled while paused).
        const visits = visitBlocks();
        if (visits) post("visits", { visits }, [visits.keys.buffer, visits.sums.buffer]);
        break;
      }
      case "setPopulation": {
        if (!Number.isInteger(msg.walkers) || msg.walkers < 2 || msg.walkers > 1024)
          throw new Error("Active walkers must be an integer between 2 and 1024");
        if (!fg) throw new Error("Initialize a run first");
        if (!fg.setPopulation(msg.walkers, msg.removal_policy)) throw new Error(fg.lastError());
        if (lastInit) Object.assign(lastInit.params, { n: msg.walkers, removal_policy: msg.removal_policy });
        post("population", { population: fg.populationStatus() });
        break;
      }
      case "setParams":
        // embind requires every FgParams field; the farm fields are only
        // meaningful at init, so zeros suffice here.
        if (fg) {
          const ok = fg.setParams(plannerDefaults({ memoryLimitBytes: allocation?.mainLimitBytes ?? MAIN_INITIAL, farmPtr: 0, farmWorkers: 0, farmBlobLen: 0,
                         algorithm: fg.algorithm(), maxWalkers: 0, eraseCoef: 0.05, aggBlock: 5,
                         visitReward: (msg.params.algorithm ?? 0) === 1, visitCoef: 1.0,
                         ...msg.params }));
          if (ok === false) throw new Error(fg.lastError());
          if (lastInit) lastInit.params = { ...lastInit.params, ...msg.params };
          if (msg.params.distance_metric !== undefined) fg.setDistanceMetric(msg.params.distance_metric);
        }
        break;
      default:
        break;
    }
  } catch (err) {
    running = false;
    const message = failureMessage(err);
    if (msg.type === "init") disposeRuntime();
    post("error", { message, requiresReset: !["setParams", "setPopulation"].includes(msg.type), recoverable: ["setParams", "setPopulation"].includes(msg.type) || !!lastInit });
  }
}

// Serialize asynchronous initialization/disposal with later UI commands.
let commands = Promise.resolve();
self.onmessage = event => {
  commands = commands.then(() => handleMessage(event));
  return commands;
};
