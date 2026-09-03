// Genesis core worker: holds ONE instantiation of the statically-linked
// GPGX shim module (retro_shim.js) and processes step/boot/render jobs
// written into the main module's shared memory by RetroFarmEnv
// (src/retro_farm_env.cpp — the region layout and ctrl protocol live there).
//
// The blocking Atomics.wait job loop is legal here (dedicated worker, not
// the browser main thread) and runs forever until terminate().

self.onmessage = async (e) => {
  const { sab, regionOffset, game, mode, zone, act, rom } = e.data;
  const i32 = new Int32Array(sab);
  const f32 = new Float32Array(sab);
  const u8 = new Uint8Array(sab);

  // Header word indices (see RetroFarmEnv).
  const W = regionOffset >> 2;
  const CTRL = W, STATUS = W + 1, ACTION = W + 2, DT = W + 3, DONE = W + 4,
        BLOBLEN = W + 5, REWARD = W + 6, DISPLAY = W + 7;
  // Data areas (byte offsets).
  const HEADER = 128, BLOB_CAP = 0x200000, OBS_CAP = 0x100000,
        RGBA_CAP = 0x80000;
  // Live reward term weights (RetroFarmEnv::set_reward_weights): a count
  // word then float32 words in SonicRewardWeights order, re-read on every
  // STEP so a change made between steps applies to the next one.
  const WEIGHT_COUNT = W + 16, WEIGHTS = W + 17, MAX_WEIGHTS = 7;
  const weightsCache = new Float32Array(MAX_WEIGHTS).fill(NaN);
  const BLOB = regionOffset + HEADER;
  const OBS = BLOB + BLOB_CAP;
  const RGBA = OBS + OBS_CAP;
  const TILE = RGBA + RGBA_CAP;
  const RGBA_LEN = 320 * 224 * 4;
  const POS_WORDS = 6;          // header words kX..kCamY at W+8..W+13
  const TILE_LEN = 40 * 28 * 3; // fog-of-war tile

  const fail = (msg) => {
    console.error("core-worker:", msg);
    Atomics.store(i32, STATUS, -1);
    self.postMessage({ error: String(msg) });
  };

  let shim;
  try {
    const { default: createRetroShim } = await import("./retro_shim.js");
    shim = await createRetroShim();

    // ROM into the shim's own heap, then init.
    const romPtr = shim._malloc(rom.length);
    shim.HEAPU8.set(rom, romPtr);
    const blobLen = shim._shim_init(romPtr, rom.length, game, mode,
                                    zone | 0, act | 0);
    shim._free(romPtr);
    if (blobLen <= 0) {
      return fail("shim_init: " + shim.UTF8ToString(shim._shim_error()));
    }
    const obsDim = shim._shim_obs_dim();

    // Shim-side scratch buffers.
    const sBlob = shim._malloc(blobLen);
    const sObs = shim._malloc(obsDim * 4);
    const sReward = shim._malloc(4);
    const sDone = shim._malloc(4);
    const sDisplay = shim._malloc(4);
    const sRgba = shim._malloc(RGBA_LEN);
    const sPos = shim._malloc(POS_WORDS * 4);
    const sTile = shim._malloc(TILE_LEN);

    Atomics.store(i32, BLOBLEN, blobLen);
    Atomics.store(i32, STATUS, 1);  // ready
    self.postMessage({ ready: true, blobLen });

    const copyBlobIn = () =>
      shim.HEAPU8.set(u8.subarray(BLOB, BLOB + blobLen), sBlob);
    const copyBlobOut = () =>
      u8.set(shim.HEAPU8.subarray(sBlob, sBlob + blobLen), BLOB);
    const copyObsOut = () =>
      u8.set(shim.HEAPU8.subarray(sObs, sObs + obsDim * 4), OBS);

    // Job loop — blocks this worker forever.
    for (;;) {
      Atomics.wait(i32, CTRL, 0);
      const cmd = Atomics.load(i32, CTRL);
      if (cmd <= 0) continue;  // spurious wake / already-handled error state
      let rc = 0;
      try {
        if (cmd === 1) {  // STEP
          const nWeights = Math.min(Atomics.load(i32, WEIGHT_COUNT), MAX_WEIGHTS);
          if (nWeights > 0) {
            let changed = false;
            for (let k = 0; k < nWeights; k++) {
              const v = f32[WEIGHTS + k];
              if (v !== weightsCache[k]) { weightsCache[k] = v; changed = true; }
            }
            if (changed) shim._shim_set_sonic_weights(...weightsCache.subarray(0, nWeights));
          }
          copyBlobIn();
          rc = shim._shim_step(sBlob, Atomics.load(i32, ACTION),
                               Atomics.load(i32, DT), sObs, sReward, sDone,
                               sDisplay, sPos, sTile);
          if (rc === 0) {
            copyBlobOut();
            copyObsOut();
            f32[REWARD] = shim.HEAPF32[sReward >> 2];
            Atomics.store(i32, DONE, shim.HEAP32[sDone >> 2]);
            f32[DISPLAY] = shim.HEAPF32[sDisplay >> 2];
            for (let k = 0; k < POS_WORDS; k++) {
              Atomics.store(i32, W + 8 + k, shim.HEAP32[(sPos >> 2) + k]);
            }
            u8.set(shim.HEAPU8.subarray(sTile, sTile + TILE_LEN), TILE);
          }
        } else if (cmd === 2) {  // BOOT
          rc = shim._shim_boot(sBlob, sObs);
          if (rc === 0) {
            copyBlobOut();
            copyObsOut();
          }
        } else if (cmd === 3) {  // RENDER
          copyBlobIn();
          rc = shim._shim_render(sBlob, sRgba);
          if (rc === 0) {
            u8.set(shim.HEAPU8.subarray(sRgba, sRgba + RGBA_LEN), RGBA);
          }
        }
        if (rc !== 0) {
          console.error("core-worker job", cmd, "failed:",
                        shim.UTF8ToString(shim._shim_error()));
        }
      } catch (err) {
        console.error("core-worker job", cmd, "threw:", err);
        rc = -1;
      }
      Atomics.store(i32, CTRL, rc === 0 ? 0 : -1);
      Atomics.notify(i32, CTRL);
    }
  } catch (err) {
    fail(String(err));
  }
};
