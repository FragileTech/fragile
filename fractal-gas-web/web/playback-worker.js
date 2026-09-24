import { PLAYBACK_INITIAL, PLAYBACK_LIMIT, RECORDING_LIMIT, PAGE } from "./arcade-resources.js";

let module, memory, root, actions, cursor = 0, adapter;
let generation, runtime;
const post = (type, data = {}, transfer = []) => self.postMessage({
  type, request: generation, runtime, playbackBytes: memory?.buffer.byteLength || 0, ...data,
}, transfer);
async function initialize(msg) {
  generation = msg.request; runtime = msg.runtime;
  root = new Uint8Array(msg.root); actions = new Int32Array(msg.actions);
  if (!root.length || actions.length % 2 || root.byteLength + actions.byteLength > RECORDING_LIMIT)
    throw new Error("Playback recording exceeds 32 MiB or is invalid");
  const p = msg.params;
  memory = new WebAssembly.Memory({ initial: PLAYBACK_INITIAL / PAGE,
    maximum: PLAYBACK_LIMIT / PAGE, shared: p.console !== 2 });
  const { default: create } = await import(p.console === 2 ? "./retro_shim.js" : "./arcade_playback.js");
  module = await create({ wasmMemory: memory, onAbort: () => { throw new Error("Playback memory/runtime limit reached"); } });
  if (p.console === 2) {
    const alloc = n => { const ptr = module._malloc(n) >>> 0;
      if (!ptr) throw new Error("Playback memory limit reached"); return ptr; };
    const rom = new Uint8Array(msg.rom), rp = alloc(rom.length);
    module.HEAPU8.set(rom, rp);
    const size = module._shim_init(rp, rom.length, p.game, p.obsMode, p.world, p.stage);
    module._free(rp);
    const check = result => { if (result < 0) throw new Error(module.UTF8ToString(module._shim_error())); };
    check(size);
    if (size !== root.length) throw new Error("Playback snapshot size mismatch");
    const blob = alloc(size), obs = alloc(module._shim_obs_dim() * 4), reward = alloc(4),
      done = alloc(4), display = alloc(4), pos = alloc(24), tile = alloc(3360),
      frames = alloc(4), rgba = alloc(320 * 224 * 4);
    adapter = {
      restore: () => module.HEAPU8.set(root, blob),
      step: (a, dt) => { if (dt > 0) check(module._shim_step(blob, a, dt, obs, reward, done, display, pos, tile, frames)); },
      render: () => { check(module._shim_render(blob, rgba)); return module.HEAPU8.slice(rgba, rgba + 320 * 224 * 4); },
      width: 320, height: 224,
    };
  } else {
    module.initPlayback(new Uint8Array(msg.rom), p.console, p.game, p.obsMode, p.world, p.stage);
    adapter = { restore: () => module.restorePlayback(root),
      step: (a, dt) => module.stepPlayback(a, dt), render: () => module.renderPlayback(),
      width: module.frameWidth(), height: module.frameHeight() };
  }
  adapter.restore(); cursor = 0;
  post("trajectorySelected", { length: actions.length / 2 + 1, walker: msg.walker,
    walkerCount: msg.walkerCount, iteration: msg.iteration });
}
function seek(msg) {
  const target = msg.index;
  if (!Number.isInteger(target) || target < 0 || target > actions.length / 2)
    throw new Error("Invalid playback position");
  if (target < cursor) { adapter.restore(); cursor = 0; }
  // Bound individual turns so superseded seeks and termination stay responsive.
  const end = Math.min(target, cursor + 32);
  for (; cursor < end; cursor++) adapter.step(actions[cursor * 2], actions[cursor * 2 + 1]);
  const ready = cursor === target;
  const frame = ready ? adapter.render().buffer : null;
  post("trajectoryFrame", { ready, index: target, frame,
    frameWidth: adapter.width, frameHeight: adapter.height }, frame ? [frame] : []);
}
let queue = Promise.resolve();
self.onmessage = ({ data: msg }) => {
  queue = queue.then(async () => {
    try {
      if (msg.type === "init") await initialize(msg);
      else if (msg.request === generation && msg.runtime === runtime && msg.type === "trajectoryFrame") seek(msg);
    } catch (error) {
      const reason = typeof error === "number"
        ? "Emulator failed or reached its 256 MiB memory limit. Load the path again."
        : (error?.message || String(error));
      post("trajectoryFrame", { error: "Playback: " + reason });
    }
  });
};
