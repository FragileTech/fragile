import test from "node:test";
import assert from "node:assert/strict";
import { playbackController } from "../web/playback-controller.js";

function harness() {
  const workers = [], received = [], sizes = [], search = [];
  globalThis.Worker = class {
    constructor() { workers.push(this); this.sent = []; }
    postMessage(msg) { this.sent.push(msg); }
    terminate() { this.terminated = true; }
    emit(data) { const init = this.sent[0]; this.onmessage({ data: { request: init.request, runtime: init.runtime, ...data } }); }
  };
  const controller = playbackController({ sendSearch: m => search.push(m),
    receive: m => received.push(m), memory: n => sizes.push(n),
    config: () => ({ rom: new ArrayBuffer(1), params: { console: 2 } }) });
  const capture = request => { controller.send({ type: "trajectorySelect", request });
    controller.captured({ request, root: new ArrayBuffer(1), actions: new ArrayBuffer(8), walker: 1 }); };
  return { controller, capture, workers, received, sizes, search };
}
test("replacement terminates old player and ignores stale frames and errors", () => {
  const h = harness(); h.capture(1); const first = h.workers[0];
  h.capture(2); const second = h.workers[1];
  assert.ok(first.terminated);
  first.emit({ type: "trajectoryFrame", frame: new ArrayBuffer(4) });
  first.onerror({ message: "old worker failed" });
  assert.equal(h.received.length, 0); assert.ok(!second.terminated);
  second.emit({ type: "trajectorySelected", playbackBytes: 64 * 1024 ** 2 });
  assert.equal(h.received.length, 1); assert.equal(h.sizes.at(-1), 64 * 1024 ** 2);
  h.controller.dispose(); assert.ok(second.terminated); assert.equal(h.sizes.at(-1), 0);
});
test("playback failure releases heap and permits retry without pausing search", () => {
  const h = harness(); h.capture(1);
  h.workers[0].emit({ error: "Playback memory limit reached" });
  assert.ok(h.workers[0].terminated); assert.equal(h.sizes.at(-1), 0);
  assert.match(h.received[0].error, /memory limit/);
  h.capture(2); assert.equal(h.workers.length, 2);
  assert.ok(h.search.every(m => m.type === "trajectorySelect"));
  h.controller.dispose();
});
test("reset rejects in-flight capture and releases existing player", () => {
  const h = harness(); h.controller.send({ type: "trajectorySelect", request: 1 });
  h.controller.dispose(); h.controller.captured({ request: 1 });
  assert.equal(h.workers.length, 0);
  h.capture(2); h.controller.dispose(); assert.ok(h.workers[0].terminated);
});

test("worker creation failure is local and recoverable", () => {
  const h = harness(); globalThis.Worker = class { constructor() { throw new Error("worker refused"); } };
  h.capture(1);
  assert.match(h.received[0].error, /worker refused/);
  assert.equal(h.sizes.at(-1), 0);
  h.controller.dispose();
});
