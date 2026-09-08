import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";

const source = await readFile(new URL("../web/worker.js", import.meta.url), "utf8");
function harness(steps) {
  const messages = [], scheduled = [], configs = [];
  const mock = {
    step: () => steps.shift(),
    getBestFrame: () => null,
    frameWidth: () => 0,
    frameHeight: () => 0,
    algorithm: () => 3,
    maxWalkers: () => 4,
    countingVisits: () => false,
    init: (_rom, _aux, params) => { configs.push(params); return true; },
    reset: () => {},
    setParams: params => { configs.push(params); return true; },
  };
  const context = vm.createContext({
    self: { postMessage: msg => messages.push(msg) },
    setTimeout: fn => scheduled.push(fn), __mock: mock,
  });
  vm.runInContext(source + "\nfg = __mock;", context);
  const send = data => context.self.onmessage({ data });
  const tick = () => { const fn = scheduled.shift(); if (fn) fn(); };
  return { messages, scheduled, configs, send, tick };
}

test("planner worker keeps running after search extinction and stops at committed game end", async () => {
  const h = harness([
    { algorithm: 3, aliveCount: 0, gameDone: false, phase: "planning" },
    { algorithm: 3, aliveCount: 0, gameDone: false, phase: "playing", playedFrames: 2 },
    { algorithm: 3, aliveCount: 0, gameDone: true, phase: "ended", playedFrames: 4 },
  ]);
  await h.send({ type: "start" });
  h.tick(); h.tick(); h.tick();
  assert.deepEqual(h.messages.map(m => m.type), ["step", "step", "step", "gameDone"]);
  assert.equal(h.scheduled.length, 0);
});

test("Wave and Graph retain all-dead stopping", async () => {
  for (const algorithm of [0, 1]) {
    const h = harness([{ algorithm, aliveCount: 0, iteration: 5 }]);
    await h.send({ type: "start" }); h.tick();
    assert.deepEqual(h.messages.map(m => m.type), ["step", "allDead"]);
    assert.equal(h.scheduled.length, 0);
  }
});

test("pause and reset cancel scheduled execution without consuming another step", async () => {
  const h = harness([{ algorithm: 2, aliveCount: 4, playedFrames: 0 }]);
  await h.send({ type: "start" });
  await h.send({ type: "pause" }); h.tick();
  assert.equal(h.messages.length, 0);
  await h.send({ type: "start" });
  await h.send({ type: "reset" }); h.tick();
  assert.deepEqual(h.messages.map(m => m.type), ["resetDone"]);
  await h.send({ type: "start" }); h.tick();
  assert.equal(h.messages.at(-1).type, "step");
});

test("old worker callers receive planner defaults and invalid horizons remain recoverable", async () => {
  const h = harness([]);
  await h.send({ type: "init", rom: [], params: { console: 0, game: 0, algorithm: 0 } });
  assert.equal(h.configs[0].horizon, 32);
  assert.equal(h.configs[0].maxHorizon, 0);
  assert.equal(h.configs[0].consensusPrefix, true);
  await h.send({ type: "setParams", params: { algorithm: 3, horizon: 4, maxHorizon: 2 } });
  assert.equal(h.configs.length, 1);
  assert.equal(h.messages.at(-1).recoverable, true);
  assert.match(h.messages.at(-1).message, /Maximum search horizon/);
  await h.send({ type: "setParams", params: { algorithm: 3, horizon: 2.5 } });
  assert.match(h.messages.at(-1).message, /integer/);
  await h.send({ type: "setParams", params: { algorithm: 3, horizon: 4, maxHorizon: 8 } });
  assert.equal(h.configs.length, 2);
});

test("planner errors surface from the scheduled loop and permit reset", async () => {
  const h = harness([{ error: "Planner found no executable trajectory. Reset or change the search settings." }]);
  await h.send({ type: "start" }); h.tick();
  assert.equal(h.messages[0].type, "error");
  assert.equal(h.messages[0].recoverable, true);
  assert.equal(h.scheduled.length, 0);
  await h.send({ type: "reset" });
  assert.equal(h.messages.at(-1).type, "resetDone");
});
