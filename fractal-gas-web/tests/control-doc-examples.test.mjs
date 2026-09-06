// Validate the downloadable scenes used by the editor course against real WASM.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { sceneReadout } from "../web/lab/scene-presentation.js";

const module = await loadNative();
async function scene(name) {
  return JSON.parse(
    await readFile(
      new URL(
        `../../docs/_static/control_lab/examples/${name}.json`,
        import.meta.url,
      ),
    ),
  );
}
function move(engine, body, [x, y]) {
  const rows = engine.states();
  rows[8 + body] = x;
  rows[8 + engine.bodies + body] = y;
  rows[8 + 2 * engine.bodies + body] = 0;
  rows[8 + 3 * engine.bodies + body] = 0;
  engine.restoreRows(rows);
}
for (const workshop of ["foraging", "cargo", "kart"]) {
  for (const phase of ["starter", "finished"]) {
    const name = `${workshop}-${phase}`;
    test(`${name} compiles, responds to its actuator, and restores exactly`, async () => {
      const config = await scene(name),
        engine = new NativeEngine(module, config);
      try {
        engine.reset(7);
        const initial = engine.snapshot(),
          position = engine.states()[8];
        const action = engine.neutralAction();
        action[0] = 0.5;
        engine.step(action, 12);
        assert.equal(engine.metrics()[4], 12);
        assert(
          engine.states()[8] > position,
          "Positive forward input moves the initial agent along x",
        );
        const future = engine.snapshot();
        engine.restore(initial);
        engine.step(action, 12);
        assert.deepEqual(engine.snapshot(), future);
        assert.equal(engine.metrics()[3], 0);
      } finally {
        engine.dispose();
      }
    });
  }
}
test("foraging workshop counts a pickup and waits before respawning its slot", async () => {
  const config = await scene("foraging-finished"),
    engine = new NativeEngine(module, config);
  try {
    move(engine, 0, config.pickups[0].position);
    engine.step(engine.neutralAction(), 1);
    assert.equal(engine.metrics()[6], 1);
    engine.step(engine.neutralAction(), 30);
    assert.equal(
      engine.metrics()[6],
      1,
      "The same slot cannot be collected each frame",
    );
    assert.equal(sceneReadout(config, engine.metrics()).score, 1);
    assert.equal(config.evaluation.metric, "pickups");
  } finally {
    engine.dispose();
  }
});
test("cargo workshop delivers its cargo once when the cargo centre enters the base", async () => {
  const config = await scene("cargo-finished"),
    engine = new NativeEngine(module, config);
  try {
    move(engine, 1, config.bases[0].position);
    engine.step(engine.neutralAction(), 1);
    assert.equal(engine.metrics()[5], 1);
    engine.step(engine.neutralAction(), 12);
    assert.equal(
      engine.metrics()[5],
      1,
      "This workshop's cargo does not respawn",
    );
    assert.equal(sceneReadout(config, engine.metrics()).score, 1);
    assert.equal(config.evaluation.metric, "deliveries");
  } finally {
    engine.dispose();
  }
});
test("kart workshop counts the four gates in order, independently of its score label", async () => {
  const config = await scene("kart-finished"),
    engine = new NativeEngine(module, config);
  try {
    move(engine, 0, config.gates[1].position);
    engine.step(engine.neutralAction(), 1);
    assert.equal(engine.metrics()[7], 0);
    for (const [i, gate] of config.gates.entries()) {
      move(engine, 0, gate.position);
      engine.step(engine.neutralAction(), 1);
      assert.equal(engine.metrics()[7], i + 1);
    }
    engine.step(engine.neutralAction(), 12);
    assert.equal(engine.metrics()[7], 4);
    assert.equal(
      sceneReadout(config, engine.metrics()).score,
      config.evaluation.target,
    );
  } finally {
    engine.dispose();
  }
});
