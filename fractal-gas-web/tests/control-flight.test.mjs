import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  configureAntsScene,
  DEFAULT_ANTS_OPTIONS,
} from "../web/lab/ants-scene.js";
import { flightMode, resolveBodies } from "../web/lab/agent-types.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";

const readScene = async (name) =>
  JSON.parse(
    await readFile(new URL(`../web/lab/scenarios/${name}.json`, import.meta.url), "utf8"),
  );

test("Flight mode follows capability inheritance and explicit overrides", async () => {
  const rocket = await readScene("rocket");
  const racing = await readScene("racing");
  const ants = await readScene("ants");

  assert.equal(flightMode(rocket), true);
  assert.equal(flightMode(racing), false);
  assert.equal(flightMode(ants), false);
  assert.equal(
    flightMode(configureAntsScene(ants, { ...DEFAULT_ANTS_OPTIONS, agentType: "drone" })),
    true,
  );
  assert.equal(
    flightMode({ ...rocket, environment: { flight: false } }),
    false,
  );
  assert.equal(
    flightMode({
      ...ants,
      environment: { flight: true },
    }),
    true,
  );
  assert.equal(resolveBodies(rocket)[0].flight_capable, true);
});

test("Native flight mode applies downward gravity without changing state shape", async () => {
  const module = await loadNative(false);
  const scene = {
    size: [100, 100],
    physics: { dt: 0.1, substeps: 1 },
    environment: { flight: true, downward_gravity: 10 },
    bodies: [
      {
        position: [50, 50],
        controlled: true,
        flight_capable: false,
        drag: 0,
        angular_drag: 0,
      },
    ],
  };
  const engine = new NativeEngine(module, scene);
  try {
    const before = engine.states();
    engine.step(engine.neutralAction(), 1);
    const after = engine.states();
    assert.equal(after.length, engine.stride);
    assert.equal(after[8], before[8]);
    assert.ok(Math.abs(after[9] - 49.9) < 1e-5);
    assert.equal(after[8 + 3], -1);
    assert.equal(engine.inspect(engine.neutralAction())[6], -10);
  } finally {
    engine.dispose();
  }
});
