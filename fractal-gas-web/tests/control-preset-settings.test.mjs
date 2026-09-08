import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { presetControllerSettings } from "../web/lab/preset-settings.js";

const scenes = Object.fromEntries(
  await Promise.all(
    ["harvest", "mining", "racing"].map(async (name) => [
      name,
      JSON.parse(
        await readFile(
          new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
        ),
      ),
    ]),
  ),
);
const standard = {
  algorithm: "wave-jump",
  walkers: 128,
  horizon: 64,
  frames: 12,
  elites: 0,
};

test("flight mining recommendations follow fresh tasks and unchanged preset settings", () => {
  const flight = presetControllerSettings(scenes.harvest, standard);
  assert.deepEqual(flight, { ...standard, horizon: 32, frames: 6, elites: 4 });
  assert.deepEqual(
    presetControllerSettings(scenes.mining, flight, scenes.harvest),
    flight,
  );
  assert.deepEqual(
    presetControllerSettings(scenes.racing, flight, scenes.mining),
    standard,
  );
  assert.deepEqual(
    presetControllerSettings(scenes.mining, standard, scenes.racing),
    flight,
  );
});

test("scenario switches preserve explicit controller tuning and respect small populations", () => {
  const custom = {
    ...standard,
    algorithm: "icem",
    horizon: 20,
    frames: 4,
    elites: 2,
  };
  assert.deepEqual(presetControllerSettings(scenes.harvest, custom), custom);
  assert.deepEqual(
    presetControllerSettings(scenes.mining, custom, scenes.harvest),
    custom,
  );
  assert.deepEqual(
    presetControllerSettings(scenes.harvest, custom, scenes.racing, true),
    {
      ...custom,
      horizon: 32,
      frames: 6,
      elites: 4,
    },
  );
  assert.equal(
    presetControllerSettings(scenes.mining, { ...standard, walkers: 1 }).elites,
    1,
  );
  assert.deepEqual(presetControllerSettings(scenes.mining, standard), {
    ...standard,
    horizon: 32,
    frames: 6,
    elites: 4,
  });
  assert.equal(standard.horizon, 64, "input settings must remain unchanged");
});

test("invalid preset budgets cannot reach the native engine", () => {
  for (const controller_defaults of [
    { frames: 0 },
    { horizon: 2.5 },
    { elites: -1 },
  ])
    assert.throws(() =>
      presetControllerSettings({ controller_defaults }, standard),
    );
});
