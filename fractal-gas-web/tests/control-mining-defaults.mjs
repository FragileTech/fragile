// End-to-end gameplay acceptance; kept separate from the quick unit suite.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { createController } from "../web/lab/controllers/index.js";
import { presetControllerSettings } from "../web/lab/preset-settings.js";
import { configureRocks, rockOptions } from "../web/lab/rock-scene.js";
import {
  configureVehicleCount,
  vehicleCount,
} from "../web/lab/vehicle-scene.js";

const module = await loadNative(false);
const seeds = (process.env.CONTROL_MINING_SEEDS || "7").split(",").map(Number);
for (const name of ["harvest", "mining"]) {
  let scene = JSON.parse(
    await readFile(
      new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
    ),
  );
  scene = configureVehicleCount(scene, vehicleCount(scene));
  scene = configureRocks(scene, rockOptions(scene));
  assert.equal(scene.keep_delivered_rocks, true);
  const settings = presetControllerSettings(scene, {
    algorithm: "wave-jump",
    walkers: 128,
    horizon: 64,
    frames: 12,
    elites: 0,
    noise: 0.2,
    inertial: true,
    consensus_prefix: true,
    recording: 1,
    distance_coef: 1,
    reward_coef: 1,
  });
  for (const seed of seeds) {
    assert.ok(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff);
    const world = new NativeEngine(module, scene);
    const strategy = createController(module, scene, settings);
    const stats = {
      scene: name,
      seed,
      frames: 0,
      decisions: 0,
      catches: 0,
      initialAttachments: 0,
      attachedFrames: 0,
      deliveries: 0,
      releasedRocks: 0,
      firstCatch: null,
    };
    const started = performance.now();
    try {
      world.reset(seed);
      const targets = (words) =>
        scene.tethers.map((_, i) => words[world.info[7] + 2 * i]);
      let previous = targets(new Uint32Array(world.states().buffer));
      stats.initialAttachments = previous.filter((body) => body > 0).length;
      const cargo = scene.bodies.flatMap((body, i) => (body.cargo ? [i] : []));
      let previousFlags = cargo.map(
        (i) => new Uint32Array(world.states().buffer)[world.info[5] + i],
      );
      while (stats.frames < 600 && stats.decisions < 120) {
        strategy.controller.begin(
          world.snapshot(),
          (seed + stats.decisions) >>> 0,
        );
        while (!strategy.controller.advance()) {}
        const decision = strategy.controller.result();
        stats.decisions++;
        for (const edge of decision.trajectory) {
          for (
            let frame = 0;
            frame < edge.frames && stats.frames < 600;
            frame++
          ) {
            world.step(edge.action, 1);
            stats.frames++;
            const rows = world.states(),
              words = new Uint32Array(rows.buffer);
            assert.ok(
              rows.subarray(8, 8 + 6 * world.bodies).every(Number.isFinite),
            );
            assert.equal(
              words[7],
              0,
              `${name} terminated at frame ${stats.frames}`,
            );
            const attached = targets(words);
            const flags = cargo.map((i) => words[world.info[5] + i]);
            if (words[4] > stats.deliveries) {
              const delivered = cargo.filter(
                (_, i) => flags[i] === 3 && previousFlags[i] !== 3,
              );
              assert.equal(
                delivered.length,
                words[4] - stats.deliveries,
                "every delivery must retain its active body",
              );
              for (const body of delivered)
                assert.ok(
                  !attached.includes(body + 1),
                  "delivery must detach every towing hook",
                );
            }
            stats.releasedRocks += flags.filter(
              (flag, i) => flag === 1 && previousFlags[i] === 3,
            ).length;
            previousFlags = flags;
            const catches = attached.filter(
              (body, i) => body > 0 && previous[i] === 0,
            ).length;
            stats.catches += catches;
            if (catches && stats.firstCatch == null)
              stats.firstCatch = stats.frames;
            stats.attachedFrames += Number(attached.some((body) => body > 0));
            stats.deliveries = words[4];
            previous = attached;
          }
          if (stats.frames >= 600) break;
        }
        if (stats.decisions % 10 === 0)
          console.log(JSON.stringify({ progress: stats }));
      }
      assert.equal(
        stats.frames,
        600,
        "default search must make useful execution progress",
      );
      // Collaborative mining starts with its only rock already attached; a
      // successful retained delivery must not require a respawn/new catch.
      assert.ok(
        stats.catches + stats.initialAttachments >= 1,
        `${name} never hooked a rock`,
      );
      assert.ok(
        stats.attachedFrames >= 120,
        `${name} failed to keep a rock attached`,
      );
      assert.ok(stats.deliveries >= 1, `${name} never delivered a rock`);
      assert.ok(
        stats.releasedRocks >= 1,
        `${name} left its delivered rocks locked in the base`,
      );
      console.log(
        JSON.stringify({
          ...stats,
          elapsedMs: Math.round(performance.now() - started),
        }),
      );
    } finally {
      strategy.dispose();
      world.dispose();
    }
  }
}
