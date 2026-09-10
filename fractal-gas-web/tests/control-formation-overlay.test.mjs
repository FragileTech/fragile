import test from "node:test";
import assert from "node:assert/strict";
import * as T from "../web/lab/vendor/three.module.js";
import {
  formationPairs,
  formationPairScore,
} from "../web/lab/formation-pairs.js";
import {
  FormationOverlay,
  formationColor,
  FORMATION_GRADIENT,
} from "../web/lab/visuals/formation-overlay.js";
import { palette } from "../web/lab/visuals/primitives.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";

const module = await loadNative();
const points = [
  [10, 10],
  [16, 10],
  [10, 18],
  [16, 18],
];
const fixture = (count = 3) => ({
  task: "tandem",
  size: [100, 100],
  environment: { flight: false },
  formation_distance: 5,
  bodies: points
    .slice(0, count)
    .map((position) => ({ controlled: true, position })),
});
function close(a, b, tolerance = 1e-6) {
  assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
}

test("all scored pairs connect actual body centres for 1, 2, 3 and 4 agents", () => {
  for (const count of [1, 2, 3, 4]) {
    const scene = fixture(count);
    const engine = new NativeEngine(module, scene);
    const overlay = new FormationOverlay(scene, undefined, engine.bodies);
    try {
      overlay.update(engine.states());
      assert.equal(overlay.pairs.length, (count * (count - 1)) / 2);
      assert.equal(overlay.group.visible, count > 1);
      if (count === 1) continue;
      assert.equal(
        overlay.group.children.length,
        1,
        "all pairs share one draw object",
      );
      assert.equal(overlay.lines.isLineSegments, true);
      const positions = overlay.geometry.getAttribute("position");
      const distances = overlay.geometry.getAttribute("lineDistance");
      const seen = new Set();
      for (const [i, { a, b }] of overlay.pairs.entries()) {
        assert.ok(a < b);
        assert.ok(!seen.has(`${a}:${b}`));
        seen.add(`${a}:${b}`);
        assert.deepEqual(
          [positions.getX(2 * i), positions.getY(2 * i)],
          points[a],
        );
        assert.deepEqual(
          [positions.getX(2 * i + 1), positions.getY(2 * i + 1)],
          points[b],
        );
        assert.equal(distances.getX(2 * i), 0);
        close(
          distances.getX(2 * i + 1),
          Math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]),
        );
      }
    } finally {
      overlay.dispose();
      engine.dispose();
    }
  }
});

test("inherited controls, reversed overrides and fallbacks agree with native reward", () => {
  const scene = {
    ...fixture(),
    formation_distance: 10,
    formation_pairs: [
      { a: 2, b: 0, distance: 5 },
      { a: 0, b: 3, distance: 8 },
    ],
    agent_types: {
      base: { physics: { controlled: true } },
      wing: { extends: "base" },
    },
    bodies: [
      { agent_type: "wing", position: points[0] },
      { position: [80, 80], cargo: true },
      { agent_type: "wing", position: points[1] },
      { agent_type: "wing", position: points[2] },
    ],
  };
  assert.deepEqual(formationPairs(scene), [
    { a: 0, b: 2, target: 5 },
    { a: 0, b: 3, target: 8 },
    { a: 2, b: 3, target: 10 },
  ]);
  for (const weight of [0, 1, 3]) {
    const config = { ...scene, rewards: { formation: weight } };
    const engine = new NativeEngine(module, config);
    const overlay = new FormationOverlay(config, undefined, engine.bodies);
    try {
      engine.step(engine.neutralAction(), 1);
      overlay.update(engine.states());
      assert.equal(overlay.group.visible, true);
      close(overlay.scores[0], 5 / 6);
      close(overlay.scores[1], 1);
      close(overlay.scores[2], 1);
      close(
        engine.results()[0],
        weight * overlay.scores.reduce((product, score) => product * score, 1),
      );
      const colors = overlay.geometry.getAttribute("color");
      const expected = formationColor(5 / 6);
      close(colors.getX(0), expected.r);
      close(colors.getY(0), expected.g);
      close(colors.getZ(0), expected.b);
      assert.notEqual(
        colors.getX(0),
        colors.getX(2),
        "pair colors are independent",
      );
    } finally {
      overlay.dispose();
      engine.dispose();
    }
  }
  assert.equal(
    formationPairs({
      ...scene,
      formation_distance: undefined,
      formation_pairs: [],
    })[0].target,
    3,
  );
});

test("pair colors are continuous, symmetric around the target, and match legend anchors", () => {
  for (const [score, color] of [
    [0, palette.rose],
    [0.5, palette.gold],
    [1, palette.green],
  ])
    assert.equal(formationColor(score).getHex(), color);
  assert.equal(
    formationColor(formationPairScore(5, 4)).getHex(),
    formationColor(formationPairScore(5, 6)).getHex(),
  );
  assert.equal(
    formationColor(formationPairScore(5, 5)).getHex(),
    palette.green,
  );
  close(formationPairScore(5, 6), 5 / 6);
  const below = formationColor(0.5 - 1e-6),
    above = formationColor(0.5 + 1e-6);
  for (const channel of ["r", "g", "b"])
    close(below[channel], above[channel], 1e-5);
  assert.match(FORMATION_GRADIENT, /#ff538b, #ffce70, #7bffc1/);
});

test("frame updates and visibility changes reuse buffers; disposal releases resources", () => {
  const scene = fixture();
  const engine = new NativeEngine(module, scene);
  const overlay = new FormationOverlay(scene, undefined, engine.bodies);
  const parent = new T.Group();
  parent.add(overlay.group);
  const geometry = overlay.geometry,
    material = overlay.material;
  const attributes = { ...geometry.attributes };
  let geometriesDisposed = 0,
    materialsDisposed = 0;
  geometry.addEventListener("dispose", () => ++geometriesDisposed);
  material.addEventListener("dispose", () => ++materialsDisposed);
  try {
    const state = engine.states();
    for (let i = 0; i < 100; ++i) {
      state[8 + 2] = 10 + i * 0.1;
      overlay.update(state);
      overlay.setVisible(i % 2 === 0);
      assert.equal(overlay.geometry, geometry);
      assert.equal(overlay.material, material);
      for (const key of Object.keys(attributes))
        assert.equal(geometry.attributes[key], attributes[key]);
    }
    close(geometry.attributes.position.getX(3), state[10]);
    assert.equal(geometriesDisposed + materialsDisposed, 0);
    overlay.setVisible(true);
    assert.equal(overlay.group.visible, true);
  } finally {
    overlay.dispose();
    engine.dispose();
  }
  assert.equal(geometriesDisposed, 1);
  assert.equal(materialsDisposed, 1);
  assert.equal(parent.children.length, 0);
});

test("non-formation scenes and fewer than two controls have no visible formation overlay", () => {
  for (const scene of [
    { ...fixture(), task: "navigation" },
    { ...fixture(), task: "harvest" },
    { ...fixture(), bodies: [{ position: [10, 10] }] },
    fixture(1),
  ]) {
    const overlay = new FormationOverlay(scene, undefined, scene.bodies.length);
    overlay.setVisible(true);
    assert.equal(overlay.group.visible, false);
    assert.equal(overlay.geometry, undefined);
    overlay.dispose();
  }
});
