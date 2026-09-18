import assert from "node:assert/strict";
import test from "node:test";
import { OrthographicCamera, Vector3 } from "three";
import { arenaBounds } from "../web/lab/camera-fit.js";
import {
  dragOrbit,
  gizmoAxes,
  lerpOrbit,
  orbitBasis,
  orbitPosition,
  pitchLimits,
  presetOrbit,
  snapOrbit,
} from "../web/lab/camera-orbit.js";

const near = (a, b) => assert(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("presets retain the original camera positions", () => {
  const arena = arenaBounds({ size: [64, 44] });
  for (const [top, side, expected] of [
    [false, false, [32, -23, 60]],
    [true, false, [32, 21.999, 90]],
    [false, true, [32, -90, 22]],
  ]) {
    const pose = orbitPosition(
      presetOrbit(top, side),
      arena,
      arena.center,
      side,
    );
    pose.position.forEach((value, i) => near(value, expected[i]));
  }
});

test("orbit supports full turns and keeps planar and flight views upright", () => {
  for (const side of [false, true]) {
    const start = presetOrbit(false, side);
    const rotated = dragOrbit(start, (2 * Math.PI) / 0.008, 0, side);
    near(rotated.yaw, start.yaw);
    near(rotated.pitch, start.pitch);
    near(
      dragOrbit(start, 0, 10000, side).pitch,
      ((side ? 80 : 89.9) * Math.PI) / 180,
    );
    near(
      dragOrbit(start, 0, -10000, side).pitch,
      ((side ? -80 : 5) * Math.PI) / 180,
    );
    assert.deepEqual(start, presetOrbit(false, side));
  }
});

test("rotating large arenas and off-center views never clips their depth", () => {
  const arena = arenaBounds({ size: [1200, 800] });
  for (const side of [false, true]) {
    for (const center of [arena.center, [-800, 1200]]) {
      for (const yaw of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
        for (const pitch of [0.1, 0.8, 1.4]) {
          const pose = orbitPosition(
            { yaw, pitch, distance: 75 },
            arena,
            center,
            side,
          );
          const camera = new OrthographicCamera(
            -2000,
            2000,
            2000,
            -2000,
            0.1,
            pose.far,
          );
          camera.up.set(0, 0, 1);
          camera.position.set(...pose.position);
          camera.lookAt(...pose.target);
          camera.updateMatrixWorld(true);
          for (const x of [arena.minX, arena.maxX]) {
            for (const y of [arena.minY, arena.maxY]) {
              for (const height of [-16, 16]) {
                const point = new Vector3(
                  x,
                  side ? height : y,
                  side ? y : height,
                ).project(camera);
                assert(point.z > -1 && point.z < 1, `Clipped depth ${point.z}`);
              }
            }
          }
        }
      }
    }
  }
});

test("the gizmo basis matches the rendered camera", () => {
  const arena = arenaBounds({ size: [64, 44] });
  for (const orbit of [
    presetOrbit(false, false),
    { yaw: 2.1, pitch: 0.4, distance: 75 },
    { yaw: -0.7, pitch: -1.1, distance: 90 },
  ]) {
    const pose = orbitPosition(orbit, arena, arena.center, false);
    const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, pose.far);
    camera.up.set(0, 0, 1);
    camera.position.set(...pose.position);
    camera.lookAt(...pose.target);
    camera.updateMatrixWorld(true);
    const basis = orbitBasis(orbit);
    for (const [column, expected] of [
      basis.right,
      basis.up,
      basis.toCamera,
    ].entries())
      new Vector3()
        .setFromMatrixColumn(camera.matrixWorld, column)
        .toArray()
        .forEach((value, i) => near(value, expected[i]));
  }
});

test("gizmo handles follow the simulation axes in ground and flight views", () => {
  const handle = (orbit, side, axis, sign) =>
    gizmoAxes(orbit, side).find((h) => h.axis === axis && h.sign === sign);
  const ground = { yaw: 0, pitch: 0.5, distance: 75 };
  near(handle(ground, false, "x", 1).x, 1);
  near(handle(ground, false, "x", -1).x, -1);
  near(handle(ground, false, "z", 1).y, Math.cos(0.5));
  assert(handle(ground, false, "y", 1).depth < 0);
  const flight = presetOrbit(false, true);
  near(handle(flight, true, "x", 1).x, 1);
  near(handle(flight, true, "y", 1).y, 1);
  near(handle(flight, true, "z", 1).depth, 1);
  assert.equal(gizmoAxes(ground, false).length, 6);
  for (const side of [false, true])
    for (const h of gizmoAxes(ground, side))
      assert.equal(h.enabled, side || h.axis !== "z" || h.sign > 0);
});

test("axis snaps respect the pitch limits of the drag controls", () => {
  const start = { yaw: 1.3, pitch: 0.7, distance: 75 };
  const [low, high] = pitchLimits(false);
  for (const [axis, sign, yaw, pitch] of [
    ["x", 1, Math.PI / 2, low],
    ["x", -1, -Math.PI / 2, low],
    ["y", 1, Math.PI, low],
    ["y", -1, 0, low],
    ["z", 1, 0, high],
  ]) {
    const snapped = snapOrbit(start, axis, sign, false);
    near(Math.cos(snapped.yaw - yaw), 1);
    near(snapped.pitch, pitch);
    near(snapped.distance, 75);
    near(dragOrbit(snapped, 0, 0, false).pitch, snapped.pitch);
  }
  const [sideLow, sideHigh] = pitchLimits(true);
  near(snapOrbit(start, "y", 1, true).pitch, sideHigh);
  near(snapOrbit(start, "y", -1, true).pitch, sideLow);
  assert.deepEqual(snapOrbit(start, "z", 1, true), {
    ...presetOrbit(false, true),
    distance: 75,
  });
  near(Math.cos(snapOrbit(start, "z", -1, true).yaw - Math.PI), 1);
});

test("snap turns take the short way around", () => {
  const a = { yaw: 3, pitch: 0.2, distance: 75 };
  const b = { yaw: -3, pitch: 1, distance: 75 };
  const middle = lerpOrbit(a, b, 0.5);
  near(Math.abs(middle.yaw), Math.PI);
  near(middle.pitch, 0.6);
  assert.deepEqual(lerpOrbit(a, b, 1), b);
  near(lerpOrbit(a, b, 0).yaw, a.yaw);
});
