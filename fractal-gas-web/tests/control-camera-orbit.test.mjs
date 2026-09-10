import assert from "node:assert/strict";
import test from "node:test";
import { OrthographicCamera, Vector3 } from "three";
import { arenaBounds } from "../web/lab/camera-fit.js";
import {
  dragOrbit,
  orbitPosition,
  presetOrbit,
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
