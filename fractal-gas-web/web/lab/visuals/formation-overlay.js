import * as T from "../vendor/three.module.js";
import { formationPairs, formationPairScore } from "../formation-pairs.js";
import { palette } from "./primitives.js";

const colors = [palette.rose, palette.gold, palette.green];
const channels = colors.map((hex) => [
  ((hex >> 16) & 255) / 255,
  ((hex >> 8) & 255) / 255,
  (hex & 255) / 255,
]);
export const FORMATION_GRADIENT = `linear-gradient(to right, ${colors
  .map((hex) => `#${hex.toString(16).padStart(6, "0")}`)
  .join(", ")})`;

export function formationColor(score, color = new T.Color()) {
  const value = Math.max(0, Math.min(1, score));
  const band = value < 0.5 ? 0 : 1;
  const t = 2 * value - band;
  const a = channels[band],
    b = channels[band + 1];
  // Interpolate in sRGB, matching the CSS legend, then convert for vertex colors.
  return color.setRGB(
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
    T.SRGBColorSpace,
  );
}

export class FormationOverlay {
  constructor(scene, controlled, bodyCount) {
    this.pairs = formationPairs(scene, controlled);
    this.bodyCount = bodyCount;
    this.group = new T.Group();
    this.group.name = "Formation pair distances";
    this.scores = new Float32Array(this.pairs.length);
    this.color = new T.Color();
    if (this.pairs.length) {
      this.geometry = new T.BufferGeometry();
      for (const [name, width] of [
        ["position", 3],
        ["color", 3],
        ["lineDistance", 1],
      ])
        this.geometry.setAttribute(
          name,
          new T.BufferAttribute(
            new Float32Array(this.pairs.length * 2 * width),
            width,
          ).setUsage(T.DynamicDrawUsage),
        );
      this.material = new T.LineDashedMaterial({
        vertexColors: true,
        dashSize: 0.3,
        gapSize: 0.2,
        transparent: true,
        opacity: 0.8,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
        fog: false,
      });
      this.lines = new T.LineSegments(this.geometry, this.material);
      // Endpoints move every update. GPU clipping avoids a stale bounding sphere.
      this.lines.frustumCulled = false;
      this.lines.renderOrder = 2;
      this.group.add(this.lines);
    }
    this.setVisible(true);
  }
  setVisible(enabled) {
    this.group.visible = !!enabled && this.pairs.length > 0;
  }
  update(state) {
    if (!this.geometry) return;
    const positions = this.geometry.getAttribute("position");
    const colors = this.geometry.getAttribute("color");
    const distances = this.geometry.getAttribute("lineDistance");
    for (let i = 0; i < this.pairs.length; ++i) {
      const { a, b, target } = this.pairs[i];
      const ax = state[8 + a],
        ay = state[8 + this.bodyCount + a];
      const bx = state[8 + b],
        by = state[8 + this.bodyCount + b];
      const actual = Math.hypot(bx - ax, by - ay);
      const score = formationPairScore(target, actual);
      this.scores[i] = score;
      positions.setXYZ(2 * i, ax, ay, 0.3);
      positions.setXYZ(2 * i + 1, bx, by, 0.3);
      formationColor(score, this.color);
      colors.setXYZ(2 * i, this.color.r, this.color.g, this.color.b);
      colors.setXYZ(2 * i + 1, this.color.r, this.color.g, this.color.b);
      // Each independent segment starts its own dash pattern.
      distances.setX(2 * i, 0);
      distances.setX(2 * i + 1, actual);
    }
    positions.needsUpdate = colors.needsUpdate = distances.needsUpdate = true;
  }
  dispose() {
    this.geometry?.dispose();
    this.material?.dispose();
    this.group.clear();
    this.group.removeFromParent();
  }
}
