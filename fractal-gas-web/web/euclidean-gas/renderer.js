import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/addons/controls/OrbitControls.js";
import { alive } from "./config.js";

const LOW = new THREE.Color("#7ef5df");
const HIGH = new THREE.Color("#ff729b");
export class PopulationRenderer {
  constructor(container, onSelect) {
    this.container = container;
    this.onSelect = onSelect;
    this.frame = null;
    this.axes = [0, 1];
    this.selected = 0;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#090d16");
    this.camera = new THREE.OrthographicCamera(-1.3, 1.3, 1.3, -1.3, 0.01, 100);
    this.camera.position.set(0, 0, 10);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Walker positions over the objective landscape",
    );
    container.prepend(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableRotate = false;
    this.controls.enableDamping = false;
    this.controls.minZoom = 0.2;
    this.controls.maxZoom = 50;
    this.controls.addEventListener("change", () => this.render());
    const pointCanvas = document.createElement("canvas");
    pointCanvas.width = pointCanvas.height = 32;
    const ctx = pointCanvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(16, 16, 14, 0, Math.PI * 2);
    ctx.fill();
    const texture = new THREE.CanvasTexture(pointCanvas);
    this.points = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({
        size: 5,
        vertexColors: true,
        map: texture,
        transparent: true,
        alphaTest: 0.5,
        depthTest: false,
        sizeAttenuation: false,
      }),
    );
    this.points.renderOrder = 3;
    this.scene.add(this.points);
    this.marker = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({
        color: "white",
        size: 10,
        map: texture,
        transparent: true,
        alphaTest: 0.5,
        depthTest: false,
        sizeAttenuation: false,
      }),
    );
    this.marker.renderOrder = 4;
    this.scene.add(this.marker);
    const border = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1, -1, 0),
      new THREE.Vector3(1, -1, 0),
      new THREE.Vector3(1, 1, 0),
      new THREE.Vector3(-1, 1, 0),
      new THREE.Vector3(-1, -1, 0),
    ]);
    this.scene.add(
      new THREE.Line(
        border,
        new THREE.LineBasicMaterial({
          color: "#51415f",
          transparent: true,
          opacity: 0.8,
        }),
      ),
    );
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    let down = null;
    this.renderer.domElement.addEventListener("pointerdown", (e) => {
      down = [e.clientX, e.clientY];
    });
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4)
        this.selectAt(e.clientX, e.clientY);
      down = null;
    });
    this.resize();
  }
  resize() {
    const width = this.container.clientWidth,
      height = this.container.clientHeight;
    this.renderer.setSize(width, height, false);
    const aspect = width / Math.max(height, 1),
      span = 1.18;
    this.camera.left = -span * aspect;
    this.camera.right = span * aspect;
    this.camera.top = span;
    this.camera.bottom = -span;
    this.camera.updateProjectionMatrix();
    this.render();
  }
  fit() {
    this.controls.reset();
    this.camera.zoom = 1;
    const positions = this.points.geometry.attributes.position;
    if (positions?.count) {
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i),
          y = positions.getY(i);
        if (Math.abs(x) > 1e8 || Math.abs(y) > 1e8) continue;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
      if (Number.isFinite(minX)) {
        const x = (minX + maxX) / 2,
          y = (minY + maxY) / 2;
        this.camera.position.set(x, y, 10);
        this.controls.target.set(x, y, 0);
        const ratio = Math.max(
          (maxX - minX) / (this.camera.right - this.camera.left),
          (maxY - minY) / (this.camera.top - this.camera.bottom),
          0.02,
        );
        this.camera.zoom = Math.min(50, 1 / (ratio * 1.35));
      }
    }
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.render();
  }
  project(x, low, high) {
    return (2 * (x - low)) / (high - low) - 1;
  }
  update(frame, bounds, axes, selected, includeTruncated = false) {
    this.frame = frame;
    this.bounds = bounds;
    this.axes = axes;
    this.selected = selected;
    const field = frame.population.observations.fields.positions;
    if (!field) return;
    const n = field.rows,
      d = field.item_shape.reduce((a, b) => a * b, 1);
    const positions = new Float32Array(n * 3),
      colors = new Float32Array(n * 3);
    const rewards = frame.population.rewards.raw;
    const finite = Array.from(rewards).filter(Number.isFinite);
    let min = Infinity,
      max = -Infinity;
    for (const value of finite) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    const color = new THREE.Color();
    for (let i = 0; i < n; i++) {
      positions[i * 3] = this.project(field.values[i * d + axes[0]], ...bounds);
      positions[i * 3 + 1] = this.project(
        field.values[i * d + axes[1]],
        ...bounds,
      );
      positions[i * 3 + 2] = 0.02;
      if (
        !Number.isFinite(positions[i * 3]) ||
        !Number.isFinite(positions[i * 3 + 1])
      )
        positions[i * 3] = positions[i * 3 + 1] = 1e10;
      const t =
        max > min
          ? Math.max(0, Math.min(1, (rewards[i] - min) / (max - min)))
          : 0.5;
      color.copy(LOW).lerp(HIGH, Number.isFinite(t) ? t : 0);
      if (!alive(frame.population.validity[i], includeTruncated))
        color.set("#555267");
      color.toArray(colors, i * 3);
    }
    this.points.geometry.dispose();
    this.points.geometry = new THREE.BufferGeometry();
    this.points.geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );
    this.points.geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colors, 3),
    );
    this.marker.geometry.dispose();
    this.marker.geometry = new THREE.BufferGeometry();
    this.marker.geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        positions.slice(selected * 3, selected * 3 + 3),
        3,
      ),
    );
    this.render();
  }
  landscape(data) {
    if (this.surface) {
      this.scene.remove(this.surface);
      this.surface.geometry.dispose();
      this.surface.material.dispose();
    }
    const n = data.resolution;
    const geometry = new THREE.PlaneGeometry(2, 2, n - 1, n - 1);
    let min = Infinity,
      max = -Infinity;
    for (const v of data.values) {
      if (Number.isFinite(v)) {
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    }
    const colors = new Float32Array(n * n * 3);
    const color = new THREE.Color();
    const low = new THREE.Color("#101c29"),
      middle = new THREE.Color("#234b50"),
      high = new THREE.Color("#635148");
    for (let row = 0; row < n; row++)
      for (let col = 0; col < n; col++) {
        const v = data.values[(n - 1 - row) * n + col];
        const t =
          max > min ? Math.max(0, Math.min(1, (v - min) / (max - min))) : 0;
        if (t < 0.65) color.copy(low).lerp(middle, t / 0.65);
        else color.copy(middle).lerp(high, (t - 0.65) / 0.35);
        color.toArray(colors, (row * n + col) * 3);
      }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.surface = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
      }),
    );
    this.surface.position.z = -0.1;
    this.surface.visible = this.showLandscape !== false;
    this.scene.add(this.surface);
    this.render();
  }
  setLandscapeVisible(visible) {
    this.showLandscape = visible;
    if (this.surface) this.surface.visible = visible;
    this.render();
  }
  selectAt(x, y) {
    if (!this.frame) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const attribute = this.points.geometry.attributes.position;
    const point = new THREE.Vector3();
    let best = 14,
      selected = null;
    for (let i = 0; i < attribute.count; i++) {
      point.fromBufferAttribute(attribute, i).project(this.camera);
      const px = rect.left + ((point.x + 1) * rect.width) / 2,
        py = rect.top + ((1 - point.y) * rect.height) / 2;
      const distance = Math.hypot(px - x, py - y);
      if (distance < best) {
        best = distance;
        selected = i;
      }
    }
    if (selected !== null) this.onSelect(selected);
  }
  render() {
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}

export function drawConvergence(canvas, records, axis) {
  const dpr = Math.min(devicePixelRatio, 2),
    width = canvas.clientWidth,
    height = canvas.clientHeight;
  canvas.width = Math.max(1, width * dpr);
  canvas.height = Math.max(1, height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const left = 59,
    right = 12,
    top = 8,
    bottom = 23,
    w = width - left - right,
    h = height - top - bottom;
  ctx.font = "9px Consolas, monospace";
  const points = records.filter(
    (r) => Number.isFinite(r.best) && Number.isFinite(r.mean),
  );
  if (!points.length) return;
  let min = Infinity,
    max = -Infinity;
  for (const p of points) {
    min = Math.min(min, p.best, p.mean);
    max = Math.max(max, p.best, p.mean);
  }
  if (max === min) {
    max += 1;
    min -= 1;
  }
  const padding = (max - min) * 0.08;
  min -= padding;
  max += padding;
  const xKey = axis === "evaluations" ? "evaluations" : "step",
    xMax = Math.max(1, points.at(-1)[xKey]);
  const x = (value) => left + (w * value) / xMax,
    y = (value) => top + h * (1 - (value - min) / (max - min));
  for (let i = 0; i < 4; i++) {
    const v = min + ((max - min) * i) / 3,
      py = y(v);
    ctx.strokeStyle = "#312938";
    ctx.beginPath();
    ctx.moveTo(left, py);
    ctx.lineTo(width - right, py);
    ctx.stroke();
    ctx.fillStyle = "#b5a6c0";
    ctx.textAlign = "right";
    ctx.fillText(
      Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(2),
      left - 8,
      py + 3,
    );
  }
  for (const [key, color] of [
    ["mean", "#d0a9e2"],
    ["best", "#7ef5df"],
  ]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    points.forEach((p, i) =>
      i ? ctx.lineTo(x(p[xKey]), y(p[key])) : ctx.moveTo(x(p[xKey]), y(p[key])),
    );
    ctx.stroke();
    if (points.length === 1) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x(points[0][xKey]), y(points[0][key]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.fillStyle = "#b5a6c0";
  ctx.textAlign = "left";
  ctx.fillText("0", left, height - 5);
  ctx.textAlign = "right";
  ctx.fillText(String(xMax), width - right, height - 5);
}
