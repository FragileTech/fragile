import * as T from "../vendor/three.module.js";

const LIMIT = 16;
const WIDTH = 164;
const HEIGHT = 70;
const format = new Intl.NumberFormat("en", { maximumSignificantDigits: 6 });
const amountText = (value) => format.format(Math.max(0, Number(value) || 0));

// Screen-space labels share one fixed DOM pool and add no WebGL draw calls.
export class CargoReadout {
  constructor(canvas) {
    const doc = canvas.ownerDocument;
    if (!doc.getElementById("lab-cargo-readout-style")) {
      const link = doc.createElement("link");
      link.id = "lab-cargo-readout-style";
      link.rel = "stylesheet";
      link.href = new URL("./cargo-readout.css", import.meta.url).href;
      doc.head.append(link);
    }
    this.root = doc.createElement("div");
    this.root.className = "cargo-readouts";
    this.root.setAttribute("aria-label", "Vehicle resources");
    canvas.parentElement.append(this.root);
    this.point = new T.Vector3();
    this.labels = Array.from({ length: LIMIT }, () => {
      const node = doc.createElement("div");
      node.className = "cargo-readout";
      node.hidden = true;
      const heading = doc.createElement("div");
      heading.className = "cargo-readout-heading";
      const name = doc.createElement("span");
      name.className = "cargo-readout-name";
      const amount = doc.createElement("strong");
      heading.append(name, amount);
      const bar = doc.createElement("div");
      bar.className = "cargo-readout-bar";
      bar.setAttribute("aria-hidden", "true");
      const fill = doc.createElement("span");
      bar.append(fill);
      const status = doc.createElement("div");
      status.className = "cargo-readout-status";
      const total = doc.createElement("div");
      total.className = "cargo-readout-total";
      node.append(heading, bar, status, total);
      this.root.append(node);
      return { node, name, amount, fill, status, total, x: 0, y: 0 };
    });
  }
  resize(width, height) {
    this.width = width;
    this.height = height;
  }
  clear() {
    for (const label of this.labels) label.node.hidden = true;
  }
  update(entries, layer, camera, coordinateFrame, selected, followed, style) {
    if (!entries?.length || !this.width || !this.height) {
      this.clear();
      return;
    }
    if (this.root.dataset.style !== style) this.root.dataset.style = style;
    coordinateFrame.updateWorldMatrix(true, false);
    camera.updateMatrixWorld();
    this.count = 0;
    // Selection wins overlap resolution, followed by the camera target.
    for (const entry of entries)
      if (entry.body === selected)
        this.place(entry, layer, camera, coordinateFrame, true);
    if (followed !== selected)
      for (const entry of entries)
        if (entry.body === followed)
          this.place(entry, layer, camera, coordinateFrame, true);
    for (const entry of entries) {
      if (this.count === LIMIT) break;
      if (entry.body !== selected && entry.body !== followed)
        this.place(entry, layer, camera, coordinateFrame, false);
    }
    for (let i = this.count; i < LIMIT; i++) this.labels[i].node.hidden = true;
  }
  place(entry, layer, camera, coordinateFrame, priority) {
    const body = entry.body;
    if (
      this.count === LIMIT ||
      entry.visible === false ||
      entry.enabled === false ||
      layer.active?.[body] === false ||
      layer.inView?.[body] === false ||
      !entry.anchor ||
      !(entry.capacity > 0)
    )
      return;
    this.point
      .copy(entry.anchor)
      .applyMatrix4(coordinateFrame.matrixWorld)
      .project(camera);
    const point = this.point;
    if (
      !Number.isFinite(point.x + point.y + point.z) ||
      Math.abs(point.x) > 1 ||
      Math.abs(point.y) > 1 ||
      Math.abs(point.z) > 1
    )
      return;
    const x = Math.round(
      Math.max(
        4,
        Math.min(
          this.width - WIDTH - 4,
          (point.x * 0.5 + 0.5) * this.width - WIDTH / 2,
        ),
      ),
    );
    const y = Math.round(
      Math.max(
        4,
        Math.min(
          this.height - HEIGHT - 4,
          (-point.y * 0.5 + 0.5) * this.height - HEIGHT - 12,
        ),
      ),
    );
    for (let i = 0; i < this.count; i++) {
      const other = this.labels[i];
      if (
        Math.abs(other.x - x) < WIDTH + 8 &&
        Math.abs(other.y - y) < HEIGHT + 6
      )
        return;
    }
    const label = this.labels[this.count++];
    if (label.body !== body) {
      this.text(label.name, `CARGO / ${body + 1}`);
      label.body = body;
    }
    if (label.held !== entry.amount || label.capacity !== entry.capacity) {
      this.text(
        label.amount,
        `${amountText(entry.amount)} / ${amountText(entry.capacity)}`,
      );
      const fraction = Math.min(
        1,
        Math.max(0, entry.amount / entry.capacity || 0),
      );
      label.fill.style.transform = `scaleX(${fraction})`;
      label.held = entry.amount;
      label.capacity = entry.capacity;
    }
    if (label.state !== entry.status || label.loading !== entry.loadingAmount) {
      const loading =
        entry.status === "Loading" && entry.loadingAmount > 0
          ? ` +${amountText(entry.loadingAmount)}`
          : "";
      this.text(label.status, `${entry.status || "Carrying"}${loading}`);
      label.state = entry.status;
      label.loading = entry.loadingAmount;
    }
    const collected = entry.collected ?? entry.amount + (entry.delivered || 0);
    if (label.collected !== collected || label.delivered !== entry.delivered) {
      this.text(
        label.total,
        `Picked ${amountText(collected)} · Delivered ${amountText(entry.delivered)}`,
      );
      label.collected = collected;
      label.delivered = entry.delivered;
    }
    if (label.node.dataset.status !== entry.status)
      label.node.dataset.status = entry.status;
    const focus = priority ? "true" : "false";
    if (label.node.dataset.priority !== focus)
      label.node.dataset.priority = focus;
    if (label.x !== x || label.y !== y || label.node.hidden)
      label.node.style.transform = `translate(${x}px, ${y}px)`;
    label.x = x;
    label.y = y;
    label.node.hidden = false;
  }
  text(node, value) {
    if (node.textContent !== value) node.textContent = value;
  }
  dispose() {
    this.root.remove();
  }
}
