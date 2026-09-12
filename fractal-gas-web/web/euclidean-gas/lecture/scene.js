import { escapeXML as esc, COLORS } from "./plots.js";

// The scene is a view of immutable numerical output: camera, layers and selection
// never alter the scientific configuration or advance the worker.
export class LectureScene {
  constructor(host) {
    this.host = host;
    this.yaw = -0.55;
    this.pitch = 0.42;
    this.hidden = new Set();
    this.selected = null;
    this.cut = 1;
    this.trace = "incident";
    host.addEventListener("input", (e) => {
      if (e.target.dataset.layer) {
        e.target.checked
          ? this.hidden.delete(e.target.dataset.layer)
          : this.hidden.add(e.target.dataset.layer);
      } else if (e.target.name === "time-cut")
        this.cut = Number(e.target.value);
      else if (e.target.name === "event")
        this.selected = e.target.value || null;
      else if (e.target.name === "vertical-scale")
        this.vertical = Number(e.target.value);
      else if (e.target.name === "trace-relation") this.trace = e.target.value;
      this.render();
    });
    host.addEventListener("click", (e) => {
      const node = e.target.closest("[data-event]");
      if (node) {
        this.selected = node.dataset.event;
        this.render();
      }
      if (e.target.dataset.sceneExport !== undefined) {
        const url = URL.createObjectURL(
          new Blob([this.svg()], { type: "image/svg+xml" }),
        );
        const a = document.createElement("a");
        a.href = url;
        a.download = "fractal-set-scene.svg";
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    });
    host.addEventListener("pointerdown", (e) => {
      if (!e.target.closest("svg")) return;
      this.drag = [e.clientX, e.clientY];
      host.setPointerCapture(e.pointerId);
    });
    host.addEventListener("pointermove", (e) => {
      if (!this.drag) return;
      this.yaw += (e.clientX - this.drag[0]) * 0.008;
      this.pitch = Math.max(
        -1.4,
        Math.min(1.4, this.pitch + (e.clientY - this.drag[1]) * 0.008),
      );
      this.drag = [e.clientX, e.clientY];
      const svg = host.querySelector(".scene-viewport");
      if (svg) svg.innerHTML = this.svg();
    });
    host.addEventListener("pointerup", () => {
      this.drag = null;
    });
    host.addEventListener("keydown", (e) => {
      if (!e.target.closest(".scene-viewport")) return;
      if (e.key === "ArrowLeft") this.yaw -= 0.1;
      else if (e.key === "ArrowRight") this.yaw += 0.1;
      else if (e.key === "ArrowUp") this.pitch -= 0.1;
      else if (e.key === "ArrowDown") this.pitch += 0.1;
      else return;
      e.preventDefault();
      host.querySelector(".scene-viewport").innerHTML = this.svg();
    });
  }
  update(scene) {
    if (scene?.title !== this.scene?.title)
      this.vertical = scene?.verticalScale || 1;
    this.scene = scene;
    this.host.hidden = !scene;
    if (scene) this.render();
  }
  project(p) {
    const center = this.center,
      scale = this.scale;
    const x = (p[0] - center[0]) * scale,
      y = (p[1] - center[1]) * scale,
      z = (p[2] - center[2]) * scale * this.vertical;
    const u = Math.cos(this.yaw) * x - Math.sin(this.yaw) * y;
    const v = Math.sin(this.yaw) * x + Math.cos(this.yaw) * y;
    return [
      420 + u,
      250 + Math.sin(this.pitch) * v - Math.cos(this.pitch) * z,
      Math.cos(this.pitch) * v + Math.sin(this.pitch) * z,
    ];
  }
  svg() {
    const s = this.scene,
      nodes = s.nodes || [],
      faces = s.faces || [],
      edges = s.edges || [];
    const positions = [
      ...nodes.map((n) => n.position),
      ...faces.flatMap((f) => f.vertices),
    ];
    if (!positions.length) return "";
    const lo = [Infinity, Infinity, Infinity],
      hi = [-Infinity, -Infinity, -Infinity];
    for (const p of positions)
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], p[k]);
        hi[k] = Math.max(hi[k], p[k]);
      }
    this.center = lo.map((v, k) => (v + hi[k]) / 2);
    this.scale =
      330 /
      Math.max(
        1e-9,
        ...hi.map((v, k) => (v - lo[k]) * (k === 2 ? this.vertical : 1)),
      );
    const cutoff = lo[2] + this.cut * (hi[2] - lo[2]);
    const visible = (layer, p) =>
      !this.hidden.has(layer || "geometry") && p[2] <= cutoff + 1e-10;
    const byId = new Map(nodes.map((n) => [String(n.id), n]));
    const related = new Set(this.selected ? [this.selected] : []);
    const relation = (e) =>
      this.trace === "cst"
        ? /cst/i.test(e.layer)
        : ["ancestry", "persistence"].includes(e.layer);
    if (this.selected && this.trace !== "incident") {
      const queue = [this.selected];
      while (queue.length) {
        const id = queue.pop();
        for (const e of edges) {
          if (!relation(e)) continue;
          const source =
            this.trace === "ancestry" ? String(e.target) : String(e.source);
          const target =
            this.trace === "ancestry" ? String(e.source) : String(e.target);
          if (source === id && !related.has(target)) {
            related.add(target);
            queue.push(target);
          }
        }
      }
    }
    const shapes = [];
    const selectedOwner = nodes.find(
      (n) => String(n.id) === this.selected,
    )?.owner;
    for (const face of faces) {
      if (selectedOwner !== undefined && face.owner !== selectedOwner) continue;
      if (!face.vertices.every((p) => visible(face.layer, p))) continue;
      const ps = face.vertices.map((p) => this.project(p));
      shapes.push({
        z: ps.reduce((a, p) => a + p[2], 0) / ps.length,
        html: `<polygon points="${ps.map((p) => p.slice(0, 2).join(",")).join(" ")}" fill="${esc(face.color || COLORS[(face.owner || 0) % COLORS.length])}" fill-opacity=".2" stroke="${esc(face.color || "#718bac")}" stroke-width=".5"/>`,
      });
    }
    for (const edge of edges) {
      const a = byId.get(String(edge.source)),
        b = byId.get(String(edge.target));
      if (
        !a ||
        !b ||
        !visible(edge.layer, a.position) ||
        !visible(edge.layer, b.position)
      )
        continue;
      const p = this.project(a.position),
        q = this.project(b.position),
        active =
          this.trace === "incident"
            ? [String(a.id), String(b.id)].includes(this.selected)
            : related.has(String(a.id)) &&
              related.has(String(b.id)) &&
              relation(edge);
      shapes.push({
        z: (p[2] + q[2]) / 2,
        html: `<path d="M${p[0]},${p[1]} L${q[0]},${q[1]}" stroke="${esc(edge.color || "#72abd5")}" stroke-width="${active ? 3 : 1}" opacity="${this.selected && !active ? 0.18 : 0.7}" fill="none"/>`,
      });
    }
    for (const n of nodes) {
      if (!visible(n.layer, n.position)) continue;
      const p = this.project(n.position),
        selected = String(n.id) === this.selected;
      shapes.push({
        z: p[2],
        html: `<circle data-event="${esc(n.id)}" cx="${p[0]}" cy="${p[1]}" r="${selected ? 6 : 3}" fill="${esc(n.color || "#65c8d0")}" stroke="${selected ? "white" : "none"}"><title>${esc(n.label || n.id)}</title></circle>`,
      });
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 500" role="img" aria-label="${esc(s.title)}"><rect width="840" height="500" fill="#101e32"/>${shapes
      .sort((a, b) => a.z - b.z)
      .map((s) => s.html)
      .join(
        "",
      )}<text x="18" y="480" fill="#b7c6dc" font-size="13">x₁, x₂, time · drag or use arrow keys to rotate</text></svg>`;
  }
  render() {
    const s = this.scene;
    const layers = [
      ...new Set(
        [...(s.edges || []), ...(s.faces || []), ...(s.nodes || [])].map(
          (x) => x.layer || "geometry",
        ),
      ),
    ];
    const selected = (s.nodes || []).find(
      (n) => String(n.id) === this.selected,
    );
    this.host.innerHTML = `<div class="chart-heading"><h2>${esc(s.title)}</h2><button data-scene-export>Save scene SVG</button></div><div class="scene-controls">${layers.map((l) => `<label><input type="checkbox" data-layer="${esc(l)}" ${this.hidden.has(l) ? "" : "checked"}>${esc(l)}</label>`).join("")}<label>Time cut <input name="time-cut" type="range" min="0" max="1" step=".01" value="${this.cut}"></label><label>Vertical display scale ×${this.vertical} <input name="vertical-scale" type="range" min="1" max="100" step="1" value="${this.vertical}"></label><label>Trace <select name="trace-relation">${[
      ["incident", "Incident edges"],
      ["ancestry", "Material ancestors"],
      ["descendants", "Material descendants"],
      ["cst", "CST future"],
    ]
      .map(
        ([v, l]) =>
          `<option value="${v}" ${this.trace === v ? "selected" : ""}>${l}</option>`,
      )
      .join(
        "",
      )}</select></label><label>Event <select name="event"><option value="">All events</option>${(
      s.nodes || []
    )
      .slice(-1000)
      .map(
        (n) =>
          `<option value="${esc(n.id)}" ${String(n.id) === this.selected ? "selected" : ""}>${esc(n.label || n.id)}</option>`,
      )
      .join(
        "",
      )}</select></label></div><div class="scene-viewport" tabindex="0" aria-label="Rotate spacetime scene with arrow keys">${this.svg()}</div><p>${esc(selected?.detail || s.message || "Select an event to inspect its identity and incident edges.")}</p>`;
  }
}
