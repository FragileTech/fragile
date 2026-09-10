import { hierarchy, tree } from "./vendor/d3-hierarchy/index.js";
import { color, metricFormat, METRICS } from "./analysis-data.js";

export class AnalysisCanvas {
  constructor(canvas, minimap, onSelect, onCollapse) {
    this.canvas = canvas;
    this.mini = minimap;
    this.onSelect = onSelect;
    this.onCollapse = onCollapse;
    this.camera = { x: 70, y: 180, k: 1 };
    this.rows = [];
    this.edges = [];
    this.positions = new Map();
    this.resize = new ResizeObserver(() => this.schedule());
    this.resize.observe(canvas);
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      canvas.focus();
      canvas.setPointerCapture(e.pointerId);
      drag = {
        x: e.clientX,
        y: e.clientY,
        cx: this.camera.x,
        cy: this.camera.y,
        moved: false,
      };
    });
    canvas.addEventListener("pointermove", (e) => {
      if (drag) {
        const dx = e.clientX - drag.x,
          dy = e.clientY - drag.y;
        drag.moved ||= Math.hypot(dx, dy) > 4;
        this.camera.x = drag.cx + dx;
        this.camera.y = drag.cy + dy;
        this.schedule();
      } else {
        const p = this.hit(e.offsetX, e.offsetY);
        canvas.style.cursor = p ? "pointer" : "grab";
        canvas.title = p
          ? this.label(p, true)
          : "Drag to pan · scroll to zoom · arrow keys to navigate";
      }
    });
    canvas.addEventListener("pointerup", (e) => {
      if (drag && !drag.moved) {
        const p = this.hit(e.offsetX, e.offsetY);
        if (p) this.onSelect(p);
      }
      drag = null;
    });
    canvas.addEventListener("pointercancel", () => {
      drag = null;
    });
    canvas.addEventListener("dblclick", (e) => {
      const p = this.hit(e.offsetX, e.offsetY);
      if (p) this.onCollapse(p.key);
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.zoom(Math.exp(-e.deltaY * 0.0015), e.offsetX, e.offsetY);
      },
      { passive: false },
    );
    canvas.addEventListener("keydown", (e) => {
      let p = this.positions.get(this.selectedKey),
        next;
      if (e.key === "ArrowLeft") next = this.positions.get(p?.parent);
      if (e.key === "ArrowRight")
        next = this.rows.find((n) => n.parent === p?.key);
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const siblings = this.rows.filter((n) => n.parent === p?.parent);
        const i = siblings.indexOf(p);
        next = siblings[i + (e.key === "ArrowDown" ? 1 : -1)];
      }
      if (e.key === "Home") next = this.positions.get("root");
      if (next) {
        this.onSelect(next);
        this.focus(next.key);
      }
      if (e.key === "+" || e.key === "=") this.zoom(1.3);
      if (e.key === "-") this.zoom(1 / 1.3);
      if (e.key === "f") this.focus(this.selectedKey);
      if ((e.key === "Enter" || e.key === " ") && p) this.onCollapse(p.key);
      if (
        [
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "Home",
          "+",
          "=",
          "-",
          "f",
          "Enter",
          " ",
        ].includes(e.key)
      )
        e.preventDefault();
    });
    minimap.addEventListener("click", (e) => {
      if (!this.miniTransform) return;
      const { scale, x, y } = this.miniTransform;
      const wx =
          ((e.offsetX * minimap.width) / minimap.clientWidth - x) / scale,
        wy = ((e.offsetY * minimap.height) / minimap.clientHeight - y) / scale;
      this.camera.x = this.width / 2 - wx * this.camera.k;
      this.camera.y = this.height / 2 - wy * this.camera.k;
      this.schedule();
    });
  }
  setData(
    model,
    index,
    { axis, metric, step, selected, selectedKey, decision, overlays, shared },
  ) {
    this.axis = axis;
    this.index = index;
    this.metric = metric;
    this.step = step;
    this.selected = selected;
    this.decision = decision;
    this.overlays = overlays;
    this.shared = shared ?? new Set();
    // The layout needs a rooted hierarchy; actual Graph edges remain separate
    // and unchanged, even if a malformed legacy graph has a cycle/self-edge.
    const byKey = new Map(
      model.nodes.map((r) => [r.key, { ...r, children: [] }]),
    );
    const root = byKey.get("root");
    if (!root) {
      this.rows = [];
      this.schedule();
      return;
    }
    const attached = new Set(["root"]);
    for (const row of model.nodes) {
      if (attached.has(row.key)) continue;
      const path = [],
        seen = new Set();
      let cursor = byKey.get(row.key);
      while (cursor && !attached.has(cursor.key) && !seen.has(cursor.key)) {
        path.push(cursor);
        seen.add(cursor.key);
        cursor = byKey.get(cursor.parent);
      }
      let parent = cursor && attached.has(cursor.key) ? cursor : root;
      for (const node of path.reverse()) {
        parent.children.push(node);
        attached.add(node.key);
        parent = node;
      }
    }
    const layout = tree().nodeSize([38, 1])(hierarchy(root));
    this.rows = layout.descendants().map((n) => {
      const row = n.data,
        sequence = index.nodes[row.id];
      const depth =
        axis === "tokens"
          ? (sequence?.tokens ?? 0)
          : (index.birth[row.id] ?? 0);
      const x =
        depth *
        (axis === "tokens"
          ? Math.max(
              2,
              Math.min(18, 140 / (index.record?.config.chunk_tokens ?? 32)),
            )
          : 170);
      return {
        ...row,
        children: row.children.length,
        x,
        y: n.x,
        depth: n.depth,
        value: index.value(row.id, metric, step, row.slot).value,
      };
    });
    // A zero-token terminal transition (and duplicate Graph prefixes) must
    // remain a distinct clickable node at exactly the same token coordinate.
    const columns = new Map();
    for (const row of this.rows) {
      if (!columns.has(row.x)) columns.set(row.x, []);
      columns.get(row.x).push(row);
    }
    for (const column of columns.values()) {
      column.sort(
        (a, b) => a.y - b.y || a.depth - b.depth || a.key.localeCompare(b.key),
      );
      let previous = -Infinity;
      for (const row of column) {
        row.y = Math.max(row.y, previous + 22);
        previous = row.y;
      }
    }
    this.positions = new Map(this.rows.map((r) => [r.key, r]));
    this.edges = model.edges.map((e) => ({
      ...e,
      a: this.positions.get(e.source),
      b: this.positions.get(e.target),
    }));
    this.selectedKey =
      this.positions.has(selectedKey) &&
      this.positions.get(selectedKey).id === selected
        ? selectedKey
        : this.rows.find((r) => r.id === selected)?.key;
    this.ancestry = new Set();
    let current = this.positions.get(this.selectedKey);
    while (current && !this.ancestry.has(current.key)) {
      this.ancestry.add(current.key);
      current = this.positions.get(current.parent);
    }
    this.bounds = { x0: 0, y0: 0, x1: 1, y1: 1 };
    for (const r of this.rows) {
      this.bounds.x0 = Math.min(this.bounds.x0, r.x);
      this.bounds.x1 = Math.max(this.bounds.x1, r.x);
      this.bounds.y0 = Math.min(this.bounds.y0, r.y);
      this.bounds.y1 = Math.max(this.bounds.y1, r.y);
    }
    this.canvas.dataset.nodes = String(this.rows.length);
    this.schedule();
  }
  label(row, full = false) {
    const n = this.index.nodes[row.id];
    if (row.key === "root") return "Prompt";
    if (row.unused) return `Unused slot ${row.slot}`;
    const prefix = `#${row.id}${row.slot !== null ? ` · slot ${row.slot}` : ""}`;
    return full
      ? `${prefix} · ${n.tokens} tokens · ${metricFormat(this.metric, row.value)} · ${this.index.chunk(n.id).slice(0, 90).replace(/\s+/g, " ")}`
      : prefix;
  }
  zoom(factor, x = this.width / 2, y = this.height / 2) {
    const c = this.camera,
      k = Math.max(0.015, Math.min(6, c.k * factor)),
      ratio = k / c.k;
    c.x = x - (x - c.x) * ratio;
    c.y = y - (y - c.y) * ratio;
    c.k = k;
    this.schedule();
  }
  fit() {
    this.measure();
    const b = this.bounds ?? { x0: 0, y0: 0, x1: 1, y1: 1 };
    const k = Math.min(
      1.8,
      Math.max(
        0.015,
        Math.min(
          (this.width - 170) / (b.x1 - b.x0 + 50),
          (this.height - 100) / (b.y1 - b.y0 + 50),
        ),
      ),
    );
    this.camera = {
      k,
      x: 65 - b.x0 * k,
      y: (this.height - (b.y1 - b.y0) * k) / 2 - b.y0 * k,
    };
    this.schedule();
  }
  focus(key = this.selectedKey) {
    const p = this.positions.get(key);
    if (!p) return;
    this.measure();
    this.camera.k = Math.max(0.6, this.camera.k);
    this.camera.x = this.width * 0.4 - p.x * this.camera.k;
    this.camera.y = this.height / 2 - p.y * this.camera.k;
    this.schedule();
  }
  hit(x, y) {
    let nearest = null,
      best = 14;
    for (const p of this.visible ?? []) {
      const d = Math.hypot(
        p.x * this.camera.k + this.camera.x - x,
        p.y * this.camera.k + this.camera.y - y,
      );
      if (d < best) {
        nearest = p;
        best = d;
      }
    }
    return nearest;
  }
  measure() {
    const r = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, r.width);
    this.height = Math.max(1, r.height);
  }
  schedule() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.draw();
    });
  }
  path(ctx, a, b) {
    if (a.key === b.key) {
      ctx.beginPath();
      ctx.arc(
        a.x + 12 / this.camera.k,
        a.y - 12 / this.camera.k,
        15 / this.camera.k,
        0,
        Math.PI * 2,
      );
      ctx.stroke();
      return;
    }
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    const mid = (a.x + b.x) / 2;
    ctx.bezierCurveTo(mid, a.y, mid, b.y, b.x, b.y);
    ctx.stroke();
  }
  draw() {
    this.measure();
    if (this.width < 2 || this.height < 2) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1),
      canvas = this.canvas;
    canvas.width = Math.round(this.width * ratio);
    canvas.height = Math.round(this.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    ctx.fillStyle = "#15111b";
    ctx.fillRect(0, 0, this.width, this.height);
    const c = this.camera,
      view = {
        x0: -c.x / c.k - 100,
        x1: (this.width - c.x) / c.k + 100,
        y0: -c.y / c.k - 30,
        y1: (this.height - c.y) / c.k + 30,
      };
    this.visible = this.rows.filter(
      (p) =>
        p.x >= view.x0 && p.x <= view.x1 && p.y >= view.y0 && p.y <= view.y1,
    );
    const unit =
      this.axis === "tokens"
        ? Math.max(
            2,
            Math.min(18, 140 / (this.index?.record?.config.chunk_tokens ?? 32)),
          )
        : 170;
    const spacing = Math.max(1, Math.ceil(130 / (unit * c.k))),
      begin = Math.max(0, Math.ceil(-c.x / (unit * c.k * spacing)) * spacing);
    ctx.font = "10px sans-serif";
    for (
      let value = begin;
      value * unit * c.k + c.x < this.width;
      value += spacing
    ) {
      const x = value * unit * c.k + c.x;
      ctx.strokeStyle = "#282131";
      ctx.beginPath();
      ctx.moveTo(x, 28);
      ctx.lineTo(x, this.height - 30);
      ctx.stroke();
      ctx.fillStyle = "#8e7f99";
      ctx.fillText(
        `${value} ${this.axis === "tokens" ? "tokens" : "steps"}`,
        x + 5,
        18,
      );
    }
    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.scale(c.k, c.k);
    let edgeCount = 0;
    for (const e of this.edges) {
      if (
        !e.a ||
        !e.b ||
        Math.max(e.a.x, e.b.x) < view.x0 ||
        Math.min(e.a.x, e.b.x) > view.x1 ||
        Math.max(e.a.y, e.b.y) < view.y0 ||
        Math.min(e.a.y, e.b.y) > view.y1
      )
        continue;
      const active =
        this.ancestry?.has(e.source) && this.ancestry?.has(e.target);
      ctx.strokeStyle = active
        ? "#d0a9e2"
        : this.shared.has(e.a.id) && this.shared.has(e.b.id)
          ? "#efc87b"
          : "#4a3b58";
      ctx.lineWidth = (active ? 2.6 : 1) / c.k;
      ctx.setLineDash([]);
      this.path(ctx, e.a, e.b);
      edgeCount++;
    }
    let labels = 0;
    for (const p of this.visible) {
      const selected = p.key === this.selectedKey,
        r = (p.key === "root" ? 8 : selected ? 7 : 5) / c.k;
      ctx.globalAlpha = p.match || p.key === "root" ? 1 : 0.42;
      ctx.fillStyle =
        p.key === "root"
          ? "#f1eaf5"
          : color(this.metric, p.value, this.index.domains[this.metric]);
      ctx.beginPath();
      if (p.unused) ctx.rect(p.x - r, p.y - r, r * 2, r * 2);
      else ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = (selected ? 2.5 : 1) / c.k;
      ctx.strokeStyle = selected ? "#fff" : p.current ? "#7ef5df" : "#15111b";
      ctx.stroke();
      if (p.collapsed) {
        ctx.fillStyle = "#f1eaf5";
        ctx.font = `${12 / c.k}px sans-serif`;
        ctx.fillText("+", p.x + 9 / c.k, p.y + 4 / c.k);
      }
      if (labels < 180 && (c.k > 0.55 || selected || p.key === "root")) {
        ctx.font = `${11 / c.k}px sans-serif`;
        ctx.fillStyle = selected ? "#fff" : "#b5a6c0";
        const n = this.index.nodes[p.id],
          detailed = c.k > 1.45 && n && p.key !== "root";
        const title = this.label(p) + (detailed ? ` · ${n.tokens} tokens` : "");
        const shorten = (text) => {
          const max = 150 / c.k;
          while (text.length > 1 && ctx.measureText(text).width > max)
            text = text.slice(0, -2) + "…";
          return text;
        };
        ctx.fillText(shorten(title), p.x + 12 / c.k, p.y - 8 / c.k);
        if (detailed) {
          ctx.fillStyle = "#8e7f99";
          ctx.fillText(
            shorten(this.index.chunk(p.id).replace(/\s+/g, " ")),
            p.x + 12 / c.k,
            p.y + 7 / c.k,
          );
        }
        labels++;
      }
    }
    ctx.globalAlpha = 1;
    if (this.decision && this.overlays) {
      const d = this.decision;
      const locate = (id, slot) =>
        this.rows.find((n) => n.id === id && n.slot === slot) ??
        this.rows.find((n) => n.id === id);
      const from =
        d.evaluated === null
          ? { x: (26 - c.x) / c.k, y: (60 - c.y) / c.k }
          : locate(d.evaluated, d.slot);
      if (d.evaluated === null) {
        ctx.fillStyle = "#241d2d";
        ctx.fillRect(from.x - 8 / c.k, from.y - 18 / c.k, 200 / c.k, 28 / c.k);
        ctx.fillStyle = "#f1eaf5";
        ctx.font = `${11 / c.k}px sans-serif`;
        ctx.fillText(`Evaluated unused slot ${d.slot}`, from.x, from.y);
      }
      for (const [id, slot, stroke, dash] of [
        [d.companion, d.companion_slot, "#7ef5df", [3, 4]],
        [d.donor, d.donor_slot, "#efc87b", [9, 5]],
      ]) {
        const to = locate(id, slot);
        if (!from || !to) continue;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 2 / c.k;
        ctx.setLineDash(dash.map((x) => x / c.k));
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.quadraticCurveTo(
          (from.x + to.x) / 2,
          (from.y + to.y) / 2 - 60 / c.k,
          to.x,
          to.y,
        );
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(to.x, to.y, 10 / c.k, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();
    ctx.fillStyle = "#b5a6c0";
    ctx.font = "11px sans-serif";
    ctx.fillText(
      `${Math.round(c.k * 100)}% · ${this.visible.length.toLocaleString()} / ${this.rows.length.toLocaleString()} nodes in view`,
      16,
      this.height - 16,
    );
    canvas.dataset.drawn = String(this.visible.length);
    canvas.dataset.labels = String(labels);
    canvas.dataset.edges = String(edgeCount);
    this.drawMini();
  }
  drawMini() {
    const canvas = this.mini,
      ctx = canvas.getContext("2d"),
      w = canvas.width,
      h = canvas.height,
      b = this.bounds;
    if (!b) return;
    ctx.fillStyle = "#1d1825";
    ctx.fillRect(0, 0, w, h);
    const scale = Math.min(
        (w - 20) / (b.x1 - b.x0 + 20),
        (h - 20) / (b.y1 - b.y0 + 20),
      ),
      x = 10 - b.x0 * scale,
      y = (h - (b.y1 - b.y0) * scale) / 2 - b.y0 * scale;
    this.miniTransform = { scale, x, y };
    ctx.fillStyle = "#b5a6c0";
    for (const r of this.rows)
      ctx.fillRect(r.x * scale + x, r.y * scale + y, 2, 2);
    const c = this.camera;
    ctx.strokeStyle = "#7ef5df";
    ctx.lineWidth = 1;
    ctx.strokeRect(
      (-c.x / c.k) * scale + x,
      (-c.y / c.k) * scale + y,
      (this.width / c.k) * scale,
      (this.height / c.k) * scale,
    );
  }
  async export() {
    this.draw();
    const out = document.createElement("canvas");
    out.width = this.canvas.width;
    out.height = this.canvas.height + 110;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#15111b";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(this.canvas, 0, 0);
    ctx.fillStyle = "#f1eaf5";
    ctx.font = "16px sans-serif";
    ctx.fillText(
      `LLM Analysis · step ${this.step} · ${METRICS[this.metric].label}`,
      24,
      this.canvas.height + 28,
    );
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#b5a6c0";
    ctx.fillText(METRICS[this.metric].unit, 24, this.canvas.height + 50);
    const domain = this.index.domains[this.metric];
    if (this.metric === "status") {
      ["Partial", "Finished", "Capped"].forEach((label, i) => {
        ctx.fillStyle = color("status", i, domain);
        ctx.fillRect(24 + i * 120, this.canvas.height + 64, 10, 10);
        ctx.fillStyle = "#f1eaf5";
        ctx.fillText(label, 40 + i * 120, this.canvas.height + 74);
      });
    } else if (domain) {
      for (let i = 0; i < 240; i++) {
        ctx.fillStyle = color(
          this.metric,
          domain[0] + (i / 239) * (domain[1] - domain[0]),
          domain,
        );
        ctx.fillRect(24 + i, this.canvas.height + 64, 1, 10);
      }
      ctx.fillStyle = "#f1eaf5";
      ctx.fillText(
        `${metricFormat(this.metric, domain[0])} … ${metricFormat(this.metric, domain[1])}    Gray: not recorded`,
        280,
        this.canvas.height + 75,
      );
    }
    ctx.fillStyle = "#b5a6c0";
    ctx.fillText(
      "Solid: ancestry · dotted mint: distance companion · dashed gold: clone donor · gold ancestry: shared branch",
      24,
      this.canvas.height + 98,
    );
    return new Promise((resolve) => out.toBlob(resolve, "image/png"));
  }
}
