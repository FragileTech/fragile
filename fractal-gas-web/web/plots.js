// Tiny dependency-free canvas line plots, mirroring the panes of the Panel
// dashboard (dashboard.py): reward, virtual reward, clone %, alive count,
// mean dt.

export class LinePlot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{title: string, series: {name: string, color: string}[], maxPoints?: number}} opts
   */
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.title = opts.title;
    this.series = opts.series.map((s) => ({ ...s, data: [] }));
    this.maxPoints = opts.maxPoints ?? 2000;
    this.render();
  }

  /** Append one value per series (same order as construction). */
  append(values) {
    for (let i = 0; i < this.series.length; i++) {
      const data = this.series[i].data;
      data.push(values[i]);
      if (data.length > this.maxPoints) data.shift();
    }
    this.render();
  }

  clear() {
    for (const s of this.series) s.data = [];
    this.render();
  }

  render() {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    const padLeft = 46;
    const padRight = 8;
    const padTop = 22;
    const padBottom = 16;

    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#8a90a2";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(this.title, padLeft, 5);

    // Legend
    let legendX = w - padRight;
    ctx.textAlign = "right";
    for (let i = this.series.length - 1; i >= 0; i--) {
      const s = this.series[i];
      ctx.fillStyle = s.color;
      ctx.fillText(s.name, legendX, 5);
      legendX -= ctx.measureText(s.name).width + 14;
    }

    const n = Math.max(...this.series.map((s) => s.data.length));
    if (n < 2) return;

    let min = Infinity;
    let max = -Infinity;
    for (const s of this.series) {
      for (const v of s.data) {
        if (Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return;
    if (max === min) {
      max += 1;
      min -= 1;
    }
    const span = max - min;
    min -= span * 0.05;
    max += span * 0.05;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;
    const xOf = (i) => padLeft + (i / (n - 1)) * plotW;
    const yOf = (v) => padTop + (1 - (v - min) / (max - min)) * plotH;

    // Axis labels + gridlines
    ctx.strokeStyle = "#2c3040";
    ctx.fillStyle = "#6b7183";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const frac of [0, 0.5, 1]) {
      const v = min + frac * (max - min);
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();
      ctx.fillText(formatTick(v), padLeft - 5, y);
    }

    for (const s of this.series) {
      if (s.data.length < 2) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const offset = n - s.data.length;
      for (let i = 0; i < s.data.length; i++) {
        const x = xOf(i + offset);
        const y = yOf(s.data[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
}

function formatTick(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(2);
}
