import {
  METHOD_STYLE,
  finite,
  maxValue,
  minValue,
} from "./comparison-metrics.js";
const NS = "http://www.w3.org/2000/svg";
export const fmt = (x) =>
  finite(x)
    ? x !== 0 && Math.abs(x) < 0.0001
      ? x.toExponential(3)
      : Math.abs(x) >= 10000
        ? x.toLocaleString(undefined, { maximumFractionDigits: 0 })
        : Number(x.toPrecision(4)).toLocaleString(undefined, {
            maximumFractionDigits: 5,
          })
    : "Unavailable";
export function element(tag, text, attrs = {}) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function svg(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
export function download(content, filename, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = element("a", null, { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function marker(method, x, y, size = 4) {
  const style = METHOD_STYLE[method] ?? METHOD_STYLE.fractal,
    common = { fill: style.color, stroke: style.color };
  if (style.marker === "square")
    return svg("rect", {
      ...common,
      x: x - size,
      y: y - size,
      width: size * 2,
      height: size * 2,
    });
  if (style.marker === "diamond")
    return svg("polygon", {
      ...common,
      points: `${x},${y - size * 1.5} ${x + size},${y} ${x},${y + size * 1.5} ${x - size},${y}`,
    });
  if (style.marker === "triangle")
    return svg("polygon", {
      ...common,
      points: `${x},${y - size * 1.5} ${x + size * 1.2},${y + size} ${x - size * 1.2},${y + size}`,
    });
  return svg("circle", { ...common, cx: x, cy: y, r: size });
}
export function legend(methods) {
  const e = element("div", null, { class: "compare-legend" });
  for (const method of methods) {
    const span = element("span", METHOD_STYLE[method].label);
    const icon = svg("svg", {
      viewBox: "0 0 16 16",
      width: 16,
      height: 16,
      "aria-hidden": "true",
    });
    icon.append(marker(method, 8, 8, 3));
    span.prepend(icon);
    e.append(span);
  }
  return e;
}
export function chart(
  host,
  {
    title,
    xLabel,
    yLabel,
    series,
    kind = "line",
    onSelect,
    note = "",
    domain,
    zeroY = false,
  },
) {
  const card = element("section", null, { class: "compare-chart" }),
    heading = element("div", null, { class: "section-title" });
  heading.append(element("h3", title));
  const save = element("button", "SVG", {
      type: "button",
      "aria-label": `Download ${title} as SVG`,
    }),
    png = element("button", "PNG", {
      type: "button",
      "aria-label": `Download ${title} as PNG`,
    });
  heading.append(save, png);
  card.append(heading);
  const all = series
    .flatMap((s) => s.points)
    .filter((p) => finite(p.x) && finite(p.y));
  host.append(card);
  if (!all.length) {
    card.append(
      element("p", note || "Unavailable for the selected answers.", {
        class: "hint",
      }),
    );
    save.disabled = png.disabled = true;
    return card;
  }
  const W = 600,
    H = 300,
    L = 70,
    R = 20,
    T = 18,
    B = 62,
    plot = svg("svg", {
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      "aria-label": `${title}. ${xLabel}; ${yLabel}.`,
      xmlns: NS,
    });
  plot.append(svg("rect", { width: W, height: H, fill: "#1c1226" }));
  let xmin = domain?.[0] ?? minValue(all.map((p) => p.x)),
    xmax = domain?.[1] ?? maxValue(all.map((p) => p.x));
  let ymin = zeroY ? 0 : minValue(all.map((p) => p.y)),
    ymax = maxValue(all.map((p) => p.y));
  if (xmin === xmax) {
    xmin -= 0.5;
    xmax += 0.5;
  }
  if (ymin === ymax) {
    ymin -= 0.5;
    ymax += 0.5;
  }
  const x = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R),
    y = (v) => H - B - ((v - ymin) / (ymax - ymin)) * (H - T - B);
  for (let i = 0; i < 5; i++) {
    const xx = L + (i * (W - L - R)) / 4,
      yy = T + (i * (H - T - B)) / 4;
    plot.append(
      svg("line", {
        x1: L,
        x2: W - R,
        y1: yy,
        y2: yy,
        stroke: "#49334e",
        "stroke-width": 0.7,
      }),
      svg(
        "text",
        {
          x: L - 9,
          y: yy + 4,
          "text-anchor": "end",
          fill: "#bba8c5",
          "font-size": 11,
        },
        fmt(ymax - (i * (ymax - ymin)) / 4),
      ),
      svg(
        "text",
        {
          x: xx,
          y: H - B + 20,
          "text-anchor": "middle",
          fill: "#bba8c5",
          "font-size": 11,
        },
        fmt(xmin + (i * (xmax - xmin)) / 4),
      ),
    );
  }
  plot.append(
    svg(
      "text",
      {
        x: (L + W - R) / 2,
        y: H - 10,
        "text-anchor": "middle",
        fill: "#e5d4ed",
        "font-size": 12,
      },
      xLabel,
    ),
    svg(
      "text",
      {
        x: 14,
        y: (T + H - B) / 2,
        transform: `rotate(-90 14 ${(T + H - B) / 2})`,
        "text-anchor": "middle",
        fill: "#e5d4ed",
        "font-size": 12,
      },
      yLabel,
    ),
  );
  for (const s of series) {
    const pts = s.points.filter((p) => finite(p.x) && finite(p.y)),
      color = METHOD_STYLE[s.method]?.color ?? "#d0a9e2";
    if (kind === "line" || kind === "step") {
      let d = "";
      pts.forEach((p, i) => {
        d += i
          ? kind === "step"
            ? ` H${x(p.x)} V${y(p.y)}`
            : ` L${x(p.x)},${y(p.y)}`
          : `M${x(p.x)},${y(p.y)}`;
      });
      plot.append(
        svg("path", {
          d,
          stroke: color,
          fill: "none",
          "stroke-width": 2,
          "stroke-dasharray": s.dashed ? "5 4" : "none",
          opacity: 0.8,
        }),
      );
    }
    if (kind === "histogram")
      for (const p of pts) {
        const left = x(p.range[0]),
          right = x(p.range[1]),
          rect = svg("rect", {
            x: left,
            y: y(p.y),
            width: Math.max(1, right - left),
            height: Math.max(0, y(0) - y(p.y)),
            fill: color,
            "fill-opacity": 0.14,
            stroke: color,
            "stroke-width": 1,
          });
        rect.append(
          svg(
            "title",
            {},
            `${METHOD_STYLE[s.method].label}: ${fmt(p.range[0])} to ${fmt(p.range[1])}; mass ${fmt(p.y)}`,
          ),
        );
        if (onSelect) {
          rect.setAttribute("role", "button");
          rect.onclick = () => onSelect(p, s);
        }
        plot.append(rect);
      }
    // Keep charts interactive without putting thousands of invisible controls in tab order.
    const stride = Math.max(1, Math.ceil(pts.length / 2500));
    for (let i = 0; i < pts.length; i += stride) {
      const p = pts[i],
        mark = marker(s.method, x(p.x), y(p.y), kind === "scatter" ? 3.5 : 2.5);
      mark.append(
        svg(
          "title",
          {},
          `${METHOD_STYLE[s.method]?.label ?? ""}${s.label ? " · " + s.label : ""}: ${fmt(p.x)}, ${fmt(p.y)}${p.label ? " · " + p.label : ""}`,
        ),
      );
      if (onSelect) {
        mark.setAttribute("role", "button");
        mark.setAttribute("aria-label", `${title}: ${fmt(p.x)}, ${fmt(p.y)}`);
        if (i === 0) mark.setAttribute("tabindex", "0");
        mark.onclick = () => onSelect(p, s);
        mark.onkeydown = (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect(p, s);
          }
        };
      }
      plot.append(mark);
    }
  }
  if (series.some((s) => s.points.length > 2500))
    card.append(
      element(
        "p",
        "Markers are thinned to at most 2,500 per series for display. All values remain in the numeric export and trace table.",
        { class: "hint" },
      ),
    );
  card.append(plot, legend([...new Set(series.map((s) => s.method))]));
  if (note) card.append(element("p", note, { class: "hint" }));
  const name = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  save.onclick = () =>
    download(
      new XMLSerializer().serializeToString(plot),
      `${name}.svg`,
      "image/svg+xml",
    );
  png.onclick = async () => {
    const url = URL.createObjectURL(
      new Blob([new XMLSerializer().serializeToString(plot)], {
        type: "image/svg+xml",
      }),
    );
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const canvas = document.createElement("canvas");
      canvas.width = W * 2;
      canvas.height = H * 2;
      canvas.getContext("2d").drawImage(img, 0, 0, W * 2, H * 2);
      const a = element("a", null, {
        download: `${name}.png`,
        href: canvas.toDataURL(),
      });
      a.click();
    } finally {
      URL.revokeObjectURL(url);
    }
  };
  return card;
}
