export const COLORS = [
  "#51d2b7",
  "#f5ad69",
  "#91b1ff",
  "#df96db",
  "#e8d874",
  "#b4ccd6",
];
export const escapeXML = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
export function format(value) {
  if (typeof value !== "number") return String(value ?? "—");
  if (!Number.isFinite(value))
    return value === Infinity ? "∞" : value === -Infinity ? "−∞" : "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 100000)
    return value.toExponential(2);
  return Number(value.toPrecision(4)).toLocaleString("en-US", {
    maximumFractionDigits: 5,
  });
}
const valid = (p) =>
  Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]);
function extent(values, fallback = [-1, 1], logarithmic = false) {
  values = values.filter((x) => Number.isFinite(x) && (!logarithmic || x > 0));
  if (!values.length) return logarithmic ? [0.01, 1] : fallback;
  let a = Math.min(...values),
    b = Math.max(...values);
  if (a === b) {
    if (logarithmic) return [a / 2, a * 2];
    const pad = Math.max(0.5, Math.abs(a) * 0.1);
    return [a - pad, b + pad];
  }
  if (logarithmic) return [a, b];
  const pad = (b - a) * 0.06;
  return [a - pad, b + pad];
}
function text(x, y, content, extra = "") {
  return (
    '<text x="' +
    x +
    '" y="' +
    y +
    '" ' +
    extra +
    ">" +
    escapeXML(content) +
    "</text>"
  );
}
function color(value, lo, hi) {
  const q = Math.max(0, Math.min(1, (value - lo) / (hi - lo || 1)));
  return (
    "rgb(" +
    [32 + 49 * q, 49 + 161 * q, 72 + 111 * q].map(Math.round).join(",") +
    ")"
  );
}
export function chartSVG(
  chart,
  { width = 640, height = 340, background = "#101b2c" } = {},
) {
  const left = 82,
    right = width - 24,
    top = 30,
    bottom = height - 70;
  let html =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' +
    width +
    " " +
    height +
    '" role="img" aria-label="' +
    escapeXML(chart.title || "Scientific plot") +
    '" style="width:100%;height:auto;background:' +
    escapeXML(background) +
    '">' +
    "<title>" +
    escapeXML(chart.title || "Scientific plot") +
    "</title>" +
    '<g font-family="system-ui,sans-serif" font-size="11" fill="#b7c6dc">';
  if (chart.matrix?.length) {
    const matrix = chart.matrix,
      nr = matrix.length,
      nc = Math.max(...matrix.map((row) => row.length));
    const finite = matrix.flat().filter(Number.isFinite);
    const [lo, hi] = chart.colorDomain || extent(finite);
    const cw = (right - left) / nc,
      ch = (bottom - top) / nr;
    matrix.forEach((row, i) =>
      row.forEach((value, j) => {
        html +=
          '<rect x="' +
          (left + j * cw) +
          '" y="' +
          (top + i * ch) +
          '" width="' +
          (cw - 1) +
          '" height="' +
          (ch - 1) +
          '" fill="' +
          color(value, lo, hi) +
          '"><title>' +
          escapeXML(
            "Row " + (i + 1) + ", column " + (j + 1) + ": " + format(value),
          ) +
          "</title></rect>";
        if (nr <= 10 && nc <= 10)
          html += text(
            left + (j + 0.5) * cw,
            top + (i + 0.5) * ch + 4,
            format(value),
            'text-anchor="middle" fill="#eaf3fc"',
          );
      }),
    );
    for (let i = 0; i < nr; i++)
      if (nr <= 16 || i % Math.ceil(nr / 10) === 0)
        html += text(
          left - 9,
          top + (i + 0.5) * ch + 4,
          chart.rowLabels?.[i] ?? i + 1,
          'text-anchor="end"',
        );
    for (let j = 0; j < nc; j++)
      if (nc <= 16 || j % Math.ceil(nc / 10) === 0)
        html += text(
          left + (j + 0.5) * cw,
          bottom + 18,
          chart.columnLabels?.[j] ?? j + 1,
          'text-anchor="middle"',
        );
  } else {
    const series = chart.series || [];
    const points = [
      ...series.flatMap((s) => (s.points || []).filter(valid)),
      ...(chart.segments || []).flat().filter(valid),
    ];
    const logX = chart.xScale === "log",
      logY = chart.yScale === "log";
    const xd =
      chart.xDomain ||
      extent(
        points.map((p) => p[0]),
        [-1, 1],
        logX,
      );
    const yd =
      chart.yDomain ||
      extent(
        points.map((p) => p[1]),
        [-1, 1],
        logY,
      );
    const tx = (v) => (logX ? Math.log10(v) : v),
      ty = (v) => (logY ? Math.log10(v) : v);
    const x = (v) =>
      left +
      ((tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0]) || 1)) * (right - left);
    const y = (v) =>
      bottom -
      ((ty(v) - ty(yd[0])) / (ty(yd[1]) - ty(yd[0]) || 1)) * (bottom - top);
    const plottable = (p) =>
      valid(p) && (!logX || p[0] > 0) && (!logY || p[1] > 0);
    for (let i = 0; i <= 4; i++) {
      const q = i / 4,
        px = left + q * (right - left),
        py = bottom - q * (bottom - top);
      const xv = logX
        ? 10 ** (tx(xd[0]) + q * (tx(xd[1]) - tx(xd[0])))
        : xd[0] + q * (xd[1] - xd[0]);
      const yv = logY
        ? 10 ** (ty(yd[0]) + q * (ty(yd[1]) - ty(yd[0])))
        : yd[0] + q * (yd[1] - yd[0]);
      html +=
        '<path d="M' +
        px +
        " " +
        top +
        "V" +
        bottom +
        " M" +
        left +
        " " +
        py +
        "H" +
        right +
        '" stroke="#26364b" fill="none"/>';
      html += text(px, bottom + 19, format(xv), 'text-anchor="middle"');
      html += text(left - 9, py + 4, format(yv), 'text-anchor="end"');
    }
    html +=
      '<svg x="' +
      left +
      '" y="' +
      top +
      '" width="' +
      (right - left) +
      '" height="' +
      (bottom - top) +
      '" viewBox="' +
      left +
      " " +
      top +
      " " +
      (right - left) +
      " " +
      (bottom - top) +
      '" overflow="hidden">';
    for (const [a, b] of chart.segments || [])
      if (plottable(a) && plottable(b))
        html +=
          '<path d="M' +
          x(a[0]) +
          " " +
          y(a[1]) +
          "L" +
          x(b[0]) +
          " " +
          y(b[1]) +
          '" stroke="#859bb9" stroke-width="1" opacity="0.65"/>';
    series.forEach((s, index) => {
      const ink = escapeXML(s.color || COLORS[index % COLORS.length]);
      const ps = (s.points || []).filter(plottable);
      if (s.style === "points") {
        for (const [i, p] of ps.entries())
          html +=
            '<circle data-point="' +
            i +
            '" data-series="' +
            index +
            '" cx="' +
            x(p[0]) +
            '" cy="' +
            y(p[1]) +
            '" r="' +
            (s.radius || 3.2) +
            '" fill="' +
            ink +
            '" opacity="0.85"><title>' +
            escapeXML(s.name + ": " + format(p[0]) + ", " + format(p[1])) +
            "</title></circle>";
      } else if (s.style === "bars") {
        const bw = Math.max(
          1,
          Math.min(36, ((right - left) / Math.max(1, ps.length)) * 0.75),
        );
        const zero = Math.max(top, Math.min(bottom, y(logY ? yd[0] : 0)));
        ps.forEach((p) => {
          const py = y(p[1]);
          html +=
            '<rect x="' +
            (x(p[0]) - bw / 2) +
            '" y="' +
            Math.min(py, zero) +
            '" width="' +
            bw +
            '" height="' +
            Math.max(0.5, Math.abs(zero - py)) +
            '" fill="' +
            ink +
            '" opacity="0.8"><title>' +
            escapeXML(format(p[0]) + ": " + format(p[1])) +
            "</title></rect>";
        });
      } else {
        let path = "",
          continuing = false;
        for (const p of s.points || []) {
          if (!plottable(p)) {
            continuing = false;
            continue;
          }
          path +=
            (continuing ? "L" : "M") +
            x(p[0]).toFixed(3) +
            " " +
            y(p[1]).toFixed(3);
          continuing = true;
        }
        html +=
          '<path d="' +
          path +
          '" stroke="' +
          ink +
          '" stroke-width="2.2" fill="none"' +
          (s.dashed ? ' stroke-dasharray="6 4"' : "") +
          "/>";
        if (ps.length === 1)
          html +=
            '<circle cx="' +
            x(ps[0][0]) +
            '" cy="' +
            y(ps[0][1]) +
            '" r="3" fill="' +
            ink +
            '"/>';
      }
    });
    html += "</svg>";
  }
  const labelLines = [""],
    lineLength = Math.floor((width - 45) / 6);
  for (const word of String(chart.xLabel || "").split(" ")) {
    const last = labelLines.length - 1;
    if (
      labelLines[last].length + word.length > lineLength &&
      labelLines.length < 3
    )
      labelLines.push(word);
    else labelLines[last] += (labelLines[last] ? " " : "") + word;
  }
  labelLines.forEach((label, index) => {
    html += text(
      (left + right) / 2,
      height - 35 + index * 12,
      label,
      'text-anchor="middle" fill="#d9e6f7"',
    );
  });
  html +=
    '<text transform="translate(15 ' +
    (top + bottom) / 2 +
    ') rotate(-90)" text-anchor="middle" fill="#d9e6f7">' +
    escapeXML(chart.yLabel || "") +
    "</text>";
  return html + "</g></svg>";
}
export function legendHTML(chart) {
  return (chart.series || [])
    .map(
      (s, i) =>
        '<span class="series-key"><i style="background:' +
        escapeXML(s.color || COLORS[i % COLORS.length]) +
        '"></i>' +
        escapeXML(s.name) +
        "</span>",
    )
    .join("");
}
