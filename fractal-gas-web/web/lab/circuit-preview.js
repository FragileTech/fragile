// Always preview the active scene, including imported scenes and replay archives.
const SVG = "http://www.w3.org/2000/svg";
const validRing = (r) =>
  Array.isArray(r) &&
  r.length >= 3 &&
  r.length <= 4096 &&
  r.every(
    (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
  );

export function circuitPreview(scene) {
  if (
    scene.environment?.kind !== "circuit" ||
    !validRing(scene.boundary) ||
    !(scene.holes || []).every(validRing)
  )
    return null;
  const rings = [scene.boundary, ...(scene.holes || [])];
  const xs = scene.boundary.map((p) => p[0]);
  const ys = scene.boundary.map((p) => -p[1]);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  if (width <= 0 || height <= 0) return null;
  const padding = Math.max(width, height) * 0.06;
  const metadata = scene.circuit || {};
  let source;
  try {
    const url = new URL(metadata.sources?.[0]?.url);
    if (url.protocol === "https:") source = url.href;
  } catch {
    /* Imported scenes need not have a reference video. */
  }
  return {
    name:
      typeof metadata.name === "string"
        ? metadata.name
        : scene.name || "Custom circuit",
    difficulty: ["Easy", "Medium", "Hard"].includes(metadata.difficulty)
      ? metadata.difficulty
      : "Unrated",
    direction: ["clockwise", "counterclockwise"].includes(metadata.direction)
      ? metadata.direction
      : "",
    source,
    viewBox: [
      Math.min(...xs) - padding,
      Math.min(...ys) - padding,
      width + padding * 2,
      height + padding * 2,
    ].join(" "),
    path: rings
      .map((r) => `M${r.map(([x, y]) => `${x},${-y}`).join("L")}Z`)
      .join(" "),
    strokeWidth: Math.max(width, height) * 0.003,
    checkpoints: scene.gates?.length || 0,
  };
}

export function renderCircuitPreview(container, scene) {
  const preview = circuitPreview(scene);
  container.replaceChildren();
  container.hidden = !preview;
  if (!preview) return;
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", preview.viewBox);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${preview.name} track outline`);
  const path = document.createElementNS(SVG, "path");
  path.setAttribute("d", preview.path);
  path.setAttribute("fill-rule", "evenodd");
  path.setAttribute("stroke-width", preview.strokeWidth);
  svg.append(path);
  const start = scene.environment.start;
  if (
    Array.isArray(start?.position) &&
    start.position.length === 2 &&
    start.position.every(Number.isFinite) &&
    Number.isFinite(start.angle)
  ) {
    const [x, y] = start.position;
    const radius = preview.strokeWidth * 2.5;
    const dot = document.createElementNS(SVG, "circle");
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", -y);
    dot.setAttribute("r", radius);
    dot.setAttribute("class", "circuit-start");
    const arrow = document.createElementNS(SVG, "path");
    const point = (along, across) =>
      `${x + along * Math.cos(start.angle) - across * Math.sin(start.angle)},${-(y + along * Math.sin(start.angle) + across * Math.cos(start.angle))}`;
    arrow.setAttribute(
      "d",
      `M${point(radius * 4, 0)}L${point(radius, radius)}L${point(radius, -radius)}Z`,
    );
    arrow.setAttribute("class", "circuit-start");
    svg.append(dot, arrow);
  }
  const heading = document.createElement("div");
  heading.className = "circuit-heading";
  const name = document.createElement("strong");
  name.textContent = preview.name;
  const difficulty = document.createElement("span");
  difficulty.className = "circuit-difficulty";
  difficulty.dataset.difficulty = preview.difficulty.toLowerCase();
  difficulty.textContent = preview.difficulty;
  heading.append(name, difficulty);
  const detail = document.createElement("p");
  detail.textContent = `${preview.checkpoints} checkpoints${preview.direction ? ` · ${preview.direction}` : ""}`;
  container.append(svg, heading, detail);
  if (preview.source) {
    const link = document.createElement("a");
    link.href = preview.source;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Historical reference ↗";
    container.append(link);
  }
}
