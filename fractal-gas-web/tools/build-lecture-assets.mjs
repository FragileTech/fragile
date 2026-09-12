import { readFile, writeFile, mkdir } from "node:fs/promises";
import { demos, parameters } from "../web/euclidean-gas/lecture/catalog.js";
import {
  chartSVG,
  escapeXML,
  COLORS,
} from "../web/euclidean-gas/lecture/plots.js";
import init, {
  BrowserGas,
  default_config,
  partv_geometry,
  partv_analysis,
} from "../web/euclidean-gas/engine/cpu/gas.js";

await init({
  module_or_path: await readFile(
    new URL("../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  geometry: async (request) => partv_geometry(JSON.stringify(request)),
  analysis: async (request) => partv_analysis(JSON.stringify(request)),
  defaults: async () => JSON.parse(default_config()),
  create: (config) => BrowserGas.create(JSON.stringify(config)),
  restore: (bytes) => BrowserGas.restore(bytes),
};
const output = new URL("../../docs/_static_theory/gas-demos/", import.meta.url);
await mkdir(output, { recursive: true });
const placements = JSON.parse(
  await readFile(
    new URL("../web/euclidean-gas/lecture/placements.json", import.meta.url),
  ),
);
const captions = JSON.parse(
  await readFile(
    new URL("../web/euclidean-gas/lecture/captions.json", import.meta.url),
  ),
);
if (
  placements.length !== demos.length ||
  new Set(demos.map((d) => d.id)).size !== demos.length
)
  throw new Error(
    "Expected distinct lecture experiments with one placement each",
  );
const manifest = [];
for (const demo of demos) {
  const placement = placements.find((entry) => entry.id === demo.id);
  if (!placement) throw new Error("Missing placement: " + demo.id);
  const caption = captions[demo.id];
  if (!caption?.prediction)
    throw new Error("Missing reviewed caption: " + demo.id);
  const model = await demo.create({
    engine,
    params: parameters(demo),
    seed: 7,
  });
  try {
    let posterTicks = 0;
    for (; posterTicks < 12 && !model.snapshot().done; posterTicks++)
      await model.step();
    const snapshot = model.snapshot();
    const preferred = {
      "II-01": 3,
      "IV-01": 1,
      "IV-09": 1,
      "V-11": 2,
      "V-12": 1,
    };
    const chart =
      snapshot.charts[preferred[demo.id]] ||
      snapshot.charts.find(
        (chart) =>
          chart.series?.some((series) => series.points?.length) ||
          chart.matrix?.length,
      );
    if (!chart) throw new Error("No computed poster data: " + demo.id);
    let svg = chartSVG(chart, { width: 960, height: 450 });
    const legendItems = (chart.series || []).map((series) => {
      const lines = [""];
      for (const word of String(series.name).split(/\s+/)) {
        const last = lines.length - 1;
        if (lines[last] && lines[last].length + word.length + 1 > 67)
          lines.push(word);
        else lines[last] += (lines[last] ? " " : "") + word;
      }
      return { series, lines };
    });
    const legendRows = [];
    let legendBottom = 470;
    for (let i = 0; i < legendItems.length; i += 2) {
      legendRows.push(legendBottom);
      legendBottom +=
        14 *
          Math.max(...legendItems.slice(i, i + 2).map((x) => x.lines.length)) +
        8;
    }
    const posterHeight = Math.max(510, legendBottom + 10);
    svg = svg.replace(
      'viewBox="0 0 960 450"',
      `viewBox="0 0 960 ${posterHeight}"`,
    );
    const legend = legendItems
      .map(
        ({ series, lines }, index) =>
          `<g transform="translate(${24 + (index % 2) * 465} ${legendRows[Math.floor(index / 2)]})">` +
          `<rect width="13" height="3" y="-4" fill="${escapeXML(series.color || COLORS[index % COLORS.length])}"/>` +
          `<text fill="#b7c6dc" font-family="system-ui,sans-serif" font-size="11">` +
          lines
            .map(
              (line, i) =>
                `<tspan x="20" dy="${i ? 14 : 0}">${escapeXML(line)}</tspan>`,
            )
            .join("") +
          "</text></g>",
      )
      .join("");
    svg = svg.replace(/<\/svg>$/, legend + "</svg>");
    await writeFile(new URL(demo.id + ".svg", output), svg);
    manifest.push({
      ...placement,
      title: demo.title,
      kind: demo.kind,
      prediction: caption.prediction,
      question: caption.question || demo.question,
      posterAlt: chart.title + ". " + caption.prediction,
      seed: 7,
      posterTicks,
      controls: demo.controls,
      params: parameters(demo),
    });
    console.log(demo.id + ": " + posterTicks + " computed steps → poster");
  } finally {
    model.dispose?.();
  }
}
await writeFile(
  new URL("manifest.json", output),
  JSON.stringify(manifest, null, 2) + "\n",
);
