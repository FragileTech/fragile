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
} from "../web/euclidean-gas/engine/cpu/gas.js";

await init({
  module_or_path: await readFile(
    new URL("../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
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
  demos.length !== 42 ||
  placements.length !== demos.length ||
  new Set(demos.map((d) => d.id)).size !== 42
)
  throw new Error("Expected 42 distinct lecture experiments and placements");
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
    const preferred = { "II-01": 3, "IV-01": 1, "IV-09": 1 };
    const chart =
      snapshot.charts[preferred[demo.id]] ||
      snapshot.charts.find(
        (chart) =>
          chart.series?.some((series) => series.points?.length) ||
          chart.matrix?.length,
      );
    if (!chart) throw new Error("No computed poster data: " + demo.id);
    let svg = chartSVG(chart, { width: 960, height: 450 });
    svg = svg.replace('viewBox="0 0 960 450"', 'viewBox="0 0 960 510"');
    const legend = (chart.series || [])
      .map(
        (series, index) =>
          '<g transform="translate(' +
          (24 + (index % 3) * 310) +
          " " +
          (470 + Math.floor(index / 3) * 17) +
          ')"><rect width="13" height="3" y="-4" fill="' +
          escapeXML(series.color || COLORS[index % COLORS.length]) +
          '"/><text x="20" fill="#b7c6dc" font-family="system-ui,sans-serif" font-size="11">' +
          escapeXML(series.name) +
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
