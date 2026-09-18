// Refresh only the IV-07 poster and recorded result from the compiled Rust engine.
import { readFile, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import init, { LectureExperiment } from '../web/euclidean-gas/engine/cpu/gas.js';
import { demos, parameters } from '../web/euclidean-gas/lecture/catalog.js';
import { chartSVG, escapeXML, COLORS } from '../web/euclidean-gas/lecture/plots.js';
await init({module_or_path:await readFile(new URL('../web/euclidean-gas/engine/cpu/gas_bg.wasm',import.meta.url))});
const demo=demos.find(d=>d.id==='IV-07');
const model=await demo.create({params:parameters(demo),seed:7,engine:{lectureCreate:r=>LectureExperiment.create(JSON.stringify(r))}});
try {
  let ticks=0;
  while(!model.snapshot().done && ticks<1024){await model.step();ticks++;}
  const snapshot=model.snapshot();
  assert.ok(snapshot.done && snapshot.result);
  const chart=snapshot.charts[0];
  let svg=chartSVG(chart,{width:960,height:450});
  svg=svg.replace('height="450"','height="520"').replace('viewBox="0 0 960 450"','viewBox="0 0 960 520"');
  const legend=chart.series.map((s,i)=>`<text x="30" y="${478+22*i}" fill="${COLORS[i%COLORS.length]}" font-size="14">${escapeXML(s.name)}</text>`).join('');
  svg=svg.replace('</svg>',legend+'</svg>');
  const output=new URL('../../docs/_static_theory/gas-demos/',import.meta.url);
  await writeFile(new URL('IV-07.svg',output),svg);
  await writeFile(new URL('IV-07.result.json',output),JSON.stringify(snapshot.result)+'\n');
  const manifestPath=new URL('manifest.json',output);
  const manifest=JSON.parse(await readFile(manifestPath,'utf8'));
  const entry=manifest.find(e=>e.id==='IV-07');
  Object.assign(entry,{title:demo.title,target:'sec-cinf-companion-sensitivity-experiment',prediction:demo.prediction,question:demo.question,posterAlt:chart.title+'. '+demo.prediction,seed:7,posterTicks:ticks,controls:demo.controls,params:parameters(demo),calculationOrigin:snapshot.result.details.calculation_origin});
  await writeFile(manifestPath,JSON.stringify(manifest,null,2)+'\n');
  console.log(JSON.stringify({step:snapshot.step,metrics:snapshot.result.metrics,coefficient_checks:snapshot.result.details.coefficient_checks},null,2));
} finally {model.dispose();}
