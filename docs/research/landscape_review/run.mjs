import {readFile,writeFile} from 'node:fs/promises';
import init,{default_config,BrowserGas} from '../../../fractal-gas-web/web/euclidean-gas/engine/cpu/gas.js';
await init({module_or_path:await readFile(new URL('../../../fractal-gas-web/web/euclidean-gas/engine/cpu/gas_bg.wasm',import.meta.url))});
const preset=JSON.parse(await readFile(new URL('../../../algorithmic-gas/crates/algorithmic-gas/tests/fixtures/variants/euclidean_d2_dt0.04.json',import.meta.url)));
const results=[];
function measure(s){const r=s.population.rewards.raw,x=s.population.observations.fields.positions.values,n=r.length;let radius2=0,maxRadius=0;const wells={};for(let i=0;i<n;i++){const rr=x[2*i]**2+x[2*i+1]**2;radius2+=rr/n;maxRadius=Math.max(maxRadius,Math.sqrt(rr));const k=[Math.round(x[2*i]),Math.round(x[2*i+1])].join(',');wells[k]=(wells[k]||0)+1;}return {step:s.step,mean:r.reduce((a,b)=>a+b,0)/n,best:Math.min(...r),radius2,maxRadius,wells};}
for(const benchmark of ['sphere','rastrigin'])for(const center of [0,3])for(const seed of [7,19]){
 const c=JSON.parse(default_config());Object.assign(c,{benchmark,walkers:64,dimensions:2,initial_lower:center-.2,initial_upper:center+.2,gas:structuredClone(preset)});c.gas.seed=seed;c.gas.boundary={kind:'unbounded'};c.gas.kinetic.integrator.dt=.01;
 const g=await BrowserGas.create(JSON.stringify(c));const series=[measure(g.snapshot())];const start=Date.now();
 for(let k=0;k<150;k++)series.push(measure(await g.step(10)));
 g.free();const tail=series.filter(s=>s.step>1000);const summary={benchmark,center,seed,initialMean:series[0].mean,tailMean:tail.reduce((a,b)=>a+b.mean,0)/tail.length,tailMinMean:Math.min(...tail.map(x=>x.mean)),tailMaxMean:Math.max(...tail.map(x=>x.mean)),maxObservedRadius:Math.max(...series.map(x=>x.maxRadius)),tailRadius2:tail.reduce((a,b)=>a+b.radius2,0)/tail.length,upwardObservations:series.slice(1).filter((s,i)=>s.mean>series[i].mean).length,final:series.at(-1),seconds:(Date.now()-start)/1000};results.push({config:c,summary,series});console.log(JSON.stringify(summary));await writeFile(new URL('./results.json',import.meta.url),JSON.stringify(results,null,2));
}
