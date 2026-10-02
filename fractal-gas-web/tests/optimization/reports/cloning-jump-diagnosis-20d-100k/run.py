"""Small diagnostic ablation, separate from the ongoing million-evaluation sweep."""
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[4]
SPEC=importlib.util.spec_from_file_location('benchmark',ROOT/'tools/benchmark-long-budget.py')
BENCH=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BENCH)
BENCH.VARIANTS['Diagnostic']={'perturbation':'cloning_guided'}
VARIANTS={
    'Current':{},
    'No drift':{'cloning_drift':False},
    'No covariance':{'cloning_geometry':False},
    'Scales / 10':{'adaptive_min_scale':.00001,'adaptive_max_scale':.02},
    'Scales x 10':{'adaptive_min_scale':.001,'adaptive_max_scale':2},
}
PROBLEMS=['quadratic','bbob_10','rastrigin']
def run(case):
    problem,variant,seed=case
    row=BENCH.run_case((problem,20,'Diagnostic',seed,100000),
        {'walkers':128,'max_walkers':128,'controller_enabled':False,'boundary':'cma',**VARIANTS[variant]},target_error=1e-5)
    row['variant']=variant
    return row
if __name__=='__main__':
    if (OUT/'runs.jsonl').exists(): raise RuntimeError('Do not overwrite measurements')
    files=[Path(__file__),Path(BENCH.__file__),BENCH.LIBRARY,*sorted((ROOT/'src').rglob('*.cpp')),*sorted((ROOT/'src').rglob('*.hpp'))]
    manifest={'dimensions':20,'population':128,'elites':5,'budget':100000,'seeds':[0,1],'boundary':'cma','variants':VARIANTS,'fingerprints':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    with (OUT/'runs.jsonl').open('w') as stream, concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(run,(p,v,s)) for p in PROBLEMS for v in VARIANTS for s in range(2)]
        for job in concurrent.futures.as_completed(jobs):
            row=job.result();rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(len(rows),row['problem'],row['variant'],row['seed'],row['regret'],row['error'],flush=True)
    assert len(rows)==30
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['fingerprints'].items())
    lines=['# Cloning-guided jump diagnosis','',
      'Pilot ablation: 20D, 128 walkers, five elites, two seeds, 100,000 evaluations per run, target 1e-5, CMA boundary repair, no restarts. Current scale bounds are [0.0001, 0.2]. Mixed precision: FP32 walkers, FP64 objectives. These short runs measure sensitivity, not a definitive explanation of the million-evaluation comparison.','',
      '| Variant | Quadratic median error | Rotated ellipsoid median error | Rastrigin median error |','|---|---:|---:|---:|']
    for v in VARIANTS:
        lines.append('| '+v+' | '+' | '.join(f"{statistics.median(r['regret'] for r in rows if r['variant']==v and r['problem']==p):.6g}" for p in PROBLEMS)+' |')
    lines+=['',f"Execution errors: {sum(bool(r['error']) for r in rows)}. Actual evaluations: {sum(r['evaluations'] for r in rows):,}.",'']
    (OUT/'report.md').write_text('\n'.join(lines))
