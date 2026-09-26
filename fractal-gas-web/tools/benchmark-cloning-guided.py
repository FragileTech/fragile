"""Equal-budget ablation of cloning-guided movement using the shared native runner."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('long_budget',Path(__file__).with_name('benchmark-long-budget.py'))
BENCH=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
VARIANTS={
    'Gaussian': {'perturbation':'gaussian'},
    'Local covariance': {'perturbation':'local_covariance'},
    'Bounded adaptive': {'perturbation':'adaptive_fractal'},
    'Cloning geometry': {'perturbation':'cloning_guided','cloning_drift':False},
    'Cloning drift': {'perturbation':'cloning_guided','cloning_geometry':False},
    'Cloning combined': {'perturbation':'cloning_guided'},
    'BIPOP-active CMA-ES': {'algorithm':'cmaes_bipop'},
}
BENCH.VARIANTS.update(VARIANTS)

def run(case):
    return BENCH.run_case(case,{'walkers':128,'max_walkers':128,'controller_enabled':False})

def report(output,rows):
    scalar=[k for k in rows[0] if k not in ('config','initial_status','final_status')]
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=scalar,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Cloning-guided movement: equal-budget ablation','',
           'Six 20D problems; ten seeds per configuration; 100,000-evaluation cap. Fractal methods use Wave with 128 walkers, five elites, scale bounds [0.0001, 0.2], and no automatic restarts. Gaussian/local covariance standard deviation is 0.2. CMA-ES uses its own population and restart schedule. All results were freshly measured with the same engine. No settings were tuned after examining these results.','',
           'Error is best evaluated objective minus the known optimum; lower is better. Intervals are inclusive 25th–75th percentiles. Errors retain their last best objective. Timing is process CPU time in a concurrent benchmark, not isolated latency. Actual evaluation counts can be below the cap because complete steps must fit.','',
           f"{len(rows)} runs; {sum(bool(r['error']) for r in rows)} execution errors; {sum(r['evaluations'] for r in rows):,} actual evaluations.",'',
           '| Problem | '+' | '.join(VARIANTS)+' |','|---|'+'---:|'*len(VARIANTS)]
    for problem,label in BENCH.PROBLEMS.items():
        vals=[]
        for variant in VARIANTS:
            group=[r for r in rows if r['problem']==problem and r['variant']==variant]
            vals.append(f"{statistics.median(r['regret'] for r in group):.6g}"+('†' if any(r['error'] for r in group) else ''))
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines+=['','## Variability, accounting and movement diagnostics','',
            '| Problem | Method | Median error | IQR | Errors | Success ≤1e-6 | Evaluations min–max | CPU seconds median | Local models median | Independent families/model median | Fallback models median |',
            '|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|']
    for problem,label in BENCH.PROBLEMS.items():
        for variant in VARIANTS:
            group=[r for r in rows if r['problem']==problem and r['variant']==variant]
            vals=[r['regret'] for r in group];q1,_,q3=statistics.quantiles(vals,n=4,method='inclusive')
            diagnostics=[r['final_status'].get('exploration') or {} for r in group]
            def diag(key): return f"{statistics.median(d.get(key,0) for d in diagnostics):.3g}" if variant.startswith('Cloning') else '—'
            lines.append(f"| {label} | {variant} | {statistics.median(vals):.6g} | {q1:.4g}–{q3:.4g} | {sum(bool(r['error']) for r in group)} | {sum(v<=1e-6 for v in vals)}/10 | {min(r['evaluations'] for r in group)}–{max(r['evaluations'] for r in group)} | {statistics.median(r['cpu_seconds'] for r in group):.3f} | {diag('model_count')} | {diag('effective_parents')} | {diag('fallback_count')} |")
    lines+=['','## Reproduction','','```sh','python3 fractal-gas-web/tools/benchmark-cloning-guided.py --workers 8','```','',
            'The manifest fingerprints the engine, source files and runners. Full configurations and final diagnostics are retained in runs.jsonl. This experiment compares optimization variants; it does not establish Gibbs invariance or universal superiority.','']
    (output/'report.md').write_text('\n'.join(lines))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',default='tests/optimization/reports/cloning-guided-20d-100k');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[BENCH.LIBRARY,Path(__file__),Path(BENCH.__file__),*sorted((ROOT/'src/optimization').glob('*.*')),*sorted((ROOT/'src/fractal').glob('*.hpp'))]
    manifest={'dimensions':20,'walkers':128,'elites':5,'budget':100000,'seeds':10,'variants':VARIANTS,'workers':args.workers,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    rows=[]
    if args.resume:
        old=json.loads((out/'manifest.json').read_text());assert old['fingerprints']==manifest['fingerprints']
        rows=[json.loads(l) for l in (out/'runs.jsonl').read_text().splitlines()]
    elif (out/'runs.jsonl').exists(): raise RuntimeError('Choose a new output or --resume; existing data is not overwritten')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cases=[(problem,20,variant,seed,100000) for problem in BENCH.PROBLEMS for variant in VARIANTS for seed in range(10)]
    done={(r['problem'],r['dimensions'],r['variant'],r['seed'],r['budget']) for r in rows};cases=[c for c in cases if c not in done];random.Random(20260926).shuffle(cases)
    with (out/'runs.jsonl').open('a') as f, ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run,c) for c in cases]):
            row=future.result();rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            print(f"{len(rows)}/420 {row['problem']} {row['variant']} seed={row['seed']} error={row['regret']:.6g} {row['error']}",flush=True)
    assert len(rows)==len({(r['problem'],r['variant'],r['seed']) for r in rows})==420
    for p,h in manifest['fingerprints'].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    rows.sort(key=lambda r:(r['problem'],r['variant'],r['seed']));report(out,rows)

if __name__=='__main__': main()
