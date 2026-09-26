"""Isolated 2x2 test of clone-field drift normalization and covariance shrinkage.

Builds experimental libraries from the tested native build without editing the
production engine. Every completed run is flushed; manifests fingerprint inputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("long_budget", Path(__file__).with_name("benchmark-long-budget.py"))
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
VARIANTS = {
    "current": (False, .1),
    "normalized_drift": (True, .1),
    "weak_regularization": (False, .001),
    "both": (True, .001),
}
PROBLEMS = ["quadratic", "bbob_2", "bbob_10", "rosenbrock", "bbob_5", "bbob_15", "bbob_24"]
for variant in VARIANTS:
    BENCH.VARIANTS[variant] = {"perturbation": "cloning_guided", "boundary": "cma"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_one(name, output, build):
    normalized, shrink = VARIANTS[name]
    folder = output / "builds" / name
    folder.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "src/optimization/cloning_guided.cpp").read_text()
    source = source.replace("constexpr double radius=.1, shrink=.1;", f"constexpr double radius=.1, shrink={shrink!r};")
    if normalized:
        source = source.replace("double total=0,field_weight=0;", "double total=0,field_weight=0,field_length=0;")
        source = source.replace("field_weight+=weight*std::abs(response);", "field_weight+=weight*std::abs(response);\n      field_length+=weight*std::abs(response)*direction.norm()/std::sqrt(double(d));")
        source = source.replace("if(field_weight>0) m.field=field/(radius*field_weight);", "if(field_length>1e-15) m.field=field/field_length;")
    cpp = folder / "cloning_guided.cpp"
    cpp.write_text(source)
    obj = folder / "cloning_guided.cpp.o"
    includes = ["-I" + str(ROOT / "src")]
    for path in [ROOT / "third_party/eigen", ROOT / "third_party/lbfgspp/include", ROOT / "third_party/libcmaes/include", ROOT / "third_party/coco", build / "optimization/cma-generated"]:
        includes += ["-isystem", str(path)]
    command = ["/usr/bin/c++", "-O3", "-DNDEBUG", "-std=gnu++17", "-fPIC", "-ffp-contract=off", "-DEIGEN_DONT_PARALLELIZE", "-DEIGEN_DONT_VECTORIZE", "-DEIGEN_MPL2_ONLY", *includes, "-c", str(cpp), "-o", str(obj)]
    commands = [command]
    subprocess.run(command, check=True)
    archive = folder / "libfg_optimization_core.a"
    shutil.copy2(build / "optimization/libfg_optimization_core.a", archive)
    subprocess.run(["ar", "r", str(archive), str(obj)], check=True)
    commands.append(["ar", "r", str(archive), str(obj)])
    lib = folder / "libfg_optimization.so"
    command = ["/usr/bin/c++", "-shared", "-o", str(lib), str(build / "optimization/CMakeFiles/fg_optimization.dir/__/src/optimization/c_api.cpp.o"), str(archive), str(build / "libfg_swarm_core.a"), str(build / "libfg_fractal_core.a"), str(build / "optimization/libfg_coco.a"), "-lm", str(build / "optimization/libfg_libcmaes.a")]
    subprocess.run(command, check=True)
    commands.append(command)
    (folder / "commands.json").write_text(json.dumps(commands, indent=2))
    return name, str(lib)


def run_case(task):
    problem, variant, seed, budget, library = task
    BENCH.LIBRARY = Path(library)
    row = BENCH.run_case((problem, 20, variant, seed, budget), {"walkers": 128, "max_walkers": 128, "controller_enabled": False}, target_error=1e-5)
    diagnostics = row["final_status"]["exploration"]
    assert diagnostics["weighting"] == "signed_clone_score"
    assert diagnostics["drift_noise_ratio"] <= .250001
    assert diagnostics["condition_number"] <= 1000001
    return row


def probe(library):
    lib = ctypes.CDLL(str(library))
    lib.fgo_create.argtypes = [ctypes.c_char_p]; lib.fgo_create.restype = ctypes.c_uint32
    lib.fgo_step.argtypes = [ctypes.c_uint32]; lib.fgo_step.restype = ctypes.c_int
    lib.fgo_snapshot.argtypes = [ctypes.c_uint32]; lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
    lib.fgo_snapshot_size.argtypes = [ctypes.c_uint32]; lib.fgo_snapshot_size.restype = ctypes.c_uint32
    lib.fgo_destroy.argtypes = [ctypes.c_uint32]
    result = []
    for problem in PROBLEMS:
        cfg = {"algorithm":"wave", "benchmark":problem, "dimensions":20, "walkers":128, "max_walkers":128, "elites":5, "seed":3, "perturbation":"cloning_guided", "boundary":"cma", "adaptive_min_scale":.0001, "adaptive_max_scale":.2}
        h = lib.fgo_create(json.dumps(cfg).encode()); assert h
        try:
            for _ in range(20): assert lib.fgo_step(h) >= 0
            result.append(list(lib.fgo_snapshot(h)[:lib.fgo_snapshot_size(h)]))
        finally: lib.fgo_destroy(h)
    return result


def report(output, rows, expected):
    lines = ["# Clone-field drift/covariance ablation", "", f"{len(rows)}/{expected} runs completed; {sum(bool(r['error']) for r in rows)} execution errors.", "", "Seven 20D functions, 128 walkers, five elites, seeds 0–4, CMA boundary repair, no restarts. Up to 1,000,000 evaluations or absolute error ≤1e-5. Scale bounds [0.0001, 0.2], drift strength 0.25. Mixed precision: FP32 walkers/fitness, FP64 objectives/covariance.", "", "A predeclared 2×2 comparison: drift denominator is either fixed radius × summed absolute score response (current), or summed absolute response × comparison RMS length (normalized drift). The latter removes contraction-dependent drift damping while retaining directional cancellation. Covariance identity regularization is either 10% (current) or 0.1% (weak). All other settings and random seeds match. Production engine is unchanged.", "", "Baseline rebuilt with the same pipeline reproduces the production snapshots exactly on seven problems after 20 steps. Each variant is also checked for evaluation bounds and finite, bounded movement diagnostics. Final diagnostics describe the maximum over local models, including retained inactive models.", "", "| Problem | Variant | Runs | Targets | Median error | Median evaluations | Median drift/noise |", "|---|---|---:|---:|---:|---:|---:|"]
    for p in PROBLEMS:
        for v in VARIANTS:
            g = [r for r in rows if r['problem']==p and r['variant']==v]
            if g:
                lines.append(f"| {p} | {v} | {len(g)} | {sum(r['target_reached'] for r in g)} | {statistics.median(r['regret'] for r in g):.6g} | {statistics.median(r['evaluations'] for r in g):g} | {statistics.median(r['final_status']['exploration']['drift_noise_ratio'] for r in g):.5g} |")
    lines += ["", "## Matched comparisons against current", "", "Both-target cases are compared by evaluations separately; sub-tolerance objective differences are not wins.", "", "| Variant | Matched | Lower error | Higher error | Both targets | Current/new targets | Median new/current evaluations when both solve |", "|---|---:|---:|---:|---:|---|---:|"]
    baseline = {(r['problem'],r['seed']):r for r in rows if r['variant']=='current'}
    for v in list(VARIANTS)[1:]:
        pairs=[(baseline[(r['problem'],r['seed'])],r) for r in rows if r['variant']==v and (r['problem'],r['seed']) in baseline]
        both=[(a,b) for a,b in pairs if a['target_reached'] and b['target_reached']]
        lower=sum(b['regret']<a['regret'] and not(a['target_reached'] and b['target_reached']) for a,b in pairs)
        higher=sum(b['regret']>a['regret'] and not(a['target_reached'] and b['target_reached']) for a,b in pairs)
        speed=f"{statistics.median(b['evaluations']/a['evaluations'] for a,b in both):.3g}" if both else '—'
        lines.append(f"| {v} | {len(pairs)} | {lower} | {higher} | {len(both)} | {sum(a['target_reached'] for a,b in pairs)}/{sum(b['target_reached'] for a,b in pairs)} | {speed} |")
    (output / "report.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=Path('/tmp/fragile-clone-score-native'))
    parser.add_argument('--output',type=Path,default=ROOT/'tests/optimization/reports/clone-field-factorial-20d-1m')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args(); output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'manifest.json').exists(): raise RuntimeError('Existing experiment; choose a new output')
    with ThreadPoolExecutor(max_workers=3) as pool:
        libraries=dict(pool.map(lambda v:build_one(v,output,args.build.resolve()),VARIANTS))
    assert probe(libraries['current'])==probe(ROOT/'build-optimization-native/optimization/libfg_optimization.so'), 'Rebuilt baseline differs from production'
    for v,lib in libraries.items():
        smoke=run_case(('quadratic',v,0,4096,lib)); assert not smoke['error'],smoke['error']
    inputs=[Path(__file__),Path(BENCH.__file__),*sorted((ROOT/'src').rglob('*.cpp')),*sorted((ROOT/'src').rglob('*.hpp')),*sorted((ROOT/'src').rglob('*.h')),*[Path(v) for v in libraries.values()],*[output/'builds'/v/'cloning_guided.cpp' for v in VARIANTS]]
    fingerprints={str(p):digest(p) for p in inputs}
    manifest={'dimensions':20,'walkers':128,'elites':5,'seeds':list(range(5)),'budget':1000000,'target':1e-5,'problems':PROBLEMS,'variants':VARIANTS,'libraries':libraries,'fingerprints':fingerprints,'expected_runs':140,'baseline_probe':'exact on seven problems at 20 steps'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tasks=[(p,v,s,1000000,libraries[v]) for p in PROBLEMS for v in VARIANTS for s in range(5)]
    random.Random(20260926).shuffle(tasks);rows=[];report(output,rows,len(tasks))
    print('Builds and baseline parity verified; starting 140 runs',flush=True)
    with (output/'runs.jsonl').open('w') as stream, ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_case,t) for t in tasks]):
            row=future.result();rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(f"{len(rows)}/{len(tasks)} {row['problem']} {row['variant']} seed={row['seed']} error={row['regret']:.6g} evaluations={row['evaluations']} {row['error']}",flush=True)
            report(output,rows,len(tasks))
    for path,value in fingerprints.items(): assert digest(Path(path))==value,path
    (output/'COMPLETE.json').write_text(json.dumps({'runs':len(rows),'errors':sum(bool(r['error']) for r in rows),'fingerprints_verified':True},indent=2)+'\n')

if __name__=='__main__': main()
