import importlib.util,json
from pathlib import Path
p=Path('/home/guillem/fragile/fractal-gas-web/tools/benchmark-long-budget.py')
s=importlib.util.spec_from_file_location('bench',p); b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
out=b.ROOT/'tests/optimization/reports/fractal-populations-followup-20d-100k'
out.mkdir(exist_ok=True)
with (out/'single-gaussian.jsonl').open('x') as f:
 for problem in b.PROBLEMS:
  for seed in range(5):
   row=b.run_case((problem,20,'Gaussian',seed,100000),{'walkers':320,'max_walkers':320,'controller_enabled':False,'population_auto':False,'scale_auto':False})
   f.write(json.dumps(row)+'\n');f.flush()
   print(problem,seed,row['regret'],row['error'],flush=True)
