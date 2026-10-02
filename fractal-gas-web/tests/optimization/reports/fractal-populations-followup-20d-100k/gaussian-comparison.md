# Original Gaussian population versus one Gaussian swarm

20D, five seeds, 100,000 total evaluations per run. Ten × 32 walkers versus one × 320 walkers; five elites per swarm. Population uses ten scales from 0.0001 to 0.2; single swarm uses the established 0.2 baseline. This comparison changes scale diversity and total protected elites as well as partitioning/exchange; it does not isolate migration alone.

| Problem | Population median error | Single median error | Population paired wins |
|---|---:|---:|---:|
| quadratic | 5.12736e-09 | 0.0212669 | 5/5 |
| bbob_10 | 6362.45 | 13974.5 | 3/5 |
| rastrigin | 79.5965 | 105.669 | 4/5 |
| bbob_15 | 165.163 | 113.423 | 2/5 |
| rosenbrock | 18.8594 | 57.4423 | 5/5 |
| bbob_5 | 83.1354 | 47.7299 | 0/5 |
