# Volume 3 Proof Obligations

This document captures the remaining proof obligations inside the Volume 3 framework with repo-local references.

## Remaining obligations

- Replace the Euclidean lattice Riemann-sum step with a Fractal Set sampling limit driven by `thm-propagation-chaos-qsd` plus `mt:emergent-continuum`/`mt:continuum-injection`/`mt:cheeger-gradient`, and make the confining envelope explicit for R^4; see `docs/source/3_fractal_gas/2_fractal_set/05_yang_mills_noether.md:1596`, `docs/source/2_hypostructure/10_information_processing/02_fractal_gas.md:1115`, `docs/source/2_hypostructure/10_information_processing/02_fractal_gas.md:1378`, `docs/source/3_fractal_gas/appendices/07_discrete_qsd.md:386`.
- Prove the graph-Laplacian/Dirichlet-form convergence for the Fractal Set rather than leaving it as “requires proof,” using the conditional metatheorems and QSD sampling; see `docs/source/3_fractal_gas/2_fractal_set/03_lattice_qft.md:721`, `docs/source/2_hypostructure/10_information_processing/02_fractal_gas.md:1115`, `docs/source/2_hypostructure/10_information_processing/02_fractal_gas.md:1570`.
- Close the LSI dependency loop inside Volume 3 (no external missing volumes); `thm-n-uniform-lsi-exchangeable` currently defers to the KL proof while `15_kl_convergence.md` cites non-repo sources, so the hypocoercive proof needs to live here; see `docs/source/3_fractal_gas/appendices/12_qsd_exchangeability_theory.md:573`, `docs/source/3_fractal_gas/appendices/15_kl_convergence.md:27`.
- Upgrade the equilibrium flux-balance to a pointwise stationarity result (so the cloning current vanishes at QSD and generalized detailed balance holds for equilibrium correlators); see `docs/source/3_fractal_gas/appendices/07_discrete_qsd.md:337`, `docs/source/3_fractal_gas/appendices/08_mean_field.md:654`.
- Make locality exact in the continuum limit by controlling Gaussian-kernel tails (for example, `epsilon,rho -> 0` scaling with LSI concentration or causal-observable restriction), not just “spacelike” IG intuition; kernel definition at `docs/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent.md:267`.
- Tie the algorithmic spectral gap to the OS-reconstructed Hamiltonian gap (and show it persists under `tau,rho -> 0`), rather than relying on reversibility of the full generator; see `docs/source/3_fractal_gas/2_fractal_set/05_yang_mills_noether.md:2725`.
