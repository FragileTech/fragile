# Euclidean Gas Lab

The **Elite walkers** setting retains the best walkers found after completed
steps. Fresh lab runs start with two elites; choose zero to disable retention,
or any integer up to the walker count. Use **Apply & reset** after changing it.

Before each subsequent step, elites return to the first population slots. They
are protected from cloning, remain available as donors, and still undergo normal
motion. After each step, the updated best-ever bank is copied back into the
first slots in rank order. Ranking uses the objective's minimize/maximize direction, with older
elites winning ties. The retained bank stores historical scores, including for
stochastic objectives.

Configuration exports save the count as `gas.n_elite`; checkpoints also save the
bank for deterministic continuation. Imports without `n_elite` use zero. Existing
theory presets and the Rust API also default to zero.

The lab can continue when the active population has no eligible walkers but
the elite bank still contains survivors to restore on the next step.

Validation: run `npm run test:euclidean-gas` after rebuilding the engines,
then serve `web/` and run `npm run test:euclidean-gas-browser`. The optional
`node tests/euclidean-gas/elites-webgpu-browser.mjs` check exercises elite
protection and checkpoint continuation on WebGPU and reports a skip when no
adapter is available. Set `LECTURE_BASE_URL` if the server is not on port 8770.
