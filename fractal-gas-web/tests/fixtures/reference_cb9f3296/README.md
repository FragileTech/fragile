# Reference implementation of the tree ("Graph") algorithm

Verbatim copies of the Python files at git commit `cb9f3296` ("Update
requirements", 2025-02-16) that implement the FractalTree / MontezumaTree
algorithm the C++ port in `src/fractal_tree.*` reproduces:

- `fragile/core.py`        — BaseFractalTree, FractalTree, get_is_cloned, get_is_leaf, RandomPolicy
- `fragile/fractalai.py`   — relativize, random_alive_compas, calculate_clone, clone_tensor
- `fragile/videogames.py`  — MontezumaTree (visit counting), aggregate_visits users
- `fragile/actions.py`     — UniformDtSampler
- `fragile/random_state.py`

Only two modules are stubs, because core.py imports symbols from them that the
algorithm never uses: `fragile/benchmarks.py` (Rastrigin) and
`fragile/utils.py` (which keeps the verbatim `numpy_dtype_to_torch_dtype`).
`generate_tree_fixtures.py` prepends this directory to `sys.path`, so
`import fragile` resolves to this package and the recorded RNG draws and
expected values come from the historical algorithm itself, not from a
transcription. Regenerate with `git show cb9f3296:src/fragile/<file>.py`.
