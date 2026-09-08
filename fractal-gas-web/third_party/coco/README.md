# COCO benchmark engine

Pinned upstream: [numbbo/coco-experiment v2.8.2](https://github.com/numbbo/coco-experiment/tree/e5d068f69e36f346c86cc2934413369abe36fc22), commit `e5d068f69e36f346c86cc2934413369abe36fc22`.

`coco.c` expands the local includes of upstream `coco_random.c`, `coco_suite.c`,
`coco_observer.c`, `coco_archive.c`, and `coco_runtime_c.c` in that order, once
per file, following `scripts/fabricate`. The generated `coco_version` is `2.8.2`.
No objective formulas, transformations, instance generators, or optimizers are
modified. `upstream-files.json` records SHA-256 hashes of the original inputs.
Regenerate with `python3 fractal-gas-web/tools/vendor-coco.py /path/to/upstream`.

Optimization Lab uses the public C suite API for the 24 noiseless BBOB functions.
The C++ ownership adapter is in `src/optimization/coco_benchmark.cpp`. Display and
optimization have separate problem objects, and optimum inspection taints only
the display object. No official COCO observer data is produced by this app.

The upstream license is BSD-3-Clause with exceptions listed in `LICENSE` and
`AUTHORS`. In particular, the included AVL implementation is LGPL-3.0-or-later;
`COPYING` and `COPYING.LESSER` contain its license texts. Source and the CMake
build needed to rebuild/relink the application are provided in this repository.
The web build copies these notices into `optimization/vendor/`.
`NOTICE` combines the upstream license and authors with the third-party copyright
headers in `brentq.c` and `mo_avl_tree.c` for the browser's license link.

The WASM target reserves a 1 MiB stack with overflow checks. Upstream rotation
generation nests 48 KiB and 16 KiB local scratch arrays, exceeding Emscripten's
default stack once caller frames are included.

`tests/optimization/coco-fixtures.json` contains a compact subset of upstream
`src/bbob2009_testcases.txt`: all 24 functions, six official dimensions, instances
1, 2, and 15, and the first three test vectors. Values retain upstream precision.
