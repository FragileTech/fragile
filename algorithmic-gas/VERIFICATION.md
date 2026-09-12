# Implementation verification

## Review fixes: 2026-09-12

The checks below were rerun after the correctness and safety fixes. Documentation publication and full browser UI checks in the initial-verification section were not rerun for this patch.

| Check | Result |
|---|---|
| `cargo test --workspace --offline --locked` | 55 contract/integration tests passed, including 13 new regression tests |
| `cargo clippy --workspace --all-targets --all-features --offline --locked -- -D warnings` | Passed; all production crates forbid unsafe code |
| Workspace and standalone fuzz-harness formatting | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --offline --locked` | Passed |
| CPU and WebGPU release WASM bundles | Rebuilt successfully with Rust 1.95.0 and wasm-bindgen 0.2.114 |
| `npm --prefix fractal-gas-web run test:euclidean-gas` | 12 tests passed against the rebuilt bindings |
| Miri: tensor deserialization and local-statistics regression tests | Both passed on nightly 2026-09-12 |
| AddressSanitizer/libFuzzer decode smoke test | 3,779,909 executions in 61 seconds; no crash reported; input limit 64 KiB |
| `cargo audit --file Cargo.lock --deny unsound` | No reported vulnerabilities or unsoundness; two unmaintained-dependency warnings remain |

The regressions cover squared-distance reductions, scale-safe cosine, local self exclusion and fallback, insufficient donor padding, linear uniform sampling, deserialization invariants, aggregate allocation budgets, post-clone boundary timing, replacement provenance, malformed nested checkpoints, and extinction before remaining kinetic/provider calls.

### Sampler scaling diagnostic

`cargo run -p algorithmic-gas --example sampler_scaling --offline --locked` measured 30 draws per size with one companion per walker. Setup is excluded. This debug-build CPU diagnostic checks scaling, not absolute performance or GPU speedup.

| Walkers | Uniform independent | Fisher–Yates mutual |
|---:|---:|---:|
| 2,000 | 13.086 ms | 12.830 ms |
| 4,000 | 21.028 ms | 25.474 ms |
| 8,000 | 44.665 ms | 50.596 ms |
| 16,000 | 86.594 ms | 98.551 ms |

The measured scaling is consistent with the corrected linear paths. Gaussian sampling and other matching laws have separate costs; these numbers do not describe full engine steps.

Checkpoint schema 2 and built-in operator identity v2 distinguish the corrected semantics. Schema-1 checkpoints are rejected, not silently migrated. GPU runtime behavior remains unverified. See [SAFETY.md](SAFETY.md) for dependency warnings, resource-limit scope and why these checks are not an unconditional memory-safety guarantee.

## Initial implementation

Local verification on 2026-09-12. This records evidence, not a guarantee of accelerator parity or completion of every architecture target.

| Check | Result |
|---|---|
| `cargo test --workspace --locked` (cached dependencies) | 42 contract/integration tests passed |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | Passed, including CUDA/WGPU adapters |
| `cargo fmt --all -- --check` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` | Passed |
| CPU and WebGPU release WASM bundles | Built with Rust 1.95.0 and wasm-bindgen 0.2.114 |
| `npm --prefix fractal-gas-web run test:euclidean-gas` | 8 tests passed; f32/f64 stepping, replay, configuration, objectives and Langevin |
| Existing Optimization Lab tests | 22 passed outside the sandbox; its Python subprocess calls timed out inside the sandbox |
| Documentation publication unit tests | 14 passed |
| Source/navigation and built-book checks | Passed |
| Theory build | Succeeded; 116 book-wide Sphinx warnings, not a warning-free build |
| Native CPU example | 64 walkers, f64 Rastrigin, seed 7, 10 steps; finite output and resolved execution statistics |

Browser checks exercised initialized/paused populations, step and run/pause, f32/f64, multiple distance companions, phase-space BAOAB with anisotropic noise, IndexedDB save/restore, and explicit WebGPU/f64 errors. Desktop and 390px/320px layouts were visually reviewed. All six architecture Mermaid diagrams rendered, and Expert Mode retained the implementation notes and formal content.

No usable WebGPU adapter was returned by this browser. GPU initialization failed explicitly and preserved the prior CPU run. CUDA runtime execution was not verified. The accelerator implementation is host-orchestrated with counted transfers; device-resident performance and hardware numerical/distribution comparisons remain future acceptance work. See [README.md](README.md) for the implemented API and remaining targets.
