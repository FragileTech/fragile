# Repository Guidelines

## Scope

This repository owns the Algorithmic Gas Rust workspace, Volume 2 of the Fractal Gas
book, its proof and validation programs, and the Rust-backed browser laboratory. The
Python Fractal Gas engines and the C++ Arcade, Control, LLM, and Optimization labs live
in `FragileTech/fragile` and are not dependencies of normal builds.

## Development

- Run Rust commands from the repository root with the toolchain pinned in
  `rust-toolchain.toml`.
- Keep book statements and Rust behavior aligned. Rust source-linked tests embed the
  chapters under `docs/source/2_fractal_gas/`.
- Preserve vectorized, deterministic execution and the existing archive and WASM wire
  formats unless a change explicitly includes a migration.
- Keep complete proofs and analytic hypotheses visible in Expert Mode. Distinguish
  finite-particle chains, quasi-stationary laws, invariant measures, mean-field limits,
  and continuum models.
- Follow `docs/CLAUDE.md` for documentation style. Use the Feynman educator agent for
  new explanatory prose.

## Verification

Run `make check` for Rust formatting, Clippy, tests, and documentation source checks.
Run `make web-test` for the WASM lab and its non-browser JavaScript suites, `make docs`
for the standalone Volume 2 book, and `make browser-test` with `make serve` running for
browser integration coverage.

Python parity fixtures are checked-in reference data. Regenerating them is an explicit
cross-repository operation documented in `proof-validation/reference/README.md`; normal
CI must not import `fragile` from a parent checkout.
