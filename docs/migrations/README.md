# Volume 2 migration records

These records are excluded from the published book and prompt downloads.

- `volume2_inventory.json` records the 84 original Fractal Gas source files,
  hashes, publication destinations, supplementary proofs, and recovery of
  historical proof files that were missing from the working tree.
- `proof_label_dispositions.json` traces the 1,326 originally labelled formal
  items in canonical chapters and supporting proof files to their current
  destinations. Retaining a label does not assert that an old statement is
  unchanged; the revised statement specifies its mathematical scope.
- `analytical_recovery.md` records the recovered arguments and corrections.
- `kl_recovery.md` and `kl_label_map.json` document the functional-inequality
  and entropy arguments, including the distinction between reference laws,
  conservative invariant laws, and the normalized killed evolution.
- `geometric_gas_recovery.md` records the coefficient and geometric proofs.
- `validation.md` records the final source, build, HTML, and tooling checks.

Publication validation uses `docs/check_book.py` and `docs/check_built_book.py`.
The latter checks the built HTML, including proof targets and Expert Mode
visibility. Regenerate downloads with `make prompt` before the final build.
A clean build is required to remove obsolete pages and search-index terms.
