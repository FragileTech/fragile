# Recovered mean-field proof sources

The archive contains the exact pre–September 5 proof baseline from commit
`b416d3a98112b360e6ae7f6655fb47497916cfcb`: 35 source files, including the
convergence chapters through Chapter 15, their standalone proof directory,
and the separate geometric LSI argument. `manifest.json` records every source
path, content hash, and 805 formal labels with their original line numbers.

These are engineering reference sources. The published lectures remain a
single presentation in `docs/source/2_fractal_gas/convergence_program/`.

Reproduce the archive from Git with:

```bash
python3 algorithmic-gas/tools/recover_mean_field_proofs.py
```

The recovered sources are compared and repaired claim by claim in
[`MEAN_FIELD_AUDIT.md`](../MEAN_FIELD_AUDIT.md). Recovery preserves the text;
it does not certify every recovered assertion. Formal statements are
integrated into the lectures only with their corresponding proof review.
