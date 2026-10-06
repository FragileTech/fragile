# History-preserving extraction

This directory is intentionally an ordinary directory until the boundary change lands.
The eventual `FragileTech/algorithmic-gas` repository should be created from a fresh
clone of the parent with `git filter-repo`. Do not use `git subtree split`: much of the
owned history predates the `algorithmic-gas/` directory and would otherwise be lost.

## Extraction procedure

1. Commit the boundary move in the parent repository so that the current tree is one
   coherent extraction point.
2. Make a fresh clone of the parent. Never run the history rewrite in a working clone.
3. Translate every entry in `history/path-map.txt` into a `--path` selector and, when
   the destination differs, a `--path-rename OLD:NEW` argument. The first rule,
   `algorithmic-gas/ -> /`, lifts the present subtree to the new repository root.
4. Run `git filter-repo --force` with all selectors and renames in one invocation. A
   schematic command is:

   ```console
   git filter-repo --force \
     --path algorithmic-gas/ \
     --path docs/source/2_fractal_gas/ \
     --path fractal-gas-web/web/euclidean-gas/ \
     --path-rename algorithmic-gas/: \
     --path-rename fractal-gas-web/web/euclidean-gas/:web/euclidean-gas/
   ```

   The abbreviated command is illustrative; the real invocation must include every
   entry in the path map, including files that already have the desired destination.
5. Compare the rewritten `HEAD` tree with this directory, inspect `git log --follow`
   for representative Rust, Volume II, proof-validation, and browser-lab files, and
   verify author and tag metadata.
6. Push the rewritten history to the empty `FragileTech/algorithmic-gas` repository,
   enable its CI and Pages deployment, and verify the published site at
   `https://fragiletech.github.io/algorithmic-gas/`.
7. In a later parent-repository change, replace this directory with a submodule pinned
   to the verified extraction commit.

The path map is historical provenance, not an instruction to reintroduce parent-only
shared infrastructure. Files copied into the child solely to make it standalone begin
their child history at the boundary commit.
