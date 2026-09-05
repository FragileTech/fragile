# Publication validation

Scope: manuscript sources, documentation tooling, and generated HTML.

| Check | Result |
|---|---|
| Jupyter Book and MyST navigation | Same 101 published pages, in the same order |
| Source references and retired terminology | Pass |
| Original labelled formal items | 1,326 retained or mapped to a consolidation target |
| Mathematical downloads | Eight current files for the two volumes; archives excluded |
| Fractal Gas formal export | 2,163 blocks from 41 source pages, including proofs |
| Compatibility redirects | All 45 destinations and mapped fragments pass |
| Built proof targets and Expert Mode visibility | Pass |
| Built search and removed-volume cleanup | Pass |
| Documentation tests with documentation dependencies | 8 passed |
| Documentation tests in the base environment | 7 passed; optional Sphinx test skipped |
| Ruff checks and formatting for documentation tools/tests | Pass |
| Project documentation copies and tracked whitespace check | Pass |

The clean Jupyter Book build succeeded with 165 warnings: 132 unavailable DOI
previews, 29 existing Volume 1 warnings, and four saved notebook MIME-output
warnings. The final incremental refresh succeeded with 20 DOI-preview warnings
and no Volume 2 theorem, directive, or cross-reference warnings. The complete
HTML audit passed after that refresh.

Commands used:

```sh
make prompt
UV_CACHE_DIR=/tmp/fragile-docs-uv uv run --offline --no-sync python docs/check_book.py
UV_CACHE_DIR=/tmp/fragile-docs-uv uv run --offline --no-sync --with-requirements docs/requirements.txt jupyter-book clean docs/ --all
UV_CACHE_DIR=/tmp/fragile-docs-uv uv run --offline --no-sync --with-requirements docs/requirements.txt jupyter-book build docs/
UV_CACHE_DIR=/tmp/fragile-docs-uv uv run --offline --no-sync --with-requirements docs/requirements.txt python docs/check_built_book.py
UV_CACHE_DIR=/tmp/fragile-docs-uv uv run --offline --no-sync --with-requirements docs/requirements.txt pytest -q tests/docs/test_book_tools.py
```

The mathematical recovery notes record the analytical corrections and numerical
spot checks separately. Current theorem statements specify their law, operator,
boundary, regularity, and uniformity hypotheses.
