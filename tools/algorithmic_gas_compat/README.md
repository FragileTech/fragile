# Algorithmic Gas compatibility tools

These optional tools use Fragile's Python or native C++ implementations to export
reference data. Normal Fragile and Algorithmic Gas builds do not call them or require
another checkout. Run them from the Fragile repository and supply an output directory:

```sh
uv run python tools/algorithmic_gas_compat/export_qft_fixtures.py --output /tmp/qft-fixtures
uv run python tools/algorithmic_gas_compat/export_tessellation_fixtures.py --output /tmp/tessellation-fixtures
make optimization-native
uv run python tools/algorithmic_gas_compat/generate_objective_goldens.py --output /tmp/objective-fixtures
uv run python tools/algorithmic_gas_compat/eh_archive_to_history.py run.json history.pt
```

Review generated fixtures before copying them into the independent Algorithmic Gas
repository. Its normal tests consume the checked-in JSON references.
