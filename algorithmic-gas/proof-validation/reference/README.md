# Frozen Python references

These files are provenance snapshots used by proof-validation programs. They are not a
Python implementation of Algorithmic Gas and are not imported by the Rust or browser
runtimes. Their source is the corresponding file in `FragileTech/fragile` at the commit
where the repository boundary was created.

Large QFT and tessellation parity fixtures remain checked-in under the Rust tests. Their
optional regeneration scripts stay with the Python implementation in
`FragileTech/fragile/tools/algorithmic_gas_compat/`; normal builds consume only the JSON
fixtures and do not require a sibling checkout.
