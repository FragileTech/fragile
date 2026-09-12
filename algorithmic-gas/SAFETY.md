# Safety and robustness

The three production workspace crates inherit `unsafe_code = "forbid"`. This also covers their examples/tests and prevents introducing explicit unsafe blocks, implementations or attributes. CI runs all-feature Clippy and the contract/regression tests.

This is not an unconditional end-to-end memory-safety guarantee. Burn Flex uses unsafe SIMD internals; CUDA/WGPU use native APIs and drivers. Safe callers rely on those abstractions being sound. No test suite proves all executions safe. Panics, arithmetic errors and allocation exhaustion are also distinct from memory corruption.

## Data and checkpoint boundaries

- `TensorBatch` has private shape/storage fields and validates during construction **and deserialization**. Gather checks sizes/indices before reserving storage and uses fallible reservation.
- Restore validates population/configuration, nested report shapes and masks, source frame/slot/version/generation identities, reciprocal matching, clone decisions, counters and provenance before committing.
- CBOR decode has a 256 MiB encoded-size limit, a 64-level recursion limit, and rejects trailing bytes. The browser applies the same archive validation before preparing a restored engine. These are structural checks, not authentication of a checkpoint's origin.
- `max_memory_bytes` reserves a conservative engine working set, including multiple populations, source pools, reports, archive frames and input copies. Each numerical graph additionally budgets all live nodes and staging buffers, not merely its largest tensor. Noise batches are checked before materialization.
- Memory reservations do not cap the entire process, arbitrary custom callbacks, decoder expansion, allocator overhead or a driver's internal workspace. Use process/container memory limits and bounded browser inputs for untrusted workloads. A recoverable allocation error cannot be promised after the allocator or driver has terminated a process/worker.

## Additional checks

From this directory:

```sh
cargo install cargo-audit --version 0.22.2 --locked
cargo audit --file Cargo.lock --deny unsound

rustup toolchain install nightly --component miri
cargo +nightly miri test -p algorithmic-gas --test review_regressions tensor_deserialization_validates_shapes_without_panicking
cargo +nightly miri test -p algorithmic-gas --test review_regressions local_statistics_exclude_self_and_record_singleton_fallback

cargo install cargo-fuzz --version 0.13.2 --locked
mkdir -p fuzz/corpus/decode
cp fuzz/seeds/*.json fuzz/corpus/decode/
cargo +nightly fuzz run decode -- -max_total_time=60 -max_len=65536 -rss_limit_mb=1024
```

The Miri tests exercise pure data/statistics code, not GPU execution or unsupported native SIMD paths. The host fuzz harness uses AddressSanitizer and checks tensor/population JSON plus CBOR checkpoint decoding. It has its own pinned lockfile. A short fuzz run is a smoke check; longer continuous campaigns and GPU hardware testing remain useful.

## Dependency audit

The 2026-09-12 audit found no reported vulnerabilities or unsoundness advisories for the production lockfile. It reports two **unmaintained** transitive dependencies: `bincode 2.0.1` through `burn-core` (RUSTSEC-2025-0141), and `paste 1.0.15` through Flex's `gemm`/`pulp` stack (RUSTSEC-2024-0436). These warnings are visible, not suppressed. They require tracking upstream replacements; an advisory scan is not a source-level dependency proof. CI fails on vulnerabilities and unsoundness and prints maintenance warnings.
