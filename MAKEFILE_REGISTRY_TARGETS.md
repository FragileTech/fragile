# Makefile Registry Targets

This document describes the new Make targets for building mathematical registries from enriched data.

## Available Targets

### `make transform-enriched`
**Purpose:** Transform all `refined_data/` directories to optimized `pipeline_data/` format.

**What it does:**
- Converts enriched JSON data (output from document-refiner agent) to pipeline-ready format
- Creates optimized data structures for fast registry loading
- Caches transformation results in `pipeline_data/` directories

**Current coverage:**
- Chapter 1: `01_fragile_gas_framework`

**When to use:**
- After running document refinement workflow
- Once per chapter (results are cached)
- When enriched data is updated

**Example output:**
```
Transforming refined_data to pipeline_data format...
Chapter 1: Euclidean Gas Framework
✓ Transformation complete
```

---

### `make build-registry`
**Purpose:** Build unified `combined_registry/` from all chapters.

**What it does:**
- Automatically discovers all `refined_data/` directories in `docs/source/`
- Builds complete mathematical registry with cross-references
- Creates unified index with all entities across chapters
- Validates relationships and dependencies

**Output location:** `combined_registry/`

**Output structure:**
```
combined_registry/
├── index.json              # Master index
├── axioms/                # All axioms
├── objects/               # Mathematical objects
├── parameters/            # Algorithm parameters
├── theorems/              # Theorems, lemmas, corollaries
└── relationships/         # Cross-references
```

**When to use:**
- To build complete registry for analysis
- For dashboard visualization
- For cross-chapter dependency analysis

**Example output:**
```
Building unified combined_registry from all chapters...
(Automatically discovers all refined_data directories)
✓ Combined registry built at combined_registry/
```

---

### `make build-chapter-registries`
**Purpose:** Build individual per-chapter registries.

**What it does:**
- Builds separate registry for each chapter from `pipeline_data/`
- Faster than combined registry (uses pre-transformed data)
- Useful for chapter-specific analysis

**Output locations:**
- `docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_registry/`

**Current coverage:**
- Chapter 1: `01_fragile_gas_framework`

**When to use:**
- For fast chapter-specific registry building
- For testing registry changes on single chapter
- When you only need one chapter's data

**Example output:**
```
Building individual chapter registries...
Chapter 1: Euclidean Gas Framework
✓ Chapter registries built
```

---

### `make build-all-registries` (Recommended)
**Purpose:** Full pipeline - transform enriched data and build unified registry.

**What it does:**
1. Runs `make transform-enriched`
2. Runs `make build-registry`

**When to use:**
- Complete registry building workflow
- After document refinement is complete
- When you want everything up-to-date

**Example output:**
```
Transforming refined_data to pipeline_data format...
Chapter 1: Euclidean Gas Framework
✓ Transformation complete

Building unified combined_registry from all chapters...
(Automatically discovers all refined_data directories)
✓ Combined registry built at combined_registry/
✓ Complete registry pipeline finished!
```

---

### `make clean-registries`
**Purpose:** Clean all registry outputs.

**What it does:**
- Removes `combined_registry/` and `refined_registry/`
- Removes all `pipeline_registry/` directories
- Does NOT remove `pipeline_data/` (cached transformations)

**When to use:**
- Before rebuilding registries from scratch
- To clean up disk space
- When testing registry building changes

**Example output:**
```
Cleaning registry outputs...
✓ Registry outputs cleaned
```

---

## Typical Workflows

### First-Time Setup (New Chapter)
```bash
# After running document-refiner agent
make transform-enriched      # Transform to pipeline format (once)
make build-registry          # Build unified registry
```

### Quick Rebuild
```bash
# If pipeline_data already exists
make build-registry          # Fast rebuild from cached data
```

### Complete Rebuild
```bash
make clean-registries        # Clean old outputs
make build-all-registries    # Full pipeline
```

### Chapter-Specific Work
```bash
make transform-enriched              # Transform specific chapter
make build-chapter-registries        # Build chapter registry
```

---

## Performance Characteristics

| Target | Speed | When to Use |
|--------|-------|-------------|
| `transform-enriched` | Slow (once) | After refinement, caches results |
| `build-registry` | Fast | Most common use case |
| `build-chapter-registries` | Fast | Chapter-specific analysis |
| `build-all-registries` | Slow (first time) | Complete workflow |
| `clean-registries` | Instant | Cleanup |

**Recommendation:** Use `make build-all-registries` for complete workflow. After first run, `make build-registry` is fast because it uses cached `pipeline_data/`.

---

## Adding New Chapters

When new chapters get `refined_data/`, update the Makefile:

### 1. Update `transform-enriched` target
```makefile
transform-enriched:
	@echo "Chapter 2: Geometric Gas"
	uv run python -m fragile.proofs.tools.enriched_to_math_types \
		--input docs/source/2_geometric_gas/<document>/refined_data \
		--output docs/source/2_geometric_gas/<document>/pipeline_data
```

### 2. Update `build-chapter-registries` target
```makefile
build-chapter-registries:
	@echo "Chapter 2: Geometric Gas"
	uv run python -m fragile.proofs.tools.build_pipeline_registry \
		--pipeline-dir docs/source/2_geometric_gas/<document>/pipeline_data \
		--output docs/source/2_geometric_gas/<document>/pipeline_registry
```

**Note:** `make build-registry` automatically discovers new chapters - no changes needed!

---

## Troubleshooting

### Error: "No refined_data found"
**Cause:** Chapter hasn't been processed by document-refiner agent yet.
**Solution:** Run document refinement workflow first.

### Error: "Pipeline data missing"
**Cause:** Need to run transformation first.
**Solution:** Run `make transform-enriched` before `make build-chapter-registries`.

### Registry is outdated
**Cause:** Enriched data was updated after building registry.
**Solution:** Run `make build-all-registries` to rebuild everything.

### Want to rebuild from scratch
**Solution:**
```bash
make clean-registries
make build-all-registries
```

---

## Related Documentation

- **Registry Building Tools:** `src/fragile/proofs/tools/REGISTRY_BUILDERS_README.md`
- **Source Location Tools:** `scripts/README_SOURCE_LOCATION_TOOLS.md`
- **Agent Documentation:** `AGENTS.md`

---

**Created:** 2025-10-29
**Purpose:** Document new Makefile targets for mathematical registry building
**Status:** Active (Chapter 1 coverage)
