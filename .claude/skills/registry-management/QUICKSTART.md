# Registry Management - Quick Start

## TL;DR

**One command builds everything automatically:**

```bash
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

**What happens**:
- Discovers all documents (no hardcoded names!)
- Transforms refined_data → pipeline_data
- Builds per-document registries
- Aggregates combined registries

**Output**:
- `registries/per_document/{document}/{refined,pipeline}/` - Individual documents
- `registries/combined/{refined,pipeline}/` - All documents merged

---

## Common Use Cases

### Use Case 1: Build Everything Automatically ⭐

```bash
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

**Time**: ~3-5 minutes (depends on number of documents)
**Input**: All `refined_data/` directories (auto-discovered)
**Output**: Complete registry structure
**Use when**: Primary workflow - always start here

---

### Use Case 2: View in Dashboard

```bash
# After building registries
panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show
```

**Features**:
- Auto-discovers all available registries
- Browse entities, search, visualize relationships
- Dynamically populated data source dropdown

**Data sources** (automatically shown):
- Per-document: `{document_name} (Pipeline/Refined)`
- Combined: `Combined Pipeline/Refined Registry`

---

### Use Case 3: Adding New Documents

```bash
# 1. Extract and refine new document
#    Creates: docs/source/{chapter}/{new_document}/refined_data/

# 2. Build all registries (automatic discovery!)
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

**Result**: New document automatically:
- Discovered (no configuration!)
- Transformed (refined → pipeline)
- Built (per-document registries)
- Aggregated (combined registries)
- Available in dashboard

---

## Verification

### Check Registry Structure

```bash
# Check per-document registries
ls -lh registries/per_document/
# Expected: Document directories (e.g., 01_fragile_gas_framework, 02_euclidean_gas)

ls -lh registries/per_document/01_fragile_gas_framework/
# Expected: refined/, pipeline/

# Check combined registries
ls -lh registries/combined/
# Expected: refined/, pipeline/

# Check entity structure
ls -lh registries/combined/pipeline/
# Expected: objects/, axioms/, theorems/, parameters/, relationships/, index.json
```

### Count Entities

```bash
# Count per-document entities
find registries/per_document/01_fragile_gas_framework/pipeline -name "*.json" | wc -l

# Count combined registry entities
find registries/combined/pipeline -name "*.json" | wc -l
```

### Load Registry Programmatically

```bash
# Load per-document registry
python -c "
from fragile.proofs import load_registry_from_directory, MathematicalRegistry
registry = load_registry_from_directory(
    MathematicalRegistry,
    'registries/per_document/01_fragile_gas_framework/pipeline'
)
print(f'Objects: {len(registry.get_all_objects())}')
print(f'Theorems: {len(registry.get_all_theorems())}')
"

# Load combined registry
python -c "
from fragile.proofs import load_registry_from_directory, MathematicalRegistry
registry = load_registry_from_directory(
    MathematicalRegistry,
    'registries/combined/pipeline'
)
print(f'Objects: {len(registry.get_all_objects())}')
print(f'Theorems: {len(registry.get_all_theorems())}')
"
```

---

## Quick Troubleshooting

### Problem: No documents discovered

```bash
# Check if refined_data exists
find docs/source -type d -name "refined_data"
# Should show at least one directory
```

**Solution**: Run [extract-and-refine](../extract-and-refine/QUICKSTART.md) workflow first to create `refined_data/`.

---

### Problem: Dashboard shows no registries

```bash
# Check if registries were created
ls -lh registries/per_document/
ls -lh registries/combined/
```

**Solution**: Run `build_all_registries.py` to create registries:
```bash
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

---

### Problem: Validation errors

```bash
# Check error messages in output
python -m fragile.proofs.tools.build_all_registries \
  --docs-root docs/source 2>&1 | grep "Error"
```

**Solution**: Errors are often expected for some data. Check if registries were created despite errors.

---

## Workflow Integration

### Complete Pipeline

```bash
# 1. Extract and refine documents
#    (See ../extract-and-refine/QUICKSTART.md)
#    Creates: docs/source/{chapter}/{document}/refined_data/

# 2. Build all registries (automatic!)
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source

# 3. Visualize in dashboard
panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show
```

### Programmatic Access

```python
from fragile.proofs import load_registry_from_directory, MathematicalRegistry

# Load per-document registry
registry = load_registry_from_directory(
    MathematicalRegistry,
    'registries/per_document/01_fragile_gas_framework/pipeline'
)

# Or load combined registry (all documents)
combined = load_registry_from_directory(
    MathematicalRegistry,
    'registries/combined/pipeline'
)

# Query objects
euclidean_gas = registry.get_object_by_label('obj-euclidean-gas')
print(euclidean_gas.name)

# Query theorems
all_theorems = combined.get_all_theorems()
print(f'Found {len(all_theorems)} theorems across all documents')

# Search by tag
cloning_entities = combined.search_by_tag('cloning')
```

---

**Full Documentation**: [SKILL.md](./SKILL.md)
**Step-by-Step**: [WORKFLOW.md](./WORKFLOW.md)
**Issues**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
