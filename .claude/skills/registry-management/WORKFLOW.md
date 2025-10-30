# Registry Management - Complete Workflow

## Workflow 0: Automated (Recommended) ⭐

**Use when**: Always - this is the primary workflow

### Single Command Builds Everything

```bash
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

**What happens automatically**:
1. **Discovery**: Scans for all documents with `refined_data/` (no hardcoded names)
2. **Transformation**: Converts `refined_data` → `pipeline_data` (if needed)
3. **Per-Document Build**: Creates registries for each document
4. **Aggregation**: Merges all per-document registries into combined registries

**Output**:
```
registries/
├── per_document/
│   ├── 01_fragile_gas_framework/
│   │   ├── refined/
│   │   └── pipeline/
│   └── {any_discovered_document}/
│       ├── refined/
│       └── pipeline/
└── combined/
    ├── refined/   # All documents merged
    └── pipeline/  # All documents merged
```

**Time**: ~3-5 minutes (scales with number of documents)

---

## Workflow 1: Dashboard Visualization

**Use when**: Viewing and querying registries interactively

### Launch Dashboard

```bash
panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show
```

### Features

**Auto-Discovery**: Dashboard automatically finds all available registries

**Data Source Dropdown** (populated at runtime):
- Per-document: `{document_name} (Pipeline)`, `{document_name} (Refined)`
- Combined: `Combined Pipeline Registry`, `Combined Refined Registry`
- Custom: User-specified path

### Usage

1. Select data source from dropdown
2. Click "Reload Data"
3. Browse entities, search, visualize relationships

**No code changes needed when adding documents!**

---

## Workflow 2: Adding New Documents

**Use when**: Processing new documents into the registry system

### Step 1: Extract and Refine

Run extract-and-refine workflow (see `../extract-and-refine/WORKFLOW.md`):

```bash
# This creates: docs/source/{chapter}/{new_document}/refined_data/
```

### Step 2: Build All Registries

```bash
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

**What happens**:
- New document **automatically discovered** (no configuration!)
- Transformed to pipeline format
- Per-document registries created
- Combined registries updated (includes new document)
- Dashboard dropdown updated (shows new document)

### Example: Adding 02_euclidean_gas

```bash
# 1. After creating refined_data/, run build_all
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source

# 2. Verify new document appears
ls -lh registries/per_document/
# Expected: 01_fragile_gas_framework, 02_euclidean_gas

# 3. Launch dashboard to view
panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show
# Dropdown automatically shows: "02_euclidean_gas (Pipeline/Refined)"
```

---

## Workflow 3: Manual Per-Document Build (Advanced)

**Use when**: Building specific document only (rare - usually use Workflow 0)

### Build Single Document

```bash
# Transform refined → pipeline (if needed)
python -m fragile.proofs.tools.enriched_to_math_types \
  --input docs/source/{chapter}/{document}/refined_data \
  --output docs/source/{chapter}/{document}/pipeline_data

# Build refined registry
python -m fragile.proofs.tools.build_refined_registry \
  --docs-root docs/source/{chapter} \
  --output registries/per_document/{document}/refined

# Build pipeline registry
python -m fragile.proofs.tools.build_pipeline_registry \
  --pipeline-dir docs/source/{chapter}/{document}/pipeline_data \
  --output registries/per_document/{document}/pipeline
```

### Aggregate Manually

```bash
# Aggregate refined registries
python -m fragile.proofs.tools.aggregate_registries \
  --type refined \
  --per-document-root registries/per_document \
  --output registries/combined/refined

# Aggregate pipeline registries
python -m fragile.proofs.tools.aggregate_registries \
  --type pipeline \
  --per-document-root registries/per_document \
  --output registries/combined/pipeline
```

**Note**: This workflow is rarely needed - `build_all_registries.py` does all of this automatically.

---

## Programmatic Access

### Load Per-Document Registry

```python
from fragile.proofs import load_registry_from_directory, MathematicalRegistry

# Load specific document
registry = load_registry_from_directory(
    MathematicalRegistry,
    'registries/per_document/01_fragile_gas_framework/pipeline'
)

print(f'Objects: {len(registry.get_all_objects())}')
print(f'Theorems: {len(registry.get_all_theorems())}')
```

### Load Combined Registry

```python
# Load combined registry (all documents)
combined = load_registry_from_directory(
    MathematicalRegistry,
    'registries/combined/pipeline'
)

print(f'Total objects: {len(combined.get_all_objects())}')
print(f'Total theorems: {len(combined.get_all_theorems())}')
```

### Query Registry

```python
# Get specific entity
obj = registry.get_object_by_label('obj-euclidean-gas')
print(f'{obj.name}: {obj.description}')

# Get all entities of type
theorems = registry.get_all_theorems()
for thm in theorems[:5]:
    print(f'- {thm.label}: {thm.name}')

# Search by tag
cloning_entities = registry.search_by_tag('cloning')
print(f'Found {len(cloning_entities)} entities tagged with "cloning"')
```

### Discover Available Documents

```python
from pathlib import Path

# Discover per-document registries
per_doc_root = Path('registries/per_document')
documents = [d.name for d in per_doc_root.iterdir() if d.is_dir()]
print(f'Available documents: {documents}')

# Check which registries exist for a document
doc_path = per_doc_root / '01_fragile_gas_framework'
has_refined = (doc_path / 'refined' / 'index.json').exists()
has_pipeline = (doc_path / 'pipeline' / 'index.json').exists()
print(f'Refined: {has_refined}, Pipeline: {has_pipeline}')
```

---

## Performance Comparison

| Workflow | Time | Automation | Scalability | Use Case |
|----------|------|------------|-------------|----------|
| **Workflow 0 (Automated)** ⭐ | ~3-5 min | 🤖 Full | ♾️ Unlimited | Primary workflow |
| Manual per-document | ~10+ min | ❌ None | ⚠️ Poor | Advanced only |
| Raw data build | ~2 min | 🤖 Partial | ♾️ Good | Testing only |

**Recommendation**: Always use **Workflow 0** - it's automated, scalable, and comprehensive.

---

## Document-Agnostic Architecture

### Key Principles

1. **Zero Hardcoded Names**: No document names in code
2. **Pattern-Based Discovery**: Chapter pattern `^\d+_\w+$`
3. **Automatic Detection**: Scans for `refined_data/` directories
4. **Dynamic Loading**: Dashboard discovers registries at runtime

### Adding Documents

**Old way** (hardcoded):
```python
if chapter in ['1_euclidean_gas', '2_geometric_gas']:  # ❌ Bad
    process(chapter)
```

**New way** (automatic):
```python
chapter_pattern = re.compile(r'^\d+_\w+$')  # ✅ Good
for chapter_path in docs_root.iterdir():
    if chapter_pattern.match(chapter_path.name):
        process(chapter_path)
```

### Directory Structure Convention

**Chapter naming**: `{digit(s)}_{name}`
- Examples: `1_euclidean_gas`, `2_geometric_gas`, `10_advanced_topics`

**Document discovery**: Any subdirectory with `refined_data/`
- Pattern: `docs/source/{chapter}/{document}/refined_data/`

**No configuration needed** - just follow the naming convention!

---

**Full Documentation**: [SKILL.md](./SKILL.md)
**Quick Reference**: [QUICKSTART.md](./QUICKSTART.md)
**Issues**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
