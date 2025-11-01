# Registry Management - Troubleshooting

## Document Discovery Issues

### Issue: No documents discovered

**Symptoms**:
```
✓ Discovered 0 documents with refined_data
⚠️  No documents found to process!
```

**Cause**: No `refined_data/` directories found, or wrong directory structure

**Diagnosis**:
```bash
# Check if refined_data exists
find docs/source -type d -name "refined_data"

# Check chapter naming
ls -la docs/source/
# Chapters must match pattern: {digit(s)}_{name}
# Examples: 1_euclidean_gas, 2_geometric_gas, 10_advanced_topics
```

**Solution**:
1. Run extract-and-refine workflow first (see `../extract-and-refine/QUICKSTART.md`)
2. Ensure chapter directories follow naming convention `{digit}_{name}`
3. Ensure `refined_data/` exists inside document directories

---

### Issue: Document not discovered despite having refined_data

**Symptoms**: Document exists but not included in build

**Diagnosis**:
```bash
# Check if chapter follows naming pattern
ls -la docs/source/
# Must match: ^\d+_\w+$

# Example of valid names:
# ✅ 1_euclidean_gas
# ✅ 2_geometric_gas
# ✅ 10_advanced_topics
# ❌ ChapterOne (no digit prefix)
# ❌ chapter_1 (digit not at start)
```

**Solution**: Rename chapter directory to match pattern `{digit}_{name}`

---

## Dashboard Issues

### Issue: Dashboard shows no registries

**Symptoms**: Dashboard dropdown only shows "Custom Path"

**Cause**: No registries built yet

**Diagnosis**:
```bash
# Check if registries exist
ls -lh registries/per_document/
ls -lh registries/combined/

# Check if index.json exists
find registries -name "index.json"
```

**Solution**: Build registries first
```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

---

### Issue: Dashboard can't load registry

**Symptoms**: Dashboard error like "Error loading registry"

**Diagnosis**:
```bash
# Check registry structure
ls -lh registries/per_document/01_fragile_gas_framework/pipeline/
# Should show: index.json, objects/, axioms/, theorems/

# Verify index.json is valid
cat registries/per_document/01_fragile_gas_framework/pipeline/index.json | python -m json.tool
```

**Solution**: Rebuild registries
```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

---

### Issue: New document doesn't appear in dashboard

**Symptoms**: Built new document but dashboard doesn't show it

**Cause**: Dashboard dropdown populated at startup

**Solution**: Restart dashboard or click "Reload Data"
```bash
# Stop dashboard (Ctrl+C)
# Restart
panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show
```

---

### Issue: Dashboard path resolution fails

**Symptoms**: Dashboard error like "Registry path not found"

**Cause**: Dashboard run from wrong directory

**Solution**: Run dashboard from project root
```bash
cd /path/to/fragile
panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show
```

---

## Build Process Issues

### Issue: Validation errors during build

**Symptoms**:
```
❌ Processing completed with 45 errors
```

**Cause**: Data quality issues in refined_data

**Diagnosis**:
```bash
# Check error details in output
python -m fragile.mathster.tools.build_all_registries \
  --docs-root docs/source 2>&1 | grep "Error"
```

**Solution**:
- Check if registries were created despite errors: `ls -lh registries/`
- If registries exist: Errors are non-critical, continue
- If registries missing: Fix specific errors shown in output

---

### Issue: Transformation fails

**Symptoms**:
```
❌ Transformation failed
Error output: ...
```

**Cause**: Incompatible refined_data format

**Diagnosis**:
```bash
# Check specific transformation error
# Check refined_data format
cat docs/source/{chapter}/{document}/refined_data/objects/obj-*.json | python -m json.tool
```

**Solution**: Re-run document-refiner to regenerate refined_data with correct format

---

### Issue: Aggregation fails

**Symptoms**:
```
❌ Refined aggregation failed
❌ Pipeline aggregation failed
```

**Diagnosis**:
```bash
# Check if per-document registries exist
ls -lh registries/per_document/

# Check specific aggregation errors in output
```

**Solution**:
```bash
# Manually run aggregation
python -m fragile.mathster.tools.aggregate_registries \
  --type pipeline \
  --per-document-root registries/per_document \
  --output registries/combined/pipeline
```

---

### Issue: Pipeline data already exists, skip transformation

**Symptoms**:
```
⏭  Pipeline data already exists, skipping transformation
```

**Not an error**: This is expected behavior

**Explanation**: `build_all_registries` skips transformation if `pipeline_data/` already exists

**To force re-transformation**:
```bash
# Remove pipeline_data first
rm -rf docs/source/{chapter}/{document}/pipeline_data

# Then rebuild
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

---

## Registry Loading Issues

### Issue: load_registry_from_directory fails

**Symptoms**:
```python
FileNotFoundError: [Errno 2] No such file or directory: 'registries/.../index.json'
```

**Cause**: Registry not built or wrong path

**Solution**:
```bash
# Check registry exists
ls -lh registries/per_document/01_fragile_gas_framework/pipeline/index.json

# Use absolute path if needed
python -c "
from pathlib import Path
from fragile.proofs import load_registry_from_directory, MathematicalRegistry
registry_path = Path('registries/per_document/01_fragile_gas_framework/pipeline').absolute()
registry = load_registry_from_directory(MathematicalRegistry, registry_path)
"
```

---

### Issue: get_object_by_label returns None

**Symptoms**:
```python
obj = registry.get_object_by_label('obj-euclidean-gas')
print(obj)  # None
```

**Cause**: Object not in registry or label mismatch

**Solution**:
```python
# List all object labels
all_objects = registry.get_all_objects()
labels = [obj.label for obj in all_objects]
print(f'Available labels: {labels[:10]}')

# Search for similar labels
search_term = 'euclidean'
matches = [label for label in labels if search_term in label.lower()]
print(f'Matches: {matches}')
```

---

## Combined Registry Issues

### Issue: Combined registry missing entities

**Symptoms**: Per-document registries have entities, but combined registry is smaller

**Cause**: Duplicate labels (first occurrence wins)

**Diagnosis**:
```bash
# Check aggregation output for duplicate warnings
python -m fragile.mathster.tools.aggregate_registries \
  --type pipeline \
  --per-document-root registries/per_document \
  --output registries/combined/pipeline
# Look for: "⚠️  Found N duplicate labels"
```

**Solution**: This is expected behavior
- Aggregator keeps first occurrence of each label
- Duplicates from other documents are skipped
- To fix: Ensure entities have unique labels across documents

---

### Issue: Combined registry empty

**Symptoms**: Combined registry exists but has no entities

**Cause**: No per-document registries to aggregate

**Diagnosis**:
```bash
# Check if per-document registries exist
ls -lh registries/per_document/

# Check if they have entities
find registries/per_document -name "*.json" | wc -l
```

**Solution**: Build per-document registries first
```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

---

## Performance Issues

### Issue: Build takes too long

**Symptoms**: Build taking >10 minutes

**Cause**: Large number of documents or slow transformation

**Solution**: Check progress
```bash
# Run with verbose output (watch console)
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

# Check what's taking time:
# - Discovery: Should be <1 second
# - Transformation: ~1-2 min per document
# - Registry build: ~30 sec per document
# - Aggregation: ~30 sec total
```

**Optimization**: Pre-transform pipeline_data to skip transformation step
```bash
# Transform all documents first (one-time)
for doc in docs/source/*/*/refined_data; do
  doc_dir=$(dirname "$doc")
  python -m fragile.mathster.tools.enriched_to_math_types \
    --input "$doc" \
    --output "$doc_dir/pipeline_data"
done

# Then build (faster)
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

---

## Getting Help

**Full Documentation**: [SKILL.md](./SKILL.md)
**Quick Reference**: [QUICKSTART.md](./QUICKSTART.md)
**Complete Workflow**: [WORKFLOW.md](./WORKFLOW.md)

**Related Workflows**:
- Extract and refine: [../extract-and-refine/TROUBLESHOOTING.md](../extract-and-refine/TROUBLESHOOTING.md)

**Master Workflow Guide**: `src/fragile/proofs/tools/WORKFLOW.md`
