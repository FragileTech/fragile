# Extract-and-Refine - Troubleshooting

## Parser Issues

### Issue: Found 0 directives

**Symptoms**: Parser runs but creates no entities
```
Phase 1: Extracting MyST directives...
  ✓ Found 0 directives
```

**Cause**: Invalid directive syntax in markdown

**Solution**: Check directive markers
```bash
grep -n "^::::" docs/source/.../document.md | head -10
```

**Fix**: Use exactly 3 colons (`:::`), not 4 (`::::`)
```markdown
# ❌ Wrong
::::{prf:definition}

# ✅ Correct
:::{prf:definition}
```

---

### Issue: Validation errors for labels

**Symptoms**: Parser reports validation errors
```
Failed to create object 'Obj-MyObject': Invalid label format
```

**Cause**: Labels not in lowercase kebab-case

**Solution**: Fix labels in markdown
```markdown
# ❌ Wrong
:label: Obj-MyObject
:label: obj_my_object

# ✅ Correct
:label: obj-my-object
```

---

### Issue: Module not found

**Symptoms**:
```bash
python -m fragile.agents.math_document_parser
ModuleNotFoundError: No module named 'fragile.agents'
```

**Cause**: Not in project directory or fragile not installed

**Solution**:
```bash
# Go to project root
cd /home/guillem/fragile

# Verify fragile installed
python -c "import fragile; print(fragile.__file__)"

# Install if needed
uv sync
```

---

## Cross-Referencer Issues

### Issue: No relationships found

**Symptoms**:
```
✓ Processed 0 explicit references
✓ Created 0 relationships
```

**Cause**: Parser not run first, or no `{prf:ref}` in document

**Solution 1**: Ensure parser ran
```bash
ls -lh docs/source/.../raw_data/
# Should show objects/, theorems/, axioms/ directories
```

**Solution 2**: Check for cross-refs in markdown
```bash
grep "{prf:ref}" docs/source/.../document.md
```

If none found, this is expected - enable LLM mode for implicit dependencies.

---

### Issue: Invalid relationship labels

**Symptoms**: Cross-referencer creates relationships but validation fails

**Cause**: Referenced objects don't exist in registry

**Solution**: Check `relationships/REPORT.md`:
```bash
cat docs/source/.../relationships/REPORT.md | grep "⚠️"
```

Look for warnings like:
```
⚠️  rel-X: source object 'obj-unknown' not found
```

**Fix**: Either:
1. Add missing object definition to markdown
2. Fix incorrect reference label

---

## Refiner Issues

### Issue: SourceLocation validation errors

**Symptoms**:
```
ValidationError for obj-X:
  source.document_id: Field required
  source.file_path: Field required
```

**Cause**: Raw data has old SourceLocation format

**Solution**: Document-refiner should handle this automatically. If persists:
```bash
# Check raw entity
cat docs/source/.../raw_data/objects/obj-X.json | python -m json.tool

# Should have 'source' field
```

**Workaround**: Re-run parser with latest version.

---

### Issue: Missing mathematical_expression field

**Symptoms**:
```
ValidationError for obj-X:
  mathematical_expression: Field required
```

**Cause**: Definition in markdown lacks mathematical content

**Solution**: Add mathematical expression to definition:
```markdown
:::{prf:definition} My Object
:label: obj-my-object

Let $X = (\mathcal{X}, d, \mu)$ be a space...   ← Mathematical content
:::
```

---

### Issue: Refiner takes too long

**Symptoms**: Refiner runs for >1 hour

**Cause**: Too many entities, or LLM rate limiting

**Solution**:
- **Break into batches**: Refine sections individually
- **Check rate limits**: Gemini 2.5 Pro has request limits
- **Monitor progress**: Refiner should show progress updates

**Prevention**: Start with small sections, scale up.

---

## General Issues

### Issue: Scattered old data interfering

**Symptoms**: Parser/cross-referencer picks up old files

**Cause**: Old data directories in unexpected locations

**Solution**: Tools now only search expected locations:
- Parser creates: `{section_dir}/raw_data/`
- Cross-referencer reads: `{section_dir}/raw_data/`, creates `{section_dir}/relationships/`
- Refiner reads: `{section_dir}/raw_data/` + `{section_dir}/relationships/`, creates `{section_dir}/refined_data/`

**Verify**: Check paths in tool output
```
Source: docs/source/1_euclidean_gas/03_cloning
Output: docs/source/1_euclidean_gas/03_cloning/raw_data
```

---

### Issue: Pydantic validation detailed errors

**Symptoms**: Generic "validation error" message

**Solution**: Get detailed error info
```python
from fragile.proofs import MathematicalObject
from pydantic import ValidationError
import json

with open('docs/source/.../objects/obj-X.json') as f:
    data = json.load(f)

try:
    obj = MathematicalObject.model_validate(data)
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Error: {error['msg']}")
        print(f"Input: {error['input']}")
        print()
```

---

### Issue: Permission denied errors

**Symptoms**:
```bash
PermissionError: [Errno 13] Permission denied: 'docs/source/...'
```

**Cause**: Insufficient file permissions

**Solution**:
```bash
# Check permissions
ls -lh docs/source/.../

# Fix if needed
chmod -R u+w docs/source/.../
```

---

## Workflow Issues

### Issue: Wrong execution order

**Symptoms**: Cross-referencer fails because no raw_data/

**Cause**: Ran cross-referencer before parser

**Solution**: Always follow order:
1. Parser (`python -m fragile.agents.math_document_parser`)
2. Cross-referencer (`python -m fragile.agents.cross_reference_analyzer`)
3. Refiner (load document-refiner agent)

---

### Issue: Incremental updates don't work

**Symptoms**: Re-parsing doesn't update existing files

**Cause**: Parser doesn't overwrite by default

**Solution**: Parser **does** overwrite. If seeing old data:
```bash
# Remove old data first
rm -rf docs/source/.../raw_data/

# Re-run parser
python -m fragile.agents.math_document_parser docs/source/.../document.md --no-llm
```

---

## Performance Issues

### Issue: Slow LLM processing

**Symptoms**: Parser/cross-referencer takes >10 minutes

**Cause**: Many entities + LLM API calls

**Solution**:
- **Use --no-llm** for fast iteration
- **Enable LLM** only for final processing
- **Batch process**: Process multiple docs in parallel

**Example parallel processing**:
```bash
# 3 terminals
python -m fragile.agents.math_document_parser doc1.md --no-llm &
python -m fragile.agents.math_document_parser doc2.md --no-llm &
python -m fragile.agents.math_document_parser doc3.md --no-llm &
wait
```

---

## Getting Help

If issues persist:

1. **Check logs**: Look for detailed error messages in console output
2. **Verify versions**: Ensure latest fragile package
3. **Check examples**: See [examples/](./examples/) for working cases
4. **Consult CLAUDE.md**: Mathematical notation requirements
5. **Check agent definitions**: Detailed protocols in `.claude/agents/`

---

**Related**:
- [QUICKSTART.md](./QUICKSTART.md) - Quick reference
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- [SKILL.md](./SKILL.md) - Complete documentation
