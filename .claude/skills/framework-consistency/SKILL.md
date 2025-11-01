---
name: framework-consistency
description: Verify mathematical content consistency with Fragile framework standards. Use when checking notation compliance, validating axiom usage, cross-referencing definitions, or identifying contradictions in mathematical documents.
---

# Framework Consistency Skill

## Purpose

Systematic verification that mathematical content aligns with the established Fragile framework, ensuring notation consistency, valid axiom usage, and absence of contradictions.

**Input**: Mathematical documents (existing or draft)
**Output**: Consistency reports identifying issues and recommendations
**Pipeline**: Index Check → Notation Audit → Axiom Validation → Contradiction Detection

---

## Why Framework Consistency Matters

### Quality Assurance

**Problem**: Mathematical documents can:
- Use inconsistent notation across sections
- Reference non-existent axioms
- Contradict established definitions
- Duplicate existing results
- Break framework assumptions

**Solution**: Systematic consistency checking catches these issues early.

### Framework Characteristics

The Fragile framework is:
- **Large**: 741 documented entries (523 Euclidean, 160 Geometric, 58 Geometric Gas)
- **Interconnected**: Heavy cross-referencing between definitions/theorems
- **Standardized**: Strict notation conventions (see `docs/glossary.md`)
- **Rigorous**: Top-tier journal standards

**Challenge**: Easy to introduce inconsistencies without systematic checking.

---

## The Mathematical Index

**Primary Resource**: `docs/glossary.md`

### What the Glossary Contains

```bash
# Structure
cat docs/glossary.md | head -50

# Contents:
# - 741 mathematical entries
# - Organized by document
# - Each entry: type, label, name, tags, location
# - Cross-references for navigation
```

### Entry Types

| Type | Prefix | Count | Examples |
|------|--------|-------|----------|
| Objects (definitions) | `obj-` | ~300 | `obj-euclidean-gas`, `obj-swarm-state` |
| Axioms | `axiom-` | ~80 | `axiom-bounded-cloning`, `axiom-lipschitz` |
| Theorems | `thm-` | ~200 | `thm-convergence-rate`, `thm-mean-field` |
| Lemmas | `lem-` | ~120 | `lem-helper-bound`, `lem-lipschitz-property` |
| Propositions | `prop-` | ~40 | `prop-intermediate-result` |

### Glossary Structure

**Example entry:**
```markdown
**obj-euclidean-gas**
- Type: Definition
- Name: "Euclidean Gas"
- Tags: langevin, cloning, euclidean
- Source: 02_euclidean_gas.md, Section 2.1
- Description: Particle system with Langevin dynamics and cloning
```

---

## Complete Consistency Workflow

### Stage 1: Index Cross-Check

**Purpose**: Verify all references point to existing entities

#### Step 1.1: Extract All References

```bash
# Find all cross-references in document
grep -o '{prf:ref}`[^`]*`' docs/source/1_euclidean_gas/your_document.md > /tmp/refs.txt

# Extract labels
cat /tmp/refs.txt | sed 's/{prf:ref}`\([^`]*\)`/\1/' | sort -u
```

**Example output:**
```
axiom-bounded-cloning
lem-lipschitz-property
obj-euclidean-gas
thm-convergence-rate
```

#### Step 1.2: Validate Against Glossary

```bash
# For each reference, check if exists in glossary
while read label; do
  if grep -q "^\*\*$label\*\*" docs/glossary.md; then
    echo "✓ $label exists"
  else
    echo "✗ $label MISSING"
  fi
done < /tmp/refs_labels.txt
```

**Common issues:**
- Typo in label: `thm-convergance-rate` → `thm-convergence-rate`
- Wrong prefix: `def-euclidean-gas` → `obj-euclidean-gas`
- Label doesn't exist: Need to add definition first

#### Step 1.3: Check For Undefined References

**If label missing:**

1. **Is it a typo?**
   ```bash
   # Find similar labels
   cat docs/glossary.md | grep "convergence" | grep "thm-"
   ```

2. **Does concept need to be defined?**
   ```bash
   # Check if concept exists under different label
   cat docs/glossary.md | grep -i "your-concept"
   ```

3. **Create definition if missing:**
   - Draft new definition
   - Follow mathematical-writing workflow
   - Add to appropriate document

---

### Stage 2: Notation Audit

**Purpose**: Ensure notation matches glossary conventions

#### Step 2.1: Identify Notation in Document

```bash
# Extract mathematical symbols from document
grep -o '\$[^$]*\$' docs/source/1_euclidean_gas/your_document.md | sort -u > /tmp/notation.txt

# Check for common symbols
grep '\\mathcal' /tmp/notation.txt  # Calligraphic letters
grep '\\gamma\|\\beta\|\\sigma' /tmp/notation.txt  # Greek letters
grep '_' /tmp/notation.txt  # Subscripts
```

#### Step 2.2: Compare Against Framework Standards

**Common symbols to check:**

| Symbol | Standard Meaning | Check |
|--------|------------------|-------|
| `$\mathcal{X}$` | State space | Must mean state space |
| `$\mathcal{Y}$` | Algorithmic space | Must mean algorithmic space |
| `$\mathcal{S}$` | Swarm configuration | Must mean swarm config |
| `$\gamma$` | Friction coefficient | Should not mean something else |
| `$\beta$` | Exploitation weight | Should not be reused |
| `$\sigma$` | Perturbation noise | Should not be reused |
| `$d_\mathcal{X}$` | Metric on state space | Must be the metric |
| `$N$` | Number of walkers | Should not mean norm |

**Verification script:**
```bash
# Check if $\mathcal{X}$ is used correctly
grep -A 5 -B 5 '\\mathcal{X}' docs/source/1_euclidean_gas/your_document.md

# Verify definition matches glossary
cat docs/glossary.md | grep -A 3 "obj-state-space"
```

#### Step 2.3: Identify Notation Conflicts

**Conflict types:**

1. **Overloaded symbols** (same symbol, different meanings):
   ```bash
   # Find all uses of $N$
   grep -n '\$N\$' docs/source/1_euclidean_gas/your_document.md

   # Check meanings at each location
   # Line 45: $N$ = number of walkers ✓
   # Line 120: $N$ = norm operator ✗ (conflict!)
   ```

2. **Inconsistent notation** (same concept, different symbols):
   ```bash
   # Document uses both $d$ and $\rho$ for metric
   grep -n '\$d_' docs/source/1_euclidean_gas/your_document.md
   grep -n '\\rho_' docs/source/1_euclidean_gas/your_document.md

   # Should unify to $d_\mathcal{X}$ (framework standard)
   ```

#### Step 2.4: Check Notation Definitions

**Every symbol must be defined:**

```bash
# Find undefined symbols (appear without introduction)
# Example: $\lambda_v$ appears without definition

# Search for definition
grep -n 'lambda_v' docs/source/1_euclidean_gas/your_document.md

# If no definition found, add one:
# - In definition directive's "Notation" section
# - Or inline when first introduced
```

---

### Stage 3: Axiom Validation

**Purpose**: Verify all axiom references are valid and correctly applied

#### Step 3.1: Extract Axiom Claims

```bash
# Find all axiom references
grep -o '{prf:ref}`axiom-[^`]*`' docs/source/1_euclidean_gas/your_document.md > /tmp/axiom_refs.txt

# Get axiom labels
cat /tmp/axiom_refs.txt | sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | sort -u
```

**Example output:**
```
axiom-bounded-cloning
axiom-geometric-consistency
axiom-lipschitz-continuity
```

#### Step 3.2: Verify Axiom Exists

```bash
# For each axiom, check glossary
while read axiom; do
  if grep -q "^\*\*$axiom\*\*" docs/glossary.md; then
    echo "✓ $axiom exists"

    # Get source location
    grep -A 5 "^\*\*$axiom\*\*" docs/glossary.md
  else
    echo "✗ $axiom MISSING"
  fi
done < /tmp/axiom_labels.txt
```

#### Step 3.3: Validate Axiom Application

**Check if axiom is correctly applied:**

1. **Read axiom statement:**
   ```bash
   # Find axiom in source document
   axiom_label="axiom-bounded-cloning"
   source_doc=$(cat docs/glossary.md | grep -A 3 "^\*\*$axiom_label\*\*" | grep "Source:" | cut -d: -f2 | awk '{print $1}')

   # Read full axiom
   grep -A 20 ":label: $axiom_label" docs/source/1_euclidean_gas/$source_doc
   ```

2. **Check hypotheses are satisfied:**
   ```markdown
   Example:

   Axiom states: "Requires complete metric space"
   Your document: Does it establish completeness first?

   Axiom states: "Assumes bounded Lipschitz constant L ≤ 1"
   Your document: Is L bounded by 1?
   ```

3. **Verify conclusion usage:**
   ```markdown
   Axiom provides: "Operator is Lipschitz with constant L"
   Your document: Uses Lipschitz property correctly?
   ```

#### Step 3.4: Check For Missing Axioms

**Identify where axioms are needed but not referenced:**

```bash
# Search for claims that should reference axioms
grep -n "Lipschitz" docs/source/1_euclidean_gas/your_document.md
# → Should these reference axiom-lipschitz-continuity?

grep -n "bounded" docs/source/1_euclidean_gas/your_document.md
# → Should these reference axiom-bounded-* ?

grep -n "complete.*metric" docs/source/1_euclidean_gas/your_document.md
# → Should reference axiom-metric-completeness?
```

**Action if missing:**
- Add explicit reference to axiom
- Verify axiom hypotheses are satisfied
- Justify axiom application in proof

---

### Stage 4: Definition Cross-Check

**Purpose**: Ensure definitions are consistent with framework

#### Step 4.1: Extract Definitions

```bash
# Find all definitions in document
grep -o ':label: obj-[^[:space:]]*' docs/source/1_euclidean_gas/your_document.md > /tmp/defs.txt

# Get labels
cat /tmp/defs.txt | sed 's/:label: //' | sort -u
```

#### Step 4.2: Check For Duplicates

```bash
# For each definition, check if already exists
while read label; do
  count=$(grep -c "^\*\*$label\*\*" docs/glossary.md)

  if [ $count -eq 0 ]; then
    echo "✓ $label is new"
  elif [ $count -eq 1 ]; then
    echo "! $label already exists - check for duplicate"
    grep -A 5 "^\*\*$label\*\*" docs/glossary.md
  else
    echo "✗ $label has $count entries - PROBLEM"
  fi
done < /tmp/def_labels.txt
```

**If duplicate found:**
1. Compare definitions - are they identical?
2. If identical: Remove duplicate, reference existing one
3. If different: This is a conflict → needs resolution

#### Step 4.3: Check Definition Consistency

**For related definitions, verify compatibility:**

```bash
# Example: Check all "gas" definitions are compatible
cat docs/glossary.md | grep "gas" | grep "obj-"

# Read each definition
# obj-euclidean-gas: Uses Langevin + cloning
# obj-adaptive-gas: Extends Euclidean Gas with adaptive mechanisms
# obj-geometric-gas: Uses Riemannian structure

# Check: Are extensions compatible with base definition?
```

**Consistency requirements:**
- Extended definitions must preserve base properties
- Notation must be consistent across definitions
- No contradictory statements

#### Step 4.4: Validate Dependencies

**Check if definition references exist:**

```bash
# Find references within definition
grep -A 30 ':label: obj-your-definition' docs/source/1_euclidean_gas/your_document.md | grep '{prf:ref}'

# Verify each reference exists in glossary
```

---

### Stage 5: Contradiction Detection

**Purpose**: Identify mathematical contradictions in framework

#### Step 5.1: Automated Checks

**Check for obvious contradictions:**

```bash
# 1. Conflicting bounds
grep -n "L \leq 1" docs/source/1_euclidean_gas/your_document.md
grep -n "L \leq 2" docs/source/1_euclidean_gas/your_document.md
# → If both found, potential contradiction

# 2. Conflicting properties
grep -n "continuous" docs/source/1_euclidean_gas/your_document.md
grep -n "discontinuous" docs/source/1_euclidean_gas/your_document.md
# → Check if referring to same object

# 3. Incompatible assumptions
grep -n "compact" docs/source/1_euclidean_gas/your_document.md
grep -n "unbounded" docs/source/1_euclidean_gas/your_document.md
# → Verify compatibility
```

#### Step 5.2: Manual Verification

**Compare claims across documents:**

1. **Your document claims**: "$\Psi$ is Lipschitz with $L = 2$"
2. **Framework states** (from glossary): "$\Psi$ satisfies $L \leq 1$"
3. **Contradiction detected**: $L = 2 \not\leq 1$

**Resolution:**
- Check which is correct (verify proofs)
- Update incorrect statement
- Add note explaining resolution

#### Step 5.3: Cross-Document Consistency

**Check consistency across multiple documents:**

```bash
# Search for claims about same entity in different documents
find docs/source/1_euclidean_gas -name "*.md" -exec grep -l "euclidean-gas" {} \;

# Read relevant sections from each
# Verify claims are compatible
```

**Example workflow:**
```bash
# Document A: "Euclidean Gas converges exponentially"
# Document B: "Euclidean Gas converges polynomially"
# → Contradiction needs resolution

# Investigate:
# 1. Read both mathster
# 2. Check hypotheses (maybe different assumptions)
# 3. Determine correct statement
# 4. Fix incorrect one
```

---

## Dual-Review for Consistency

**Use dual-review protocol** for comprehensive consistency checking.

### Consistency Review Prompt

**Submit to both Gemini 2.5 Pro + Codex:**

```
Verify framework consistency for this document:

1. **Notation Check**:
   - Does notation match docs/glossary.md standards?
   - Are symbols defined before use?
   - Any overloaded or conflicting notation?

2. **Axiom Validation**:
   - Are all axiom references valid?
   - Are axiom hypotheses satisfied?
   - Any missing axiom citations?

3. **Definition Consistency**:
   - Are definitions compatible with existing framework?
   - Any duplicate or conflicting definitions?
   - Are dependencies properly referenced?

4. **Contradiction Detection**:
   - Any claims contradicting established results?
   - Any incompatible assumptions?
   - Any logical inconsistencies?

Document content:
[Paste document sections]

Framework reference: docs/glossary.md
```

**Compare results:**
- **Consensus issues**: Both identify same problem → High confidence
- **Contradictions**: Investigate manually against framework docs
- **Unique issues**: Verify before accepting

---

## Common Consistency Issues

### Issue 1: Undefined Symbol

**Symptoms**: Symbol used without definition

**Example**: Uses $\lambda_v$ without defining it

**Fix**:
```markdown
<!-- Add to notation section -->
**Notation**:
- $\lambda_v$: Velocity weight parameter
```

---

### Issue 2: Wrong Axiom Reference

**Symptoms**: References axiom that doesn't exist

**Example**: `{prf:ref}`axiom-wrong-name``

**Fix**:
```bash
# Find correct axiom
cat docs/glossary.md | grep "axiom-" | grep "bounded"

# Update reference
```

---

### Issue 3: Notation Conflict

**Symptoms**: Same symbol means different things

**Example**: $N$ used for both "number of walkers" and "norm operator"

**Fix**:
```markdown
<!-- Use distinct notation -->
$N$ = number of walkers (framework standard)
$\|\cdot\|$ = norm operator (not $N$)
```

---

### Issue 4: Duplicate Definition

**Symptoms**: Concept already defined elsewhere

**Example**: Defining "Euclidean Gas" when `obj-euclidean-gas` exists

**Fix**:
- Remove duplicate definition
- Reference existing one: `{prf:ref}`obj-euclidean-gas``

---

### Issue 5: Contradictory Claims

**Symptoms**: Statement contradicts framework

**Example**: Claims "$L = 2$" but framework requires "$L \leq 1$"

**Fix**:
1. Verify framework requirement (check source document)
2. Determine which is correct
3. Update incorrect claim
4. Add justification

---

## Integration with Other Skills

### With Mathematical-Writing

**Before finalizing content:**
1. Write content (mathematical-writing)
2. Check consistency (framework-consistency)
3. Fix issues
4. Dual-review
5. Finalize

### With Proof-Validation

**After proof development:**
1. Develop proof (proof-validation)
2. Verify axiom usage is consistent (framework-consistency)
3. Check notation compliance
4. Finalize proof

### With Extract-and-Refine

**After extraction:**
1. Extract entities (extract-and-refine)
2. Check for duplicates (framework-consistency)
3. Verify notation consistency
4. Refine entities

---

## Consistency Checklist

Before committing any mathematical content:

**Index Verification:**
- [ ] All `{prf:ref}` labels exist in glossary
- [ ] No typos in labels
- [ ] Correct label prefixes used

**Notation Audit:**
- [ ] All symbols defined before use
- [ ] Notation matches glossary conventions
- [ ] No overloaded symbols
- [ ] No notation conflicts

**Axiom Validation:**
- [ ] All axiom references are valid
- [ ] Axiom hypotheses satisfied
- [ ] Axiom applications justified
- [ ] No missing axiom citations

**Definition Consistency:**
- [ ] No duplicate definitions
- [ ] Definitions compatible with framework
- [ ] Dependencies properly referenced
- [ ] Extensions preserve base properties

**Contradiction Detection:**
- [ ] No conflicting bounds or constants
- [ ] No incompatible assumptions
- [ ] No logical inconsistencies
- [ ] Cross-document consistency verified

---

## Tools and Scripts

### Automated Consistency Checks

**Reference validation:**
```bash
# Check all references exist
python -c "
import re
from pathlib import Path

# Read document
doc = Path('docs/source/1_euclidean_gas/your_document.md').read_text()

# Extract references
refs = re.findall(r'{prf:ref}\`([^\`]+)\`', doc)

# Read glossary
glossary = Path('docs/glossary.md').read_text()

# Check each reference
for ref in set(refs):
    if f'**{ref}**' in glossary:
        print(f'✓ {ref}')
    else:
        print(f'✗ {ref} MISSING')
"
```

**Notation extraction:**
```bash
# Extract all mathematical symbols
python -c "
import re
from pathlib import Path

doc = Path('docs/source/1_euclidean_gas/your_document.md').read_text()
symbols = re.findall(r'\$([^\$]+)\$', doc)

for sym in sorted(set(symbols)):
    if len(sym) < 50:  # Skip long expressions
        print(sym)
" | sort -u
```

---

## Related Documentation

- **docs/glossary.md**: Complete framework index
- **CLAUDE.md**: Framework standards and notation conventions
- **Mathematical-Writing Skill**: Content creation workflow
- **Proof-Validation Skill**: Proof development and review
- **Extract-and-Refine Skill**: Entity extraction workflow

---

## Version History

- **v1.0.0** (2025-10-28): Initial framework-consistency skill
  - Index cross-checking workflow
  - Notation audit procedures
  - Axiom validation protocol
  - Contradiction detection methods
  - Integration with dual-review

---

**Next**: See [QUICKSTART.md](./QUICKSTART.md) for copy-paste commands.
