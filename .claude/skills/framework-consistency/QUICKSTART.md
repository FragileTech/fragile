# Framework Consistency - Quick Start

Copy-paste ready commands for consistency verification.

---

## Quick Reference Validation

### Check if label exists
```bash
# Single label
grep "^\*\*thm-your-label\*\*" docs/glossary.md

# Multiple labels
for label in thm-convergence obj-euclidean-gas axiom-bounded; do
  echo -n "$label: "
  grep -q "^\*\*$label\*\*" docs/glossary.md && echo "✓" || echo "✗ MISSING"
done
```

### Extract all references from document
```bash
# Get all {prf:ref} references
grep -o '{prf:ref}`[^`]*`' docs/source/1_euclidean_gas/your_document.md | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/' | \
  sort -u
```

### Validate all references
```bash
# Extract and check each reference
grep -o '{prf:ref}`[^`]*`' docs/source/1_euclidean_gas/your_document.md | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/' | \
  sort -u | \
  while read label; do
    echo -n "$label: "
    grep -q "^\*\*$label\*\*" docs/glossary.md && echo "✓" || echo "✗ MISSING"
  done
```

---

## Notation Quick Check

### Find all mathematical symbols
```bash
# Extract inline math
grep -o '\$[^$]*\$' docs/source/1_euclidean_gas/your_document.md | \
  sort -u | \
  head -50

# Check for calligraphic letters
grep '\\mathcal' docs/source/1_euclidean_gas/your_document.md

# Check for Greek letters
grep -E '\\gamma|\\beta|\\sigma|\\tau|\\epsilon' docs/source/1_euclidean_gas/your_document.md
```

### Verify standard notation usage
```bash
# Check if $\mathcal{X}$ used for state space
grep -n '\\mathcal{X}' docs/source/1_euclidean_gas/your_document.md

# Check if $N$ used for number of walkers
grep -n '\$N\$' docs/source/1_euclidean_gas/your_document.md

# Look for notation definitions
grep -A 10 "Notation:" docs/source/1_euclidean_gas/your_document.md
```

### Find notation conflicts
```bash
# Check for overloaded symbols (e.g., N used multiple ways)
# Extract all uses of $N$
grep -n '\$N\$' docs/source/1_euclidean_gas/your_document.md

# Manually verify each usage has same meaning
```

---

## Axiom Validation

### List all axiom references
```bash
# Extract axiom references
grep -o '{prf:ref}`axiom-[^`]*`' docs/source/1_euclidean_gas/your_document.md | \
  sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | \
  sort -u
```

### Verify axioms exist
```bash
# Check each axiom
grep -o '{prf:ref}`axiom-[^`]*`' docs/source/1_euclidean_gas/your_document.md | \
  sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | \
  sort -u | \
  while read axiom; do
    echo -n "$axiom: "
    grep -q "^\*\*$axiom\*\*" docs/glossary.md && echo "✓" || echo "✗ MISSING"
  done
```

### Find axiom definition
```bash
# Get axiom source location
axiom="axiom-bounded-cloning"
grep -A 5 "^\*\*$axiom\*\*" docs/glossary.md

# Read full axiom statement
source_doc=$(grep -A 3 "^\*\*$axiom\*\*" docs/glossary.md | grep "Source:" | cut -d: -f2 | awk '{print $1}')
grep -A 20 ":label: $axiom" docs/source/1_euclidean_gas/$source_doc
```

---

## Definition Cross-Check

### Find all definitions in document
```bash
# Extract definition labels
grep -o ':label: obj-[^[:space:]]*' docs/source/1_euclidean_gas/your_document.md | \
  sed 's/:label: //' | \
  sort -u
```

### Check for duplicates
```bash
# For each definition, check glossary
grep -o ':label: obj-[^[:space:]]*' docs/source/1_euclidean_gas/your_document.md | \
  sed 's/:label: //' | \
  sort -u | \
  while read label; do
    count=$(grep -c "^\*\*$label\*\*" docs/glossary.md)
    if [ $count -eq 0 ]; then
      echo "✓ $label is new"
    else
      echo "! $label exists ($count times) - check for duplicate"
    fi
  done
```

### Search for related definitions
```bash
# Find definitions related to topic
cat docs/glossary.md | grep "gas" | grep "obj-"

# Find definitions with specific tag
cat docs/glossary.md | grep "tag:cloning"
```

---

## Contradiction Detection

### Check for conflicting bounds
```bash
# Search for specific constant in document
grep -n "L \leq" docs/source/1_euclidean_gas/your_document.md

# Compare bounds
# L ≤ 1 at line 45
# L ≤ 2 at line 120  # → Potential contradiction!
```

### Check for conflicting properties
```bash
# Search for property claims
grep -n "continuous" docs/source/1_euclidean_gas/your_document.md
grep -n "discontinuous" docs/source/1_euclidean_gas/your_document.md

# Verify they don't refer to same object
```

### Cross-document consistency check
```bash
# Find all documents mentioning same entity
entity="euclidean-gas"
find docs/source/1_euclidean_gas -name "*.md" -exec grep -l "$entity" {} \;

# Check claims in each document for consistency
```

---

## Complete Consistency Check Script

### All-in-one validation
```bash
#!/bin/bash
# Comprehensive consistency check

DOC="docs/source/1_euclidean_gas/your_document.md"

echo "=== Framework Consistency Check ==="
echo ""

echo "1. Reference Validation"
echo "----------------------"
grep -o '{prf:ref}`[^`]*`' "$DOC" | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/' | \
  sort -u | \
  while read label; do
    echo -n "  $label: "
    grep -q "^\*\*$label\*\*" docs/glossary.md && echo "✓" || echo "✗ MISSING"
  done

echo ""
echo "2. Axiom Validation"
echo "------------------"
grep -o '{prf:ref}`axiom-[^`]*`' "$DOC" | \
  sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | \
  sort -u | \
  while read axiom; do
    echo -n "  $axiom: "
    grep -q "^\*\*$axiom\*\*" docs/glossary.md && echo "✓" || echo "✗ MISSING"
  done

echo ""
echo "3. Definition Check"
echo "------------------"
grep -o ':label: obj-[^[:space:]]*' "$DOC" | \
  sed 's/:label: //' | \
  sort -u | \
  while read label; do
    count=$(grep -c "^\*\*$label\*\*" docs/glossary.md)
    echo -n "  $label: "
    if [ $count -eq 0 ]; then
      echo "New ✓"
    else
      echo "Exists ($count times)"
    fi
  done

echo ""
echo "4. Notation Check"
echo "----------------"
echo "  Calligraphic letters:"
grep -c '\\mathcal' "$DOC" | xargs echo -n "    Found: "
echo " uses"

echo "  Greek letters:"
grep -Ec '\\gamma|\\beta|\\sigma|\\tau|\\epsilon' "$DOC" | xargs echo -n "    Found: "
echo " uses"

echo ""
echo "=== Check Complete ==="
```

Save as `check_consistency.sh`, make executable, run:
```bash
chmod +x check_consistency.sh
./check_consistency.sh
```

---

## Dual-Review for Consistency

### Consistency review prompt (use for BOTH Gemini + Codex)

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
[Paste relevant sections here]

Framework reference: docs/glossary.md contains 741 entries
Please check against existing framework.
```

**Submit to:**
1. `mcp__gemini-cli__ask-gemini` (model: "gemini-2.5-pro")
2. `mcp__codex__codex`

**Use IDENTICAL prompts** for both reviewers.

---

## Common Fixes

### Fix 1: Missing reference
```bash
# Find similar labels
cat docs/glossary.md | grep "convergence" | grep "thm-"

# Update reference in document
vim docs/source/1_euclidean_gas/your_document.md
# Change {prf:ref}`thm-wrong` to {prf:ref}`thm-correct`
```

### Fix 2: Notation conflict
```markdown
<!-- WRONG -->
Let $N$ be the number of walkers.
...
Let $N$ be the norm operator.  <!-- Conflict!

<!-- CORRECT -->
Let $N$ be the number of walkers.
...
Let $\|\cdot\|$ be the norm operator.
```

### Fix 3: Duplicate definition
```markdown
<!-- WRONG -->
:::{prf:definition} Euclidean Gas
:label: obj-euclidean-gas
...
:::
<!-- Already exists in framework!

<!-- CORRECT -->
See {prf:ref}`obj-euclidean-gas` for the definition.
```

### Fix 4: Undefined symbol
```markdown
<!-- Add notation section -->
:::{prf:definition} Your Object
:label: obj-your-object

...

**Notation**:
- $\lambda_v$: Velocity weight parameter
- $\epsilon_F$: Regularization parameter
:::
```

---

## Quick Lookup Tables

### Common Framework Symbols

| Symbol | Meaning | Source |
|--------|---------|--------|
| `$\mathcal{X}$` | State space | obj-state-space |
| `$\mathcal{Y}$` | Algorithmic space | obj-algorithmic-space |
| `$\mathcal{S}$` | Swarm configuration | obj-swarm-state |
| `$\mathcal{A}$` | Alive set | obj-alive-set |
| `$\mathcal{D}$` | Dead set | obj-dead-set |
| `$d_\mathcal{X}$` | Metric on state space | obj-metric |
| `$d_{\text{alg}}$` | Algorithmic distance | obj-algorithmic-distance |
| `$\gamma$` | Friction coefficient | param-gamma |
| `$\beta$` | Exploitation weight | param-beta |
| `$\sigma$` | Perturbation noise | param-sigma |
| `$\tau$` | Time step | param-tau |
| `$N$` | Number of walkers | obj-swarm-state |

### Common Label Prefixes

| Prefix | Entity Type | Example |
|--------|-------------|---------|
| `obj-` | Definition | `obj-euclidean-gas` |
| `thm-` | Theorem | `thm-convergence-rate` |
| `lem-` | Lemma | `lem-lipschitz-bound` |
| `prop-` | Proposition | `prop-intermediate` |
| `cor-` | Corollary | `cor-immediate` |
| `axiom-` | Axiom | `axiom-bounded-cloning` |
| `param-` | Parameter | `param-friction` |

---

## Verification Workflow

### Before committing changes

```bash
# 1. Validate all references
grep -o '{prf:ref}`[^`]*`' your_document.md | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/' | \
  sort -u | \
  while read label; do
    grep -q "^\*\*$label\*\*" docs/glossary.md || echo "✗ $label MISSING"
  done

# 2. Check for duplicate definitions
grep -o ':label: [^[:space:]]*' your_document.md | \
  sed 's/:label: //' | \
  while read label; do
    count=$(grep -c "^\*\*$label\*\*" docs/glossary.md)
    [ $count -gt 0 ] && echo "! $label exists - check for duplicate"
  done

# 3. Verify notation usage
grep '\\mathcal{X}' your_document.md && echo "Uses state space notation"
grep '\\mathcal{Y}' your_document.md && echo "Uses algorithmic space notation"

# 4. Build and check
make build-docs 2>&1 | grep -E "WARNING|ERROR"
```

---

## Integration Examples

### With mathematical-writing
```bash
# 1. Draft content (mathematical-writing)
vim docs/source/1_euclidean_gas/your_document.md

# 2. Check consistency (framework-consistency)
./check_consistency.sh

# 3. Fix issues
vim docs/source/1_euclidean_gas/your_document.md

# 4. Build
make build-docs
```

### With proof-validation
```bash
# 1. Develop proof (proof-validation)
# Load proof-sketcher agent...

# 2. Verify axiom usage (framework-consistency)
grep -o '{prf:ref}`axiom-[^`]*`' proof_file.md | \
  sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | \
  while read axiom; do
    grep -q "^\*\*$axiom\*\*" docs/glossary.md || echo "✗ $axiom MISSING"
  done

# 3. Finalize proof
```

---

**Time estimates:**
- Reference validation: ~1 min (automated)
- Notation audit: ~5 min (automated + manual review)
- Axiom validation: ~10 min (manual verification)
- Definition cross-check: ~5 min (automated)
- Complete consistency check: ~20-30 min

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
