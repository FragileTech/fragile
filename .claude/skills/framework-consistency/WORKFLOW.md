# Framework Consistency - Complete Workflow

Detailed step-by-step procedures for verifying framework consistency.

---

## Prerequisites

- ✅ Document to verify (existing or draft)
- ✅ Access to `docs/glossary.md` (741 framework entries)
- ✅ Basic knowledge of bash/grep for automated checks
- ✅ Understanding of framework standards (CLAUDE.md)

---

## Complete Consistency Verification Pipeline

### Stage 0: Preparation

**Purpose**: Set up environment and identify scope

#### Step 0.1: Identify Document

```bash
# Determine which document to check
DOCUMENT="docs/source/1_euclidean_gas/your_document.md"

# Verify document exists
ls -lh "$DOCUMENT"
```

#### Step 0.2: Understand Document Scope

```bash
# Get document statistics
echo "Document: $DOCUMENT"
echo "Lines: $(wc -l < $DOCUMENT)"
echo "Definitions: $(grep -c ':label: obj-' $DOCUMENT)"
echo "Theorems: $(grep -c ':label: thm-' $DOCUMENT)"
echo "Axiom refs: $(grep -c '{prf:ref}\`axiom-' $DOCUMENT)"
```

#### Step 0.3: Review Glossary Structure

```bash
# Understand glossary organization
cat docs/glossary.md | head -100

# Count entries by type
echo "Objects: $(grep -c '^\*\*obj-' docs/glossary.md)"
echo "Theorems: $(grep -c '^\*\*thm-' docs/glossary.md)"
echo "Axioms: $(grep -c '^\*\*axiom-' docs/glossary.md)"
echo "Lemmas: $(grep -c '^\*\*lem-' docs/glossary.md)"
```

---

### Stage 1: Reference Validation

**Purpose**: Verify all cross-references point to existing entities

#### Step 1.1: Extract All References

```bash
# Create working directory
mkdir -p /tmp/consistency_check
cd /tmp/consistency_check

# Extract all {prf:ref} references
grep -o '{prf:ref}`[^`]*`' "$DOCUMENT" > all_refs.txt

# Extract labels only
sed 's/{prf:ref}`\([^`]*\)`/\1/' all_refs.txt | sort -u > ref_labels.txt

# View results
echo "Found $(wc -l < ref_labels.txt) unique references:"
cat ref_labels.txt
```

#### Step 1.2: Categorize References

```bash
# Separate by type
grep '^obj-' ref_labels.txt > refs_objects.txt
grep '^thm-' ref_labels.txt > refs_theorems.txt
grep '^lem-' ref_labels.txt > refs_lemmas.txt
grep '^axiom-' ref_labels.txt > refs_axioms.txt
grep '^prop-' ref_labels.txt > refs_propositions.txt

# Count by type
echo "Objects: $(wc -l < refs_objects.txt)"
echo "Theorems: $(wc -l < refs_theorems.txt)"
echo "Lemmas: $(wc -l < refs_lemmas.txt)"
echo "Axioms: $(wc -l < refs_axioms.txt)"
echo "Propositions: $(wc -l < refs_propositions.txt)"
```

#### Step 1.3: Validate Each Reference

```bash
# Check objects
echo "Validating Objects..."
while read label; do
  if grep -q "^\*\*$label\*\*" docs/glossary.md; then
    echo "  ✓ $label"
  else
    echo "  ✗ $label MISSING" | tee -a missing_refs.txt
  fi
done < refs_objects.txt

# Repeat for theorems, lemmas, axioms, propositions
echo "Validating Theorems..."
while read label; do
  if grep -q "^\*\*$label\*\*" docs/glossary.md; then
    echo "  ✓ $label"
  else
    echo "  ✗ $label MISSING" | tee -a missing_refs.txt
  fi
done < refs_theorems.txt

# ... (similar for lemmas, axioms, propositions)
```

#### Step 1.4: Investigate Missing References

**For each missing reference:**

```bash
# Get the missing label
MISSING="thm-convergance-rate"  # Example typo

# 1. Check for typos - search for similar labels
echo "Searching for similar labels..."
cat docs/glossary.md | grep -i "convergence" | grep "thm-"

# Found: thm-convergence-rate (correct spelling)

# 2. Get source location of similar label
grep -A 5 "^\*\*thm-convergence-rate\*\*" docs/glossary.md

# 3. Decide action:
#    - If typo: Fix reference in document
#    - If truly missing: Need to add definition first
```

#### Step 1.5: Fix Missing References

```bash
# For typos - fix in document
vim "$DOCUMENT"
# Search for wrong label, replace with correct one

# For truly missing entities - check if concept needs definition
# If yes, use mathematical-writing skill to create definition
# If no, maybe wrong concept is referenced
```

---

### Stage 2: Notation Audit

**Purpose**: Ensure notation matches framework conventions

#### Step 2.1: Extract All Mathematical Symbols

```bash
# Extract inline math $...$
grep -o '\$[^$]\+\$' "$DOCUMENT" | sort -u > inline_math.txt

# Extract display math $$...$$
# (More complex - need to handle multiline)
grep -oP '\$\$[^\$]+\$\$' "$DOCUMENT" | sort -u > display_math.txt

# Count symbols
echo "Inline expressions: $(wc -l < inline_math.txt)"
echo "Display expressions: $(wc -l < display_math.txt)"
```

#### Step 2.2: Identify Key Symbols

```bash
# Calligraphic letters (spaces, sets)
grep '\\mathcal' inline_math.txt > symbols_calligraphic.txt
echo "Calligraphic symbols: $(wc -l < symbols_calligraphic.txt)"
cat symbols_calligraphic.txt

# Greek letters (parameters, coefficients)
grep -E '\\gamma|\\beta|\\sigma|\\tau|\\epsilon|\\lambda|\\alpha' inline_math.txt > symbols_greek.txt
echo "Greek symbols: $(wc -l < symbols_greek.txt)"
cat symbols_greek.txt

# Subscripted symbols
grep '_' inline_math.txt > symbols_subscript.txt
echo "Subscripted: $(wc -l < symbols_subscript.txt)"
```

#### Step 2.3: Verify Standard Notation Usage

**Check each standard symbol:**

```bash
# State space: $\mathcal{X}$
echo "Checking state space notation..."
if grep -q '\\mathcal{X}' "$DOCUMENT"; then
  echo "  Found $\mathcal{X}"

  # Verify used correctly
  grep -B 2 -A 2 '\\mathcal{X}' "$DOCUMENT" | head -20

  # Should describe state space, not something else
  # Manual verification needed
fi

# Algorithmic space: $\mathcal{Y}$
echo "Checking algorithmic space notation..."
if grep -q '\\mathcal{Y}' "$DOCUMENT"; then
  echo "  Found $\mathcal{Y}"
  grep -B 2 -A 2 '\\mathcal{Y}' "$DOCUMENT" | head -20
fi

# Friction coefficient: $\gamma$
echo "Checking friction coefficient notation..."
if grep -q '\\gamma' "$DOCUMENT"; then
  echo "  Found $\gamma"
  grep -n '\\gamma' "$DOCUMENT" | head -10
  # Verify: Should mean friction coefficient
fi

# Number of walkers: $N$
echo "Checking walker count notation..."
if grep -q '\$N\$' "$DOCUMENT"; then
  echo "  Found $N"
  grep -n '\$N\$' "$DOCUMENT"
  # Verify: Should mean number of walkers, not norm or other
fi
```

#### Step 2.4: Detect Notation Conflicts

**Check for overloaded symbols:**

```bash
# Example: Check all uses of $N$
echo "Analyzing uses of N..."
grep -n '\$N\$' "$DOCUMENT" > uses_of_N.txt

# Review each use manually
while read line; do
  echo "$line"
  # Extract context
  line_num=$(echo "$line" | cut -d: -f1)
  sed -n "${line_num}p" "$DOCUMENT"
done < uses_of_N.txt

# Manual check: Do all uses mean "number of walkers"?
# If no → Notation conflict detected
```

**Check for undefined symbols:**

```bash
# Find symbols that appear without introduction
# Example: $\lambda_v$ should be defined in notation section

# Search for symbol
symbol='\\lambda_v'
echo "Checking if $symbol is defined..."

# Find first use
first_use=$(grep -n "$symbol" "$DOCUMENT" | head -1)
echo "First use: $first_use"

# Check if definition precedes first use
line_num=$(echo "$first_use" | cut -d: -f1)
head -$line_num "$DOCUMENT" | grep -A 5 "Notation:" | grep "$symbol"

# If not found → Symbol undefined
```

#### Step 2.5: Cross-Check Against Glossary

**For key symbols, verify usage matches glossary:**

```bash
# Check standard notation for state space
echo "Framework standard for state space:"
cat docs/glossary.md | grep -A 3 "obj-state-space"

# Compare with document usage
echo "Document usage:"
grep -B 2 -A 2 '\\mathcal{X}' "$DOCUMENT" | head -10

# Manual verification: Do they match?
```

---

### Stage 3: Axiom Validation

**Purpose**: Verify all axiom references are valid and correctly applied

#### Step 3.1: Extract Axiom References

```bash
# Find all axiom references
grep -o '{prf:ref}`axiom-[^`]*`' "$DOCUMENT" > axiom_refs.txt

# Extract labels
sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' axiom_refs.txt | sort -u > axiom_labels.txt

echo "Found $(wc -l < axiom_labels.txt) unique axiom references:"
cat axiom_labels.txt
```

#### Step 3.2: Validate Axiom Existence

```bash
# Check each axiom exists in glossary
echo "Validating axioms..."
while read axiom; do
  echo -n "  $axiom: "
  if grep -q "^\*\*$axiom\*\*" docs/glossary.md; then
    echo "✓"

    # Get source location
    grep -A 5 "^\*\*$axiom\*\*" docs/glossary.md >> axiom_sources.txt
  else
    echo "✗ MISSING"
    echo "$axiom" >> missing_axioms.txt
  fi
done < axiom_labels.txt

# Review sources
cat axiom_sources.txt
```

#### Step 3.3: Read Axiom Statements

**For each axiom, read full statement:**

```bash
# Example axiom
AXIOM="axiom-bounded-cloning"

# Get source document
source_doc=$(grep -A 3 "^\*\*$AXIOM\*\*" docs/glossary.md | \
  grep "Source:" | \
  cut -d: -f2 | \
  awk '{print $1}')

echo "Axiom: $AXIOM"
echo "Source: $source_doc"

# Read full axiom
grep -A 30 ":label: $AXIOM" "docs/source/1_euclidean_gas/$source_doc"
```

#### Step 3.4: Verify Axiom Application

**For each axiom usage in document, verify:**

1. **Find usage context:**
```bash
# Get context where axiom is referenced
grep -B 5 -A 5 "{prf:ref}\`$AXIOM\`" "$DOCUMENT"
```

2. **Check hypotheses:**
```markdown
Example:

Axiom states:
- Requires: Complete metric space ($\mathcal{X}$, d)
- Requires: Lipschitz constant L ≤ 1

Document context:
- Does it establish completeness? ✓ / ✗
- Does it ensure L ≤ 1? ✓ / ✗
```

3. **Verify conclusion usage:**
```markdown
Axiom provides:
- Operator is Lipschitz continuous with constant L

Document uses:
- "By axiom-X, the operator is Lipschitz" ✓
- "By axiom-X, we have L = 2" ✗ (contradicts axiom!)
```

#### Step 3.5: Identify Missing Axiom Citations

**Search for claims that should reference axioms:**

```bash
# Look for "Lipschitz" without axiom reference
grep -n "Lipschitz" "$DOCUMENT" > lipschitz_uses.txt

# For each use, check if axiom is cited nearby
while read line; do
  line_num=$(echo "$line" | cut -d: -f1)

  # Check ±5 lines for axiom reference
  start=$((line_num - 5))
  end=$((line_num + 5))

  has_ref=$(sed -n "${start},${end}p" "$DOCUMENT" | grep -c '{prf:ref}`axiom-')

  if [ $has_ref -eq 0 ]; then
    echo "Line $line_num: Lipschitz claim without axiom reference"
    sed -n "${line_num}p" "$DOCUMENT"
  fi
done < lipschitz_uses.txt
```

---

### Stage 4: Definition Cross-Check

**Purpose**: Ensure definitions are consistent with framework

#### Step 4.1: Extract Definitions

```bash
# Find all definition labels in document
grep -o ':label: obj-[^[:space:]]*' "$DOCUMENT" > def_labels_raw.txt

# Clean labels
sed 's/:label: //' def_labels_raw.txt | sort -u > def_labels.txt

echo "Found $(wc -l < def_labels.txt) definitions:"
cat def_labels.txt
```

#### Step 4.2: Check for Duplicates

```bash
# For each definition, check glossary
echo "Checking for duplicates..."
while read label; do
  count=$(grep -c "^\*\*$label\*\*" docs/glossary.md)

  echo -n "  $label: "
  if [ $count -eq 0 ]; then
    echo "New definition ✓"
  elif [ $count -eq 1 ]; then
    echo "Already exists - DUPLICATE?"

    # Get existing location
    grep -A 5 "^\*\*$label\*\*" docs/glossary.md
  else
    echo "Multiple entries ($count) - PROBLEM!"
  fi
done < def_labels.txt
```

#### Step 4.3: Compare Duplicate Definitions

**If duplicate found:**

```bash
# Extract definition from current document
LABEL="obj-euclidean-gas"
grep -A 30 ":label: $LABEL" "$DOCUMENT" > /tmp/current_def.txt

# Find existing definition
existing_doc=$(grep -A 3 "^\*\*$LABEL\*\*" docs/glossary.md | \
  grep "Source:" | \
  cut -d: -f2 | \
  awk '{print $1}')

grep -A 30 ":label: $LABEL" "docs/source/1_euclidean_gas/$existing_doc" > /tmp/existing_def.txt

# Compare
echo "=== Current Definition ==="
cat /tmp/current_def.txt
echo ""
echo "=== Existing Definition ==="
cat /tmp/existing_def.txt

# Manual comparison:
# - Are they identical? → Remove duplicate
# - Are they different? → Conflict needs resolution
```

#### Step 4.4: Verify Definition Dependencies

**Check if referenced entities exist:**

```bash
# Extract references from definition
grep -A 30 ":label: obj-your-definition" "$DOCUMENT" | \
  grep -o '{prf:ref}`[^`]*`' | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/' | \
  sort -u > def_dependencies.txt

# Validate each dependency
echo "Checking definition dependencies..."
while read dep; do
  echo -n "  $dep: "
  grep -q "^\*\*$dep\*\*" docs/glossary.md && echo "✓" || echo "✗ MISSING"
done < def_dependencies.txt
```

---

### Stage 5: Contradiction Detection

**Purpose**: Identify mathematical contradictions

#### Step 5.1: Automated Contradiction Checks

**Check for conflicting numerical bounds:**

```bash
# Extract all bounds for constant L
grep -n "L \(≤\|\\leq\|<=\) [0-9]" "$DOCUMENT" > L_bounds.txt

echo "Bounds on L:"
cat L_bounds.txt

# Manual check: Are all bounds compatible?
# Example:
# Line 45: L ≤ 1
# Line 120: L ≤ 2
# → If both refer to same L, potential contradiction
```

**Check for conflicting properties:**

```bash
# Search for contradictory claims
grep -n "continuous" "$DOCUMENT" > claims_continuous.txt
grep -n "discontinuous" "$DOCUMENT" > claims_discontinuous.txt

# If both non-empty, check if referring to same object
if [ -s claims_continuous.txt ] && [ -s claims_discontinuous.txt ]; then
  echo "WARNING: Found both continuity and discontinuity claims"
  echo "Continuous:"
  cat claims_continuous.txt
  echo "Discontinuous:"
  cat claims_discontinuous.txt
fi
```

#### Step 5.2: Cross-Document Consistency

**Compare claims across multiple documents:**

```bash
# Find all documents mentioning same concept
CONCEPT="euclidean-gas"
find docs/source/1_euclidean_gas -name "*.md" -exec grep -l "$CONCEPT" {} \; > related_docs.txt

echo "Documents mentioning $CONCEPT:"
cat related_docs.txt

# For each document, extract claims about concept
while read doc; do
  echo ""
  echo "=== $doc ==="
  grep -B 2 -A 2 "$CONCEPT" "$doc" | head -20
done < related_docs.txt

# Manual comparison: Are claims consistent?
```

#### Step 5.3: Verify Against Framework Axioms

**Check if claims are compatible with framework axioms:**

```bash
# Example: Document claims "L = 2"
# Check framework axioms about L

echo "Checking framework axioms for L..."
cat docs/glossary.md | grep "axiom-" | grep -i "lipschitz"

# Read each axiom
cat docs/glossary.md | grep -A 3 "axiom-lipschitz"

# If axiom states "L ≤ 1" but document claims "L = 2"
# → Contradiction detected
```

---

### Stage 6: Dual-Review Validation

**Purpose**: Comprehensive review using both Gemini + Codex

#### Step 6.1: Prepare Review Sections

```bash
# Extract key sections for review
mkdir -p /tmp/review_sections

# Extract definitions
grep -A 30 '{prf:definition}' "$DOCUMENT" > /tmp/review_sections/definitions.md

# Extract theorems
grep -A 50 '{prf:theorem}' "$DOCUMENT" > /tmp/review_sections/theorems.md

# Extract axiom usages (with context)
grep -B 5 -A 5 '{prf:ref}`axiom-' "$DOCUMENT" > /tmp/review_sections/axiom_contexts.md
```

#### Step 6.2: Draft Review Prompt

```bash
cat > /tmp/consistency_review_prompt.txt <<'EOF'
Verify framework consistency for this mathematical content:

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

Document sections to review:
[Paste content here]

Framework reference: docs/glossary.md (741 entries)
EOF

cat /tmp/consistency_review_prompt.txt
```

#### Step 6.3: Submit to Both Reviewers

**In Claude Code, submit IDENTICAL prompts:**

1. **Gemini 2.5 Pro**: `mcp__gemini-cli__ask-gemini` (model: "gemini-2.5-pro")
2. **Codex**: `mcp__codex__codex`

**Run in parallel** (single message, two tool calls).

#### Step 6.4: Compare Review Results

**Wait for both reviews, then analyze:**

```bash
# Save review results
cat > /tmp/gemini_review.txt <<'EOF'
[Gemini's review output]
EOF

cat > /tmp/codex_review.txt <<'EOF'
[Codex's review output]
EOF

# Identify consensus issues
echo "=== Consensus Issues (both reviewers agree) ==="
# [Manual comparison]

# Identify contradictions
echo "=== Contradictions (reviewers disagree) ==="
# [Manual comparison]

# Identify unique issues
echo "=== Gemini-Only Issues ==="
# [Manual comparison]

echo "=== Codex-Only Issues ==="
# [Manual comparison]
```

---

### Stage 7: Issue Resolution

**Purpose**: Fix identified consistency issues

#### Step 7.1: Prioritize Issues

```bash
# Create issue list
cat > /tmp/issues.txt <<'EOF'
CRITICAL:
- [Issue 1: Description]
- [Issue 2: Description]

MAJOR:
- [Issue 3: Description]

MINOR:
- [Issue 4: Description]
EOF

# Work through by priority
```

#### Step 7.2: Fix Each Issue

**For reference errors:**
```bash
vim "$DOCUMENT"
# Fix typo or add missing definition
```

**For notation conflicts:**
```bash
vim "$DOCUMENT"
# Replace conflicting notation with framework standard
```

**For axiom issues:**
```bash
vim "$DOCUMENT"
# Add missing axiom reference or verify hypotheses
```

**For contradictions:**
```bash
# Investigate both claims
# Determine which is correct
# Update incorrect claim
# Add justification
```

#### Step 7.3: Re-verify After Fixes

```bash
# Re-run automated checks
./check_consistency.sh

# Re-run specific checks for fixed issues
# Example: Check axiom references again
grep -o '{prf:ref}`axiom-[^`]*`' "$DOCUMENT" | \
  sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | \
  sort -u | \
  while read axiom; do
    grep -q "^\*\*$axiom\*\*" docs/glossary.md && echo "✓ $axiom" || echo "✗ $axiom"
  done
```

---

## Final Verification

### Complete Checklist

```bash
# Run all checks
echo "=== Final Consistency Verification ==="

# 1. References
echo "1. All references valid?"
grep -o '{prf:ref}`[^`]*`' "$DOCUMENT" | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/' | \
  sort -u | \
  while read label; do
    grep -q "^\*\*$label\*\*" docs/glossary.md || echo "✗ $label MISSING"
  done
echo "  → Check output above"

# 2. Notation
echo "2. Notation consistent?"
grep -c '\\mathcal' "$DOCUMENT" | xargs echo "  Calligraphic symbols:"
grep -Ec '\\gamma|\\beta|\\sigma' "$DOCUMENT" | xargs echo "  Greek letters:"
echo "  → Manual review required"

# 3. Axioms
echo "3. All axioms valid?"
grep -o '{prf:ref}`axiom-[^`]*`' "$DOCUMENT" | \
  sed 's/{prf:ref}`\(axiom-[^`]*\)`/\1/' | \
  sort -u | \
  while read axiom; do
    grep -q "^\*\*$axiom\*\*" docs/glossary.md || echo "✗ $axiom MISSING"
  done
echo "  → Check output above"

# 4. Definitions
echo "4. No duplicate definitions?"
grep -o ':label: obj-[^[:space:]]*' "$DOCUMENT" | \
  sed 's/:label: //' | \
  sort -u | \
  while read label; do
    count=$(grep -c "^\*\*$label\*\*" docs/glossary.md)
    [ $count -gt 0 ] && echo "  ! $label exists"
  done
echo "  → Check output above"

# 5. Build
echo "5. Document builds?"
make build-docs 2>&1 | grep -E "WARNING|ERROR" && echo "  ✗ Build issues" || echo "  ✓ Build clean"

echo ""
echo "=== Verification Complete ==="
```

---

## Time Estimates

| Stage | Time | Notes |
|-------|------|-------|
| Reference validation | ~5 min | Mostly automated |
| Notation audit | ~15 min | Requires manual review |
| Axiom validation | ~20 min | Manual verification needed |
| Definition cross-check | ~10 min | Automated + manual |
| Contradiction detection | ~15 min | Manual investigation |
| Dual-review | ~30 min | Parallel execution |
| Issue resolution | ~30-60 min | Depends on issues |
| **Total** | **~2-2.5 hours** | For typical document |

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [QUICKSTART.md](./QUICKSTART.md) - Copy-paste commands
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
