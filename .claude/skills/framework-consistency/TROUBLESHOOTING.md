# Framework Consistency - Troubleshooting

Common consistency issues and solutions.

---

## Reference Issues

### Issue: Reference not found in glossary

**Symptoms**: `{prf:ref}`thm-convergance-rate`` but label doesn't exist

**Cause**: Typo in label or entity truly missing

**Solution 1 - Typo**:
```bash
# Search for similar labels
cat docs/glossary.md | grep -i "convergence" | grep "thm-"

# Found: thm-convergence-rate (correct spelling)

# Fix in document
vim docs/source/1_euclidean_gas/your_document.md
# Change convergance → convergence
```

**Solution 2 - Missing entity**:
```bash
# Check if concept should exist
cat docs/glossary.md | grep -i "your-concept"

# If not found, need to define it first
# Use mathematical-writing skill to create definition
```

---

### Issue: Wrong label prefix

**Symptoms**: Reference exists but wrong type

**Example**: `{prf:ref}`def-euclidean-gas`` but should be `obj-euclidean-gas`

**Cause**: Used wrong prefix for entity type

**Solution**: Check glossary for correct prefix
```bash
# Find correct label
cat docs/glossary.md | grep "euclidean-gas"

# Update reference
vim your_document.md
# Change def- → obj-
```

**Label prefix reference:**
| Type | Prefix |
|------|--------|
| Definition | `obj-` |
| Theorem | `thm-` |
| Lemma | `lem-` |
| Axiom | `axiom-` |
| Proposition | `prop-` |
| Corollary | `cor-` |

---

### Issue: Circular reference

**Symptoms**: Definition A references definition B, which references A

**Example**:
- `obj-state-space` references `obj-metric`
- `obj-metric` references `obj-state-space`

**Cause**: Improper dependency ordering

**Solution**: Restructure definitions
```markdown
<!-- WRONG -->
:::{prf:definition} State Space
Requires metric from {prf:ref}`obj-metric`.
:::

:::{prf:definition} Metric
Defined on {prf:ref}`obj-state-space`.
:::

<!-- CORRECT -->
:::{prf:definition} State Space
A state space is a set $\mathcal{X}$ equipped with a metric d.
:::

:::{prf:definition} Metric on State Space
Given {prf:ref}`obj-state-space`, the metric d satisfies...
:::
```

---

## Notation Issues

### Issue: Overloaded symbol

**Symptoms**: Same symbol used for different concepts

**Example**: $N$ means both "number of walkers" and "norm operator"

**Cause**: Symbol reused without disambiguation

**Solution**: Use distinct notation
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

**Framework standard symbols** (don't reuse):
- `$N$`: Number of walkers
- `$d$`: Distance/metric
- `$\gamma$`: Friction coefficient
- `$\beta$`: Exploitation weight
- `$\sigma$`: Perturbation noise

---

### Issue: Undefined symbol

**Symptoms**: Symbol used without definition

**Example**: $\lambda_v$ appears without being introduced

**Cause**: Forgot to define notation

**Solution**: Add notation section
```markdown
:::{prf:definition} Your Object
:label: obj-your-object

[Content here]

**Notation**:
- $\lambda_v$: Velocity weight parameter
- $\epsilon_F$: Fitness force regularization
:::
```

Or define inline:
```markdown
Let $\lambda_v \in [0,1]$ denote the **velocity weight parameter**.
```

---

### Issue: Notation conflicts with framework

**Symptoms**: Document uses $\mathcal{X}$ for algorithmic space, but framework uses it for state space

**Cause**: Not following glossary conventions

**Solution**: Update to match framework
```bash
# Check framework usage
cat docs/glossary.md | grep -A 3 "obj-state-space"
# → Uses $\mathcal{X}$ for state space

# Fix document
vim your_document.md
# Change notation to match framework
# Use $\mathcal{Y}$ for algorithmic space instead
```

**Framework notation standards:**
| Symbol | Meaning | Don't use for |
|--------|---------|---------------|
| `$\mathcal{X}$` | State space | Algorithmic space |
| `$\mathcal{Y}$` | Algorithmic space | State space |
| `$\mathcal{S}$` | Swarm config | Anything else |
| `$\gamma$` | Friction | Other params |
| `$N$` | Walker count | Norm operator |

---

### Issue: Inconsistent subscript notation

**Symptoms**: Sometimes uses $x_i$, sometimes $x^{(i)}$

**Cause**: Mixed notation styles

**Solution**: Unify notation
```markdown
<!-- Choose one style and stick to it -->

<!-- Style 1: Subscript for index -->
$x_i$, $v_i$, $w_i$

<!-- Style 2: Superscript for time -->
$x^{(t)}$, $x^{(t+1)}$

<!-- Don't mix -->
$x_i^{(t)}$ for walker i at time t  <!-- OK, different indices
```

---

## Axiom Issues

### Issue: Axiom doesn't exist

**Symptoms**: References `axiom-wrong-name` but not in glossary

**Cause**: Typo or axiom truly missing

**Solution 1 - Typo**:
```bash
# Find similar axioms
cat docs/glossary.md | grep "axiom-" | grep "bounded"

# Found: axiom-bounded-cloning
# Fix reference
```

**Solution 2 - Missing axiom**:
```bash
# Check if axiom should exist
cat docs/glossary.md | grep -i "your-concept" | grep "axiom-"

# If truly missing, need to add axiom definition
# This is rare - most axioms already defined in framework
```

---

### Issue: Axiom hypotheses not satisfied

**Symptoms**: References axiom but doesn't establish required assumptions

**Example**:
- Axiom requires: "Complete metric space"
- Document: Doesn't establish completeness

**Cause**: Incomplete proof or wrong axiom

**Solution 1 - Establish hypothesis**:
```markdown
**Step 1: Establish completeness**

By {prf:ref}`lem-space-complete`, the space $(\mathcal{X}, d_\mathcal{X})$
is complete.

**Step 2: Apply axiom**

Since completeness holds, we can apply {prf:ref}`axiom-name`.
```

**Solution 2 - Use different axiom**:
```bash
# Find axiom with weaker requirements
cat docs/glossary.md | grep "axiom-" | grep -i "your-topic"
```

---

### Issue: Axiom misapplied

**Symptoms**: Uses axiom conclusion incorrectly

**Example**:
- Axiom provides: "Operator is Lipschitz with L ≤ 1"
- Document claims: "By axiom, L = 2"  ← Contradiction!

**Cause**: Misunderstanding axiom statement

**Solution**: Re-read axiom carefully
```bash
# Get axiom source
axiom="axiom-bounded-lipschitz"
source_doc=$(grep -A 3 "^\*\*$axiom\*\*" docs/glossary.md | \
  grep "Source:" | \
  cut -d: -f2 | \
  awk '{print $1}')

# Read full axiom
grep -A 30 ":label: $axiom" docs/source/1_euclidean_gas/$source_doc

# Fix document to match axiom conclusion
```

---

### Issue: Missing axiom citation

**Symptoms**: Makes claim without referencing supporting axiom

**Example**: "The operator is Lipschitz continuous" without citing axiom

**Cause**: Forgot to add reference

**Solution**: Add axiom citation
```markdown
<!-- WRONG -->
The operator is Lipschitz continuous.

<!-- CORRECT -->
The operator is Lipschitz continuous by {prf:ref}`axiom-lipschitz-continuity`
with constant $L = 1$.
```

---

## Definition Issues

### Issue: Duplicate definition

**Symptoms**: Defines concept that already exists in framework

**Example**: Defining "Euclidean Gas" when `obj-euclidean-gas` exists

**Cause**: Didn't check glossary first

**Solution**: Remove duplicate, reference existing
```bash
# Check if definition exists
cat docs/glossary.md | grep -i "euclidean-gas" | grep "obj-"

# Found: obj-euclidean-gas in 02_euclidean_gas.md

# Remove duplicate from document
vim your_document.md
# Delete duplicate definition
# Replace with reference: {prf:ref}`obj-euclidean-gas`
```

---

### Issue: Conflicting definitions

**Symptoms**: New definition contradicts existing one

**Example**:
- Existing: "Euclidean Gas uses Langevin dynamics"
- New: "Euclidean Gas uses Hamiltonian dynamics"  ← Conflict!

**Cause**: Misunderstanding or genuine contradiction

**Solution**: Investigate and resolve
```bash
# Read existing definition
existing_label="obj-euclidean-gas"
source_doc=$(grep -A 3 "^\*\*$existing_label\*\*" docs/glossary.md | \
  grep "Source:" | \
  cut -d: -f2 | \
  awk '{print $1}')

grep -A 30 ":label: $existing_label" docs/source/1_euclidean_gas/$source_doc

# Determine which is correct
# Update incorrect definition
# Add note explaining resolution
```

---

### Issue: Definition references non-existent entity

**Symptoms**: Definition uses `{prf:ref}` to missing label

**Cause**: Dependency not defined yet or typo

**Solution**: Check dependencies
```bash
# Extract references from definition
grep -A 30 ":label: obj-your-definition" your_document.md | \
  grep -o '{prf:ref}`[^`]*`' | \
  sed 's/{prf:ref}`\([^`]*\)`/\1/'

# Check each exists
# ... (validate against glossary)

# Fix: Either fix typo or define missing dependency first
```

---

## Contradiction Issues

### Issue: Conflicting numerical bounds

**Symptoms**: Different bounds for same constant

**Example**:
- Line 45: "L ≤ 1"
- Line 120: "L ≤ 2"

**Cause**: Inconsistent statements or different contexts

**Solution 1 - Same context (contradiction)**:
```markdown
<!-- Determine which bound is correct -->
<!-- Check proofs and axioms -->
<!-- Update incorrect bound -->

<!-- If L ≤ 1 is correct: -->
By {prf:ref}`axiom-lipschitz`, we have $L \leq 1$.
```

**Solution 2 - Different contexts (clarify)**:
```markdown
<!-- Distinguish the constants -->
The cloning operator has Lipschitz constant $L_{\text{clone}} \leq 1$.
...
The perturbation operator has Lipschitz constant $L_{\text{pert}} \leq 2$.
```

---

### Issue: Conflicting properties

**Symptoms**: Claims both continuous and discontinuous

**Example**:
- "The operator is continuous"
- "The operator has a discontinuity at x = 0"

**Cause**: Imprecise language or genuine contradiction

**Solution**: Clarify domain
```markdown
<!-- WRONG -->
The operator is continuous.
...
The operator is discontinuous at $x = 0$.

<!-- CORRECT -->
The operator is continuous on $\mathcal{X} \setminus \{0\}$.

At $x = 0$, the operator has a jump discontinuity.
```

---

### Issue: Incompatible assumptions

**Symptoms**: Assumes both compactness and unboundedness

**Example**:
- "Assume $\mathcal{X}$ is compact"
- "Let $\mathcal{X} = \mathbb{R}^d$" ← Contradiction!

**Cause**: Conflicting hypotheses

**Solution**: Reconcile assumptions
```markdown
<!-- WRONG -->
Assume $\mathcal{X}$ is compact.
...
Let $\mathcal{X} = \mathbb{R}^d$.  <!-- R^d not compact!

<!-- CORRECT -->
Assume $\mathcal{X} \subset \mathbb{R}^d$ is compact.

For example, $\mathcal{X} = [-1,1]^d$ satisfies this.
```

---

## Dual-Review Issues

### Issue: Reviewers provide contradictory feedback

**Symptoms**: Gemini says X, Codex says opposite

**Example**:
- Gemini: "Notation $d_\mathcal{X}$ is inconsistent"
- Codex: "Notation $d_\mathcal{X}$ matches framework"

**Cause**: This is normal (not a bug)

**Solution**: Verify manually
```bash
# Check glossary
cat docs/glossary.md | grep -A 3 "obj-metric"

# Determine ground truth
# Make evidence-based decision
# Document reasoning
```

**This is a feature** - contradictions help identify subtle issues.

---

### Issue: Both reviewers miss issue

**Symptoms**: You spot problem that reviewers didn't catch

**Example**: Obvious typo in label both reviewers missed

**Cause**: Reviewers can miss things (especially non-mathematical issues)

**Solution**: Trust your judgment
```bash
# Fix the issue
vim your_document.md

# Document what you fixed
# No need to re-review for obvious fixes
```

---

### Issue: Reviewer suggests incorrect fix

**Symptoms**: Proposed fix contradicts framework

**Example**: "Use L ≤ 2" but framework requires L ≤ 1

**Cause**: Reviewer hallucination

**Solution**: Reject incorrect feedback
```bash
# Verify against framework
cat docs/glossary.md | grep -A 5 "axiom-lipschitz"

# Document why you're rejecting feedback
echo "Rejecting suggestion because:"
echo "1. Framework axiom requires L ≤ 1"
echo "2. Changing to L ≤ 2 contradicts axiom"
echo "3. Keeping L ≤ 1"
```

---

## Tool and Script Issues

### Issue: Grep command not finding matches

**Symptoms**: `grep "pattern" file` returns nothing but matches exist

**Cause**: Pattern escaping or file path wrong

**Solution 1 - Check escaping**:
```bash
# LaTeX commands need escaping
# WRONG:
grep "\gamma" file  # Escapes the backslash

# CORRECT:
grep '\\gamma' file  # Single quotes preserve backslash
```

**Solution 2 - Check file path**:
```bash
# Verify file exists
ls -l docs/source/1_euclidean_gas/your_document.md

# Use absolute path if needed
grep "pattern" /full/path/to/file
```

---

### Issue: Script reports false positives

**Symptoms**: Validation script says reference missing but it exists

**Cause**: Pattern matching too strict

**Example**:
```bash
# Script looks for exact format:
grep "^\*\*thm-label\*\*" glossary.md

# But glossary might have extra whitespace:
"  **thm-label**"

# Solution: Make pattern more flexible
grep "^[[:space:]]*\*\*thm-label\*\*" glossary.md
```

---

### Issue: Automated check takes too long

**Symptoms**: Consistency script runs for >5 minutes

**Cause**: Processing large document or many files

**Solution**: Optimize script
```bash
# Instead of:
while read label; do
  grep "$label" large_file.md  # Reads file repeatedly
done < labels.txt

# Use:
grep -f labels.txt large_file.md  # Single pass
```

---

## Workflow Issues

### Issue: Too many issues to fix

**Symptoms**: Consistency check finds 50+ issues

**Cause**: Large document or many inconsistencies

**Solution**: Prioritize and batch
```bash
# 1. Fix all CRITICAL issues first
# - Missing references (build will fail)
# - Contradictions with axioms

# 2. Then MAJOR issues
# - Notation conflicts
# - Duplicate definitions

# 3. Finally MINOR issues
# - Undefined symbols
# - Clarity improvements

# Work in batches of 10 issues
```

---

### Issue: Fixes introduce new issues

**Symptoms**: Fixing reference A breaks reference B

**Cause**: Cascading dependencies

**Solution**: Track changes carefully
```bash
# Before making changes
cp your_document.md your_document.md.backup

# After each batch of fixes
./check_consistency.sh > report_after_batch1.txt

# Compare with previous report
diff report_before.txt report_after_batch1.txt

# Verify no new issues introduced
```

---

### Issue: Glossary is out of date

**Symptoms**: Document references new entity not yet in glossary

**Cause**: Glossary not regenerated after extraction

**Solution**: This is expected for new content
```bash
# New definitions won't be in glossary until:
# 1. Document is finalized
# 2. Extract-and-refine workflow runs
# 3. Glossary regenerated

# For now: Skip validation for known new entities
# Mark them in tracking list:
echo "thm-new-result" >> /tmp/known_new_entities.txt
```

---

## Getting Help

If issues persist:

1. **Check glossary**: `docs/glossary.md` for framework reference
2. **Read CLAUDE.md**: Framework standards and conventions
3. **Review source documents**: Read full definitions/axioms
4. **Use dual-review**: Submit to Gemini + Codex for analysis
5. **Ask Claude**: Describe specific issue with context

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [QUICKSTART.md](./QUICKSTART.md) - Copy-paste commands
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- **docs/glossary.md** - Complete framework reference
