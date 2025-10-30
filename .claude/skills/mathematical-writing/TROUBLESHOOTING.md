# Mathematical Writing - Troubleshooting

Common issues and solutions for mathematical documentation.

---

## Formatting Issues

### Issue: Math not rendering

**Symptoms**: Raw LaTeX visible instead of rendered equations

**Example**:
```
$$
x = y
$$
```
Shows as literal `$$` in browser.

**Cause**: Missing blank line before opening `$$`

**Solution**:
```bash
python src/tools/fix_math_formatting.py docs/source/1_euclidean_gas/your_document.md
```

**Manual fix**:
```markdown
<!-- WRONG -->
We have the following result:
$$
x = y
$$

<!-- CORRECT -->
We have the following result:

$$
x = y
$$
```

---

### Issue: Inline math broken

**Symptoms**: Text like `$x$` not rendering as math

**Cause**: Backticks used instead of dollar signs

**Solution**:
```bash
python src/tools/convert_backticks_to_math.py docs/source/1_euclidean_gas/your_document.md
```

**Manual fix**:
```markdown
<!-- WRONG -->
Let `x` be a variable.

<!-- CORRECT -->
Let $x$ be a variable.
```

---

### Issue: Complex subscripts broken

**Symptoms**: `x_i_j` renders as `x_i` with `_j` as text

**Cause**: Multiple underscores without braces

**Solution**:
```bash
python src/tools/fix_complex_subscripts.py docs/source/1_euclidean_gas/your_document.md
```

**Manual fix**:
```latex
<!-- WRONG -->
$x_i_j$

<!-- CORRECT -->
$x_{i,j}$ or $x_{ij}$
```

---

### Issue: Unicode math not converting

**Symptoms**: Greek letters show as Unicode characters, not LaTeX

**Example**: α instead of $\alpha$

**Cause**: Pasted Unicode characters directly

**Solution**:
```bash
python src/tools/convert_unicode_math.py docs/source/1_euclidean_gas/your_document.md
```

**Manual fix**:
```markdown
<!-- WRONG -->
The friction coefficient γ...

<!-- CORRECT -->
The friction coefficient $\gamma$...
```

---

## Directive Issues

### Issue: Directive not rendering

**Symptoms**: Raw `:::{prf:theorem}` visible in output

**Cause**: Wrong number of colons (using `::::` instead of `:::`)

**Solution**: Use exactly 3 colons

```markdown
<!-- WRONG -->
::::{prf:theorem} My Theorem
:label: thm-my-theorem
...
::::

<!-- CORRECT -->
:::{prf:theorem} My Theorem
:label: thm-my-theorem
...
:::
```

---

### Issue: Missing theorem numbering

**Symptoms**: Theorems don't have numbers

**Cause**: Missing `:label:` field

**Solution**: Add label

```markdown
<!-- WRONG -->
:::{prf:theorem} My Theorem
...
:::

<!-- CORRECT -->
:::{prf:theorem} My Theorem
:label: thm-my-theorem
...
:::
```

---

### Issue: Directive syntax error

**Symptoms**: Build fails with directive parsing error

**Cause**: Missing closing `:::` or incorrect nesting

**Solution**: Check matching directive markers

```markdown
<!-- WRONG -->
:::{prf:theorem} My Theorem
:label: thm-my-theorem

:::{prf:proof}
...
:::
<!-- Missing closing ::: for theorem

<!-- CORRECT -->
:::{prf:theorem} My Theorem
:label: thm-my-theorem

Statement here.
:::

:::{prf:proof}
...
:::
```

---

## Cross-Reference Issues

### Issue: Broken cross-reference link

**Symptoms**: Link shows as `?` or is not clickable

**Cause**: Referenced label doesn't exist

**Solution**: Find correct label

```bash
# Search glossary
cat docs/glossary.md | grep "your-concept"

# Check document for label
grep ":label:" docs/source/1_euclidean_gas/your_document.md
```

**Fix reference**:
```markdown
<!-- WRONG -->
See {prf:ref}`thm-wrong-label`.

<!-- CORRECT (after finding real label) -->
See {prf:ref}`thm-actual-label`.
```

---

### Issue: Cross-reference shows wrong text

**Symptoms**: Reference displays wrong theorem name

**Cause**: Jupyter Book caches old content

**Solution**: Clean and rebuild

```bash
make clean
make build-docs
```

---

### Issue: Self-referencing label

**Symptoms**: Can't reference theorem from its own proof

**Cause**: Proof is inside theorem block

**Solution**: Put proof outside theorem

```markdown
<!-- WRONG -->
:::{prf:theorem} My Theorem
:label: thm-my-theorem

Statement.

:::{prf:proof}
By {prf:ref}`thm-my-theorem`... <!-- Can't self-reference
:::
:::

<!-- CORRECT -->
:::{prf:theorem} My Theorem
:label: thm-my-theorem

Statement.
:::

:::{prf:proof}
By {prf:ref}`thm-my-theorem`, we have... <!-- Works
:::
```

---

## Label Issues

### Issue: Label format rejected

**Symptoms**: Build warning about label format

**Cause**: Uppercase or underscores in label

**Solution**: Use lowercase kebab-case

```markdown
<!-- WRONG -->
:label: Thm-MyTheorem
:label: thm_my_theorem

<!-- CORRECT -->
:label: thm-my-theorem
```

---

### Issue: Duplicate label

**Symptoms**: Build error about duplicate label

**Cause**: Same label used twice

**Solution**: Make labels unique

```bash
# Find duplicate
grep -r ":label: thm-convergence" docs/source/

# Rename one of them
:label: thm-convergence-rate
:label: thm-convergence-time
```

---

### Issue: Wrong label prefix

**Symptoms**: Label doesn't follow convention

**Cause**: Used wrong prefix for entity type

**Solution**: Use correct prefix

```markdown
<!-- Definitions (mathematical objects) -->
:label: obj-euclidean-gas

<!-- Theorems -->
:label: thm-main-result

<!-- Lemmas -->
:label: lem-helper-result

<!-- Axioms -->
:label: axiom-bounded-cloning

<!-- Propositions -->
:label: prop-intermediate-result

<!-- Corollaries -->
:label: cor-immediate-consequence
```

---

## Build Issues

### Issue: Build fails with Jupyter Book error

**Symptoms**: `make build-docs` exits with error

**Cause**: Various (check error message)

**Solution**: Read error message carefully

```bash
# Build and capture output
make build-docs 2>&1 | tee build_log.txt

# Check for specific errors
grep -E "ERROR|CRITICAL" build_log.txt
```

**Common fixes:**
- Missing blank line before `$$` → run `fix_math_formatting.py`
- Invalid directive → check syntax (3 colons, proper nesting)
- Broken reference → check label exists

---

### Issue: Build succeeds but warnings appear

**Symptoms**: Build completes but shows warnings

**Example warning**:
```
WARNING: undefined label: thm-missing
```

**Solution**: Fix warnings before committing

```bash
# Find all warnings
make build-docs 2>&1 | grep "WARNING"

# Fix each one
# - Undefined label → fix reference or add definition
# - Missing directive → check syntax
```

---

### Issue: Build is slow

**Symptoms**: `make build-docs` takes >5 minutes

**Cause**: Large documents or many dependencies

**Solution**: Build specific document during development

```bash
# Instead of full build
make build-docs

# Build single document (faster)
jupyter-book build docs/source --path-output docs/source/1_euclidean_gas/your_document.md
```

---

## Notation Issues

### Issue: Notation inconsistent with framework

**Symptoms**: Dual-review flags notation mismatch

**Cause**: Used different symbols than glossary

**Solution**: Check glossary and fix

```bash
# Check glossary for correct notation
cat docs/glossary.md | grep "state space"

# Found: Uses $\mathcal{X}$ for state space

# Fix in document
sed -i 's/X_state/\\mathcal{X}/g' docs/source/1_euclidean_gas/your_document.md
```

---

### Issue: Symbol undefined

**Symptoms**: Symbol used without definition

**Cause**: Forgot to introduce notation

**Solution**: Add notation section or define inline

```markdown
<!-- Add to definition -->
:::{prf:definition} State Space
:label: obj-state-space

...

**Notation**:
- $\mathcal{X}$: The state space
- $d_\mathcal{X}$: Metric on $\mathcal{X}$
- $x, y \in \mathcal{X}$: Points in state space
:::

<!-- Or define inline -->
Let $\mathcal{X}$ denote the state space...
```

---

### Issue: Overloaded notation

**Symptoms**: Same symbol means different things

**Cause**: Reused symbol for different concepts

**Solution**: Use distinct symbols

```markdown
<!-- WRONG -->
Let $N$ be the number of walkers.
...
Let $N$ be the norm operator. <!-- Collision!

<!-- CORRECT -->
Let $N$ be the number of walkers.
...
Let $\|\cdot\|$ be the norm operator.
```

---

## Dual-Review Issues

### Issue: Reviewers provide contradictory feedback

**Symptoms**: Gemini says X, Codex says opposite of X

**Example**:
- Gemini: "Proof is incomplete, missing Step 3"
- Codex: "Proof is complete and rigorous"

**Cause**: This is normal and expected (not a bug)

**Solution**: Investigate manually

```bash
# Check the specific claim against framework
cat docs/glossary.md | grep "related-lemma"

# Read source document
cat docs/source/1_euclidean_gas/framework.md | grep -A 20 "related-lemma"

# Make evidence-based decision
# - If Gemini is correct: Fix the gap
# - If Codex is correct: Keep current version
# - If unclear: Ask user for guidance
```

**This is a feature**: Contradictions help identify subtle issues.

---

### Issue: Review finds no issues but proof seems incomplete

**Symptoms**: Both reviewers say "looks good" but you spot gaps

**Cause**: Reviewers missed issue (can happen)

**Solution**: Trust your judgment

```bash
# Manually verify specific claim
# - Check against axioms
# - Verify derivation steps
# - Test with examples

# If you find issue, fix it despite review
vim docs/source/1_euclidean_gas/your_document.md

# Re-submit for focused review
# "Please specifically review Step 3 derivation.
#  I believe there may be a gap in the bound."
```

---

### Issue: Review suggests incorrect fix

**Symptoms**: Proposed fix is mathematically wrong

**Cause**: Reviewer hallucination or misunderstanding

**Solution**: Critically evaluate and reject if wrong

**Example scenario**:
```
Gemini suggests: "Use $L \leq 2$ instead of $L \leq 1$"

Your analysis:
1. Check framework: Axiom states $L \leq 1$ (verified in docs/glossary.md)
2. Proposed change contradicts axiom
3. Reject Gemini's suggestion

Response to user:
"I disagree with Gemini's suggestion because it contradicts
{prf:ref}`axiom-bounded-lipschitz` which explicitly requires $L \leq 1$.
I've verified this in the framework documents. I recommend keeping
the current bound."
```

---

## Content Issues

### Issue: Proof is too terse

**Symptoms**: Dual-review flags "proof lacks detail"

**Cause**: Skipped intermediate steps

**Solution**: Expand derivation

```markdown
<!-- WRONG (too terse) -->
By standard analysis, we have $x \leq y$.

<!-- CORRECT (detailed) -->
**Step 2a**: Apply the triangle inequality

$$
\|x - z\| \leq \|x - y\| + \|y - z\|
$$

**Step 2b**: Use hypothesis that $\|x - y\| \leq \epsilon$

From Step 2a and the hypothesis, we obtain:

$$
\|x - z\| \leq \epsilon + \|y - z\|
$$

**Step 2c**: Conclude

Since $\|y - z\| \leq C$ by {prf:ref}`lem-bounded-distance`, we have:

$$
\|x - z\| \leq \epsilon + C
$$

This establishes the bound.
```

---

### Issue: Definition too vague

**Symptoms**: Review flags "definition not precise"

**Cause**: Missing formal mathematical statement

**Solution**: Add formal definition

```markdown
<!-- WRONG (vague) -->
:::{prf:definition} Cloning Operator
:label: obj-cloning-operator

The cloning operator duplicates walkers based on fitness.
:::

<!-- CORRECT (precise) -->
:::{prf:definition} Cloning Operator
:label: obj-cloning-operator

**Informal Description**:
The cloning operator duplicates walkers based on fitness.

**Formal Definition**:
Let $\mathcal{S}_N$ denote the swarm configuration space. The **cloning operator**
$\Psi_{\text{clone}}: \mathcal{S}_N \to \mathcal{S}_N$ is defined by:

$$
\Psi_{\text{clone}}(\mathcal{S}) := \{w_1', \ldots, w_N'\}
$$

where for each $i \in \{1, \ldots, N\}$:
- Select companion $j$ with probability $p_{ij} \propto \exp(\beta V(w_j))$
- Set $w_i' := \text{clone}(w_i, w_j)$ with inelastic collision dynamics

**Properties**:
1. Preserves swarm size: $|\Psi_{\text{clone}}(\mathcal{S})| = N$
2. Fitness-biased: Higher fitness walkers more likely to be companions
:::
```

---

### Issue: Missing references

**Symptoms**: Review flags "claim not justified"

**Cause**: Assertion without proof or reference

**Solution**: Add justification

```markdown
<!-- WRONG (unjustified) -->
The operator is Lipschitz continuous.

<!-- CORRECT (justified) -->
The operator is Lipschitz continuous by {prf:ref}`lem-operator-lipschitz`
with constant $L = 1$.
```

---

## Preview Issues

### Issue: Local preview not working

**Symptoms**: `make serve-docs` fails or page doesn't load

**Cause**: Port already in use or build incomplete

**Solution**: Check port and rebuild

```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing server
pkill -f "jupyter-book serve"

# Rebuild and serve
make clean
make build-docs
make serve-docs
```

---

### Issue: Changes not reflected in preview

**Symptoms**: Edited document but browser shows old version

**Cause**: Browser cache or build cache

**Solution**: Clean rebuild and hard refresh

```bash
# Clean build
make clean
make build-docs
make serve-docs

# In browser: Ctrl+Shift+R (hard refresh)
```

---

## Publishing Issues

### Issue: Mermaid diagrams not rendering

**Symptoms**: Mermaid blocks show as code, not diagrams

**Cause**: Mermaid blocks not converted to MyST format

**Solution**: Convert before build

```bash
# Convert mermaid blocks
python src/tools/convert_mermaid_blocks.py docs/source/1_euclidean_gas/your_document.md --in-place

# Rebuild
make build-docs
```

---

### Issue: PDF export fails

**Symptoms**: `jupyter-book build --builder pdflatex` fails

**Cause**: LaTeX incompatibility or missing packages

**Solution**: Check LaTeX log

```bash
# Try PDF build
jupyter-book build docs/source --builder pdflatex

# Check log for errors
cat docs/source/_build/latex/*.log | grep ERROR

# Common fixes:
# - Install missing LaTeX packages
# - Simplify complex math expressions
# - Use HTML export instead
```

---

## Getting Help

If issues persist:

1. **Check CLAUDE.md**: Mathematical writing standards
2. **Check GEMINI.md**: Review protocol details
3. **Consult glossary**: `docs/glossary.md` for notation
4. **Read source docs**: Full framework documentation
5. **Ask Claude**: Describe issue with specific examples

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [QUICKSTART.md](./QUICKSTART.md) - Copy-paste commands
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
