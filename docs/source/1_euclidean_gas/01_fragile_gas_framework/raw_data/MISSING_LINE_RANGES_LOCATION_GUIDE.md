# Location Guide for Entities Missing Line Ranges

**Document:** `01_fragile_gas_framework.md`
**Date:** 2025-10-31
**Total Entities Missing Line Ranges:** 33 (18.6% of 177 total)

This document provides detailed locations for all entities that currently lack precise line range metadata, organized by entity type. Use this guide to manually locate and optionally add directive blocks for these entities.

---

## Summary Statistics

- **Proofs:** 11 entities (most are inline after theorems - standard mathematical writing convention)
- **Parameters:** 10 entities (inline definitions within axiom text)
- **Lemmas:** 8 entities (sub-lemmas or inline variants)
- **Remarks:** 4 entities (inline notes and admonitions)

**Note:** These entities are intentionally inline content following standard mathematical writing practices. Adding formal directive blocks for all of them would require significant document restructuring.

---

## 1. PROOFS (11 entities)

### 1.1 proof-thm-potential-operator-is-mean-square-continuous

**File:** `proof-thm-potential-operator-is-mean-square-continuous.json`
**Proves:** Proof of Theorem: Potential Operator is Mean-Square Continuous
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "Potential Operator is Mean-Square Continuous" 01_fragile_gas_framework.md
grep -n "thm-potential-operator" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Potential Operator (likely §12)

**Recommendation:** Search for the theorem label `thm-potential-operator-is-mean-square-continuous`, then look for the proof block immediately following it. Add `:label: proof-thm-potential-operator-is-mean-square-continuous` to the proof directive.

---

### 1.2 proof-lem-cloning-probability-lipschitz

**File:** `proof-lem-cloning-probability-lipschitz.json`
**Proves:** Proof of Lemma: Cloning Probability Lipschitz
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "lem-cloning-probability-lipschitz" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Cloning Operator (likely §15)

**Recommendation:** Find `lem-cloning-probability-lipschitz` directive at line 4336, then add `:label:` to the proof block that follows it.

---

### 1.3 proof-thm-k1-revival-state

**File:** `proof-thm-k1-revival-state.json`
**Proves:** Proof of Theorem: K=1 Revival State
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "thm-k1-revival-state" 01_fragile_gas_framework.md
grep -n "Revival State" 01_fragile_gas_framework.md
```

**Expected Location:** Section §17 (The Revival State)

**Recommendation:** This theorem has label `thm-k1-revival-state`. Look for the proof block after this theorem (likely within §17) and add `:label: proof-thm-k1-revival-state`.

---

### 1.4 proof-lem-total-clone-prob-value-error

**File:** `proof-lem-total-clone-prob-value-error.json`
**Proves:** Proof of Lemma: Total Clone Probability Value Error
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "lem-total-clone-prob-value-error" 01_fragile_gas_framework.md
grep -n "Total Clone Probability Value Error" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Cloning Continuity (likely §15)

**Recommendation:** The lemma is found at lines 4443-4464. Add `:label:` to the proof directive that follows.

---

### 1.5 proof-lem-total-clone-prob-structural-error

**File:** `proof-lem-total-clone-prob-structural-error.json`
**Proves:** Proof of Lemma: Total Clone Probability Structural Error
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "lem-total-clone-prob-structural-error" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Cloning Continuity (likely §15)

**Recommendation:** The lemma is found at lines 4430-4434. Add `:label:` to the proof directive that follows.

---

### 1.6 proof-sub-lem-bound-sum-total-cloning-probs

**File:** `proof-sub-lem-bound-sum-total-cloning-probs.json`
**Proves:** Proof of Sub-Lemma: Bound on Sum of Total Cloning Probabilities
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "sub-lem-bound-sum-total-cloning-probs" 01_fragile_gas_framework.md
grep -n "Bound on Sum of Total Cloning" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Cloning (likely §15, subsection on structural error)

**Recommendation:** This is a sub-lemma. Search for the parent lemma and locate the proof block inline.

---

### 1.7 proof-sub-lem-probabilistic-bound-perturbation-displacement-reproof

**File:** `proof-probabilistic-bound-perturbation-displacement.json`
**Proves:** Proof of Sub-Lemma: Probabilistic Bound Perturbation Displacement (Reproof)
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "sub-lem-probabilistic-bound-perturbation-displacement" 01_fragile_gas_framework.md
grep -n "McDiarmid" 01_fragile_gas_framework.md | head -20
```

**Expected Location:** Section §14 (Perturbation Operator), subsection on probabilistic bounds

**Recommendation:** This proof is likely embedded in the McDiarmid's inequality section. Look around line 4031 for `sub-lem-probabilistic-bound-perturbation-displacement-reproof` and add `:label:` to its proof.

---

### 1.8 proof-sub-lem-perturbation-positional-bound-reproof

**File:** `proof-perturbation-positional-bound.json`
**Proves:** Proof of Sub-Lemma: Perturbation Positional Bound (Reproof)
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "sub-lem-perturbation-positional-bound" 01_fragile_gas_framework.md
```

**Expected Location:** Section §14 (Perturbation Operator), around line 3979

**Recommendation:** Lemma found at line 3979. Add `:label:` to the proof directive that follows (around line 3988).

---

### 1.9 proof-thm-expected-cloning-action-continuity

**File:** `proof-thm-expected-cloning-action-continuity.json`
**Proves:** Proof of Theorem: Expected Cloning Action Continuity
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "thm-expected-cloning-action-continuity" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Cloning Operator (likely §15)

**Recommendation:** Find the theorem with this label and add `:label:` to its proof block.

---

### 1.10 proof-thm-total-expected-cloning-action-continuity

**File:** `proof-thm-total-expected-cloning-action-continuity.json`
**Proves:** Proof of Theorem: Total Expected Cloning Action Continuity
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "thm-total-expected-cloning-action-continuity" 01_fragile_gas_framework.md
```

**Expected Location:** Section on Cloning Operator (likely §15)

**Recommendation:** Find the theorem with this label and add `:label:` to its proof block.

---

### 1.11 proof-thm-perturbation-operator-continuity-reproof

**File:** `proof-perturbation-operator-continuity.json`
**Proves:** Proof of Theorem: Perturbation Operator Continuity (Reproof)
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "thm-perturbation-operator-continuity" 01_fragile_gas_framework.md
```

**Expected Location:** Section §14 (Perturbation Operator)

**Recommendation:** Find the theorem with this label and add `:label:` to its proof block.

---

## 2. PARAMETERS (10 entities)

**General Note:** Parameters are typically defined inline within axiom text or in descriptive paragraphs. To formalize them, you would need to create `:::{prf:definition}` blocks for each parameter.

### 2.1 param-n

**File:** `param-n.json`
**Label:** `param-n`
**Name:** Number of Walkers
**Symbol:** $N$
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "Number of Walkers" 01_fragile_gas_framework.md
grep -n "^N" 01_fragile_gas_framework.md | head -20
```

**Expected Location:** Early in document (§1-3), likely in introduction or axioms

**Recommendation:** Parameter $N$ is mentioned throughout but never formally defined in a directive. Consider creating:
```markdown
:::{prf:definition} Parameter: Number of Walkers
:label: param-n

**Symbol:** $N$

**Type:** Natural number

**Constraints:** $N \ge 1$

The number of walkers in the swarm.
:::
```

---

### 2.2 param-kappa-variance

**File:** `param-kappa-variance.json`
**Label:** `param-kappa-variance`
**Name:** Maximum Measurement Variance
**Symbol:** $\kappa^2_{\text{variance}}$
**Current Status:** Mentioned in axiom text, no standalone directive

**Search Strategy:**
```bash
grep -n "kappa.*variance" 01_fragile_gas_framework.md
grep -n "Maximum Measurement Variance" 01_fragile_gas_framework.md
```

**Expected Location:** Line ~2129 (Axiom of Bounded Measurement Variance)

**Recommendation:** Currently defined within the axiom text at line 2129. To formalize, extract into a separate parameter definition directive before the axiom.

---

### 2.3 param-f-v-ms

**File:** `param-f-v-ms.json`
**Label:** `param-f-v-ms`
**Name:** Expected Squared Value Error Bound
**Symbol:** $F_{V,ms}$
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "F_{V,ms}" 01_fragile_gas_framework.md
grep -n "Expected Squared Value Error" 01_fragile_gas_framework.md
```

**Expected Location:** Section on value error analysis (likely §11)

**Recommendation:** This is a bounding function mentioned in continuity theorems. Consider creating a definition directive if it needs standalone reference.

---

### 2.4 param-l-phi (from raw-param-005.json)

**File:** `raw-param-005.json`
**Label:** `param-l-phi`
**Symbol:** $L_φ$
**Section:** §1-introduction
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "L_\\\\varphi\|L_φ" 01_fragile_gas_framework.md | head -20
```

**Expected Location:** Introduction or early axioms

**Recommendation:** This appears to be the Lipschitz constant for the projection map $\varphi$. Extract into a parameter definition.

---

### 2.5 param-p-worst-case

**File:** `param-p-worst-case.json`
**Label:** `param-p-worst-case`
**Name:** Worst-Case Probability Parameter
**Symbol:** $p_{\text{worst-case}}$
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "p_.*worst.*case" 01_fragile_gas_framework.md
grep -n "Worst-Case Probability" 01_fragile_gas_framework.md
```

**Expected Location:** Section on cloning or measurement (likely §15)

**Recommendation:** Mentioned in structural continuity bounds. Consider creating a parameter definition if needed for cross-referencing.

---

### 2.6 NO_LABEL (refinement_report.json)

**File:** `refinement_report.json`
**Label:** NO_LABEL
**Current Status:** Invalid entity (report file, not a parameter)

**Recommendation:** This file should be removed or moved out of the parameters directory. It appears to be a processing artifact.

---

### 2.7 param-l-r

**File:** `param-l-r.json`
**Label:** `param-l-r`
**Name:** Reward Lipschitz Constant
**Symbol:** $L_R$
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "L_R\|L_{R}" 01_fragile_gas_framework.md
grep -n "Reward Lipschitz" 01_fragile_gas_framework.md
```

**Expected Location:** Early axioms or reward function definition (§3-5)

**Recommendation:** The Lipschitz constant of the reward function $R$. Consider creating a parameter definition in the section discussing reward measurement.

---

### 2.8 param-kappa-revival

**File:** `param-kappa-revival.json`
**Label:** `param-kappa-revival`
**Name:** Revival Guarantee Parameter
**Symbol:** $\kappa_{\text{revival}}$
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "kappa.*revival" 01_fragile_gas_framework.md
grep -n "Revival Guarantee" 01_fragile_gas_framework.md
```

**Expected Location:** Section §17 (The Revival State) or axioms

**Recommendation:** Mentioned in the revival state analysis. Extract from axiom text into a parameter definition.

---

### 2.9 param-l-phi (from param-l-phi.json)

**File:** `param-l-phi.json`
**Label:** `param-l-phi`
**Name:** Displacement Lipschitz Constant
**Symbol:** $L_\varphi$
**Current Status:** No directive block

**Note:** This is a duplicate of 2.4 but from a different JSON file.

**Recommendation:** Consolidate with entry 2.4. Only one parameter definition needed.

---

### 2.10 param-epsilon-std

**File:** `param-epsilon-std.json`
**Label:** `param-epsilon-std`
**Name:** Standardization Error Parameter
**Symbol:** $\varepsilon_{\text{std}}$
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "epsilon.*std\|varepsilon.*std" 01_fragile_gas_framework.md
grep -n "Standardization Error" 01_fragile_gas_framework.md
```

**Expected Location:** Section on standardization (§8-11)

**Recommendation:** This error parameter appears in standardization bounds. Consider creating a parameter definition if it needs standalone reference.

---

## 3. LEMMAS (8 entities)

### 3.1 lem-probabilistic-bound-perturbation-displacement-reproof

**File:** `lem-probabilistic-bound-perturbation-displacement.json`
**Label:** `lem-probabilistic-bound-perturbation-displacement-reproof`
**Statement:** Let $\mathcal{S}_{\text{in}}$ be an input swarm. Assume the **Axiom of Bounded Second Moment of Perturbation**...
**Current Status:** No line range found

**Search Strategy:**
```bash
grep -n "lem-probabilistic-bound-perturbation-displacement" 01_fragile_gas_framework.md
```

**Expected Location:** Section §14 (Perturbation Operator), subsection 13.2.3

**Recommendation:** This lemma exists but lacks a `:label:` line. Search for the lemma text and add `:label: lem-probabilistic-bound-perturbation-displacement-reproof` to its directive.

---

### 3.2 lem-unify-holder-terms

**File:** `sub-lem-unify-holder-terms.json`
**Label:** `lem-unify-holder-terms`
**Statement:** Multiple Hölder-type terms (√V, V^{1/3}, etc.) arising from different components can be unified into a single dominant term...
**Current Status:** No directive block (inline discussion)

**Search Strategy:**
```bash
grep -n "Hölder\|Holder" 01_fragile_gas_framework.md
grep -n "unify.*terms" 01_fragile_gas_framework.md
```

**Expected Location:** Technical subsection on bounding techniques

**Recommendation:** This is likely an inline mathematical observation rather than a formal lemma. Consider adding a remark or lemma directive if this technique is referenced elsewhere.

---

### 3.3 lem-bound-sum-total-cloning-probs

**File:** `sub-lem-bound-sum-total-cloning-probs.json`
**Label:** `lem-bound-sum-total-cloning-probs`
**Statement:** Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm states. Let $V_{\text{in}} := d_{\text{Disp},\mathcal{Y}}$...
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "sub-lem-bound-sum-total-cloning-probs" 01_fragile_gas_framework.md
grep -n "Bound on Sum of Total Cloning" 01_fragile_gas_framework.md
```

**Expected Location:** Section §15 (Cloning Operator), subsection on structural error

**Recommendation:** This is a sub-lemma used in a larger proof. Add a lemma directive with `:label: lem-bound-sum-total-cloning-probs`.

---

### 3.4 lem-inequality-toolbox

**File:** `lem-inequality-toolbox.json`
**Label:** `lem-inequality-toolbox`
**Statement:** Collection of standard inequalities: Triangle inequality, Cauchy-Schwarz, Hölder, Jensen, subadditivity...
**Current Status:** No directive block (reference collection)

**Search Strategy:**
```bash
grep -n "inequality.*toolbox\|Toolbox" 01_fragile_gas_framework.md -i
```

**Expected Location:** Appendix or early mathematical preliminaries

**Recommendation:** This appears to be a reference collection rather than a formal lemma. Consider creating an appendix section with a lemma directive listing standard inequalities used throughout.

---

### 3.5 lem-sigma-reg-derivative-bounds

**File:** `lem-sigma-reg-derivative-bounds.json`
**Label:** `lem-sigma-reg-derivative-bounds`
**Statement:** The regularized standard deviation $\sigma'_{\text{reg}}(V) = \sqrt{V + \sigma'^2_{\min}}$ has explicit derivative bounds...
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "sigma.*reg.*derivative\|derivative.*sigma.*reg" 01_fragile_gas_framework.md
grep -n "\\\\sigma'_{\\\\text{reg}}" 01_fragile_gas_framework.md | head -20
```

**Expected Location:** Section §8-9 (Regularized Standardization)

**Recommendation:** This technical lemma about derivatives of the regularized standard deviation should be formalized. Add a lemma directive where the derivative bounds are discussed.

---

### 3.6 lem-probabilistic-bound-perturbation-displacement-reproof (duplicate)

**File:** `sub-lem-probabilistic-bound-perturbation-displacement-reproof.json`
**Label:** `lem-probabilistic-bound-perturbation-displacement-reproof`
**Statement:** With high probability (under concentration inequalities), the total perturbation-induced displacement is bounded...
**Current Status:** Duplicate of 3.1

**Recommendation:** Consolidate with entry 3.1.

---

### 3.7 lem-potential-unstable-error-mean-square

**File:** `sub-lem-potential-unstable-error-mean-square.json`
**Label:** `lem-potential-unstable-error-mean-square`
**Statement:** The expected squared error component from walkers changing their survival status is bounded deterministically...
**Current Status:** No directive block

**Search Strategy:**
```bash
grep -n "sub-lem-potential-unstable-error" 01_fragile_gas_framework.md
grep -n "unstable.*error.*mean.*square" 01_fragile_gas_framework.md
```

**Expected Location:** Section §12 (Fitness Potential Operator), subsection on error decomposition

**Recommendation:** This is a sub-lemma in the potential operator continuity analysis. Add a lemma directive with `:label: lem-potential-unstable-error-mean-square`.

---

### 3.8 lem-perturbation-positional-bound-reproof

**File:** `sub-lem-perturbation-positional-bound-reproof.json`
**Label:** `lem-perturbation-positional-bound-reproof`
**Statement:** The squared positional displacement induced by the perturbation operator is bounded in expectation by a deterministic function...
**Current Status:** Mentioned in entry 1.8 (its proof)

**Search Strategy:**
```bash
grep -n "sub-lem-perturbation-positional-bound" 01_fragile_gas_framework.md
```

**Expected Location:** Section §14 (Perturbation Operator), line ~3979

**Recommendation:** The lemma is at line 3979 with label `:label: sub-lem-perturbation-positional-bound-reproof`. It already has a label! Re-run enricher to verify.

---

## 4. REMARKS (4 entities)

**General Note:** Remarks are typically inline admonitions or narrative text. To formalize them, wrap in `:::{prf:remark}` directives.

### 4.1 remark-extinction-risk-shift

**File:** `remark-extinction-risk-shift.json`
**Label:** `remark-extinction-risk-shift`
**Section:** §17 The Revival State
**Text:** The Revival State theorem reveals a profound shift in how extinction risk manifests: Before Revival State: Extinction was catastrophic...
**Current Status:** Inline text, no directive block

**Search Strategy:**
```bash
grep -n "extinction risk.*shift\|Revival State theorem reveals" 01_fragile_gas_framework.md
```

**Expected Location:** Section §17, narrative discussion

**Recommendation:** Search for this text in §17 and wrap it in:
```markdown
:::{prf:remark} Extinction Risk Shift
:label: remark-extinction-risk-shift

The Revival State theorem reveals a profound shift...
:::
```

---

### 4.2 remark-phoenix-effect

**File:** `remark-phoenix-effect.json`
**Label:** `remark-phoenix-effect`
**Section:** §17 The Revival State
**Text:** The revival state exhibits a remarkable 'phoenix effect': when disaster strikes and only one walker survives...
**Current Status:** Inline text, no directive block

**Search Strategy:**
```bash
grep -n "phoenix effect\|one walker survives" 01_fragile_gas_framework.md
```

**Expected Location:** Section §17, around line 5000+ (based on earlier grep results showing phoenix effect discussion)

**Recommendation:** This text appears around the revival state discussion. Wrap it in:
```markdown
:::{prf:remark} The Phoenix Effect
:label: remark-phoenix-effect

The revival state exhibits a remarkable 'phoenix effect'...
:::
```

---

### 4.3 remark-cloning-scope-companion-convention

**File:** `remark-cloning-scope-companion-convention.json`
**Label:** `remark-cloning-scope-companion-convention`
**Section:** §16 The Cloning Transition Measure
**Text:** All bounds in §15.2.4–§15.2.8 are stated for the regime $k_1=|\mathcal A(\mathcal S_1)|\ge 2$ (at least two alive walkers)...
**Current Status:** Inline text, no directive block

**Search Strategy:**
```bash
grep -n "companion convention\|regime.*k_1.*ge 2" 01_fragile_gas_framework.md
```

**Expected Location:** Section §16 (Cloning), narrative note about scope

**Recommendation:** Wrap this scope convention in:
```markdown
:::{prf:remark} Cloning Scope and Companion Convention
:label: remark-cloning-scope-companion-convention

All bounds in §15.2.4–§15.2.8 are stated for the regime...
:::
```

---

### 4.4 remark-cemetery-convention

**File:** `remark-cemetery-convention.json`
**Label:** `remark-cemetery-convention`
**Section:** §7
**Text:** This convention selects a maximal, state-independent distance to the cemetery law so that absorption is immediately detectable...
**Current Status:** Inline text, no directive block

**Search Strategy:**
```bash
grep -n "cemetery convention\|cemetery law" 01_fragile_gas_framework.md
```

**Expected Location:** Section §7 (likely discussing walker death/revival mechanics)

**Recommendation:** Wrap this convention in:
```markdown
:::{prf:remark} Cemetery Convention
:label: remark-cemetery-convention

This convention selects a maximal, state-independent distance...
:::
```

---

## Manual Verification Workflow

### Quick Search Script

Create a script to quickly locate all missing entities:

```bash
#!/bin/bash
# save as find_missing_entities.sh

MARKDOWN_FILE="docs/source/1_euclidean_gas/01_fragile_gas_framework.md"

echo "=== PROOFS ==="
grep -n "proof-thm-potential-operator-is-mean-square-continuous\|proof-lem-cloning-probability-lipschitz\|proof-thm-k1-revival-state" "$MARKDOWN_FILE"

echo ""
echo "=== PARAMETERS ==="
grep -n "param-n\|param-kappa-variance\|param-l-phi" "$MARKDOWN_FILE" | head -20

echo ""
echo "=== LEMMAS ==="
grep -n "lem-probabilistic-bound-perturbation-displacement\|lem-unify-holder-terms\|lem-inequality-toolbox" "$MARKDOWN_FILE"

echo ""
echo "=== REMARKS ==="
grep -n "extinction risk.*shift\|phoenix effect\|cemetery convention" "$MARKDOWN_FILE"
```

### Interactive Location Tool

Use the existing interactive tool:

```bash
cd /home/guillem/fragile
python src/tools/find_source_location.py find-text docs/source/1_euclidean_gas/01_fragile_gas_framework.md "search text" -d 01_fragile_gas_framework
```

---

## Priority Recommendations

### High Priority (Easy Wins)

1. **Proofs with existing theorem labels** - Add `:label:` to proof directives (1-2 lines per entity)
   - proof-lem-cloning-probability-lipschitz
   - proof-lem-total-clone-prob-value-error
   - proof-lem-total-clone-prob-structural-error

2. **Corollaries** - Already done! ✅

### Medium Priority (Moderate Effort)

3. **Sub-lemmas** - Add lemma directives for inline sub-lemmas
   - lem-bound-sum-total-cloning-probs
   - lem-potential-unstable-error-mean-square

4. **Remarks** - Wrap inline text in remark directives
   - remark-phoenix-effect
   - remark-cemetery-convention

### Low Priority (Requires Document Restructuring)

5. **Parameters** - Would require creating many new parameter definition blocks
   - Consider if cross-referencing is actually needed
   - Most parameters are adequately defined in axiom text

6. **Inline proofs** - These follow standard mathematical convention
   - Leave inline unless standalone reference is needed

---

## Validation After Adding Labels

After adding labels/directives, re-run the enricher:

```bash
python -m fragile.agents.text_location_enricher directory \
    docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/ \
    --source docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
    --document-id 01_fragile_gas_framework \
    --force
```

Then check coverage:

```bash
python3 -c "
import json
from pathlib import Path

base = Path('docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data')
total = 0
with_lines = 0

for entity_type in ['axioms', 'definitions', 'theorems', 'lemmas', 'proofs', 'parameters', 'propositions', 'corollaries', 'remarks']:
    for f in (base / entity_type).glob('*.json'):
        total += 1
        data = json.loads(f.read_text())
        if data.get('source', {}).get('line_range', {}).get('lines'):
            with_lines += 1

print(f'Coverage: {with_lines}/{total} ({100*with_lines/total:.1f}%)')
"
```

---

## Notes

- **Standard Practice:** Inline proofs after theorems are standard mathematical writing convention
- **Parameter Definitions:** Most parameters are adequately defined within axiom text
- **Document Restructuring:** Reaching 95%+ coverage would require significant restructuring that may reduce document readability
- **Current 81.4% Coverage:** Represents excellent traceability for the document structure

**Recommendation:** Focus on high-priority easy wins (proofs with existing theorem labels) to reach ~85% coverage without major restructuring.
