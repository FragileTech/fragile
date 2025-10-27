# Mathematical Theorem Dependency Graph - Summary Report

Generated: 2025-10-25

## Overview

Complete extraction and analysis of all mathematical theorems, lemmas, propositions, corollaries, definitions, axioms, and assumptions across the Fragile framework documentation.

**Total theorems analyzed: 855**

## Distribution by Chapter

### Chapter 1: Euclidean Gas
- **Documents**: 13
- **Theorems**: 501 (58.6%)
- **Average rigor**: 3.92/10
- **Average strategy**: 4.65/10
- **Key documents**:
  - 01_fragile_gas_framework.md: 122 theorems
  - 03_cloning.md: 108 theorems
  - 09_kl_convergence.md: 66 theorems

### Chapter 2: Geometric Gas
- **Documents**: 10
- **Theorems**: 270 (31.6%)
- **Average rigor**: 3.32/10
- **Average strategy**: 3.95/10
- **Key documents**:
  - 11_geometric_gas.md: 58 theorems
  - 20_geometric_gas_cinf_regularity_full.md: 47 theorems
  - 18_emergent_geometry.md: 34 theorems

### Chapter 3: Brascamp-Lieb
- **Documents**: 4
- **Theorems**: 84 (9.8%)
- **Average rigor**: 4.88/10
- **Average strategy**: 5.49/10
- **Key documents**:
  - eigenvalue_gap_complete_proof.md: 40 theorems
  - brascamp_lieb_proof.md: 27 theorems

## Quality Assessment

### By Recommendation
- **Accept**: 49 (5.7%) - Ready for publication
- **Minor revision**: 11 (1.3%) - Nearly ready, small fixes needed
- **Major revision**: 2 (0.2%) - Significant work required
- **Reject**: 793 (92.8%) - Needs proof or substantial development

**Overall completion rate: 7.02%** (Accept + Minor revision)

### By Proof Status
- **Proven**: 32 (3.7%) - Marked as proven with status markers
- **Has complete proof**: 32 (3.7%) - Contains full proof block
- **Needs proof**: 791 (92.5%) - No proof or only statement

### Quality Metrics
- **Average rigor score**: 3.83/10
- **Average strategy score**: 4.51/10

## Dependency Analysis

### Circular Dependencies
- **Total circular dependencies**: 65
- **Severity**: All marked as critical
- **Pattern**: Many are self-referential (theorem depending on itself)
- **Action needed**: Review and resolve cycles manually

### Missing Dependencies
- **Total missing references**: 53
- **Impact**: All marked as critical
- **Common missing labels**:
  - `lem-sigma-patch-lipschitz`
  - `def-sasaki-structural-coeffs-sq`
  - `thm-sasaki-standardization-value`

## Key Insights

### 1. Proof Gap Analysis
The framework has a **92.8% proof gap** - the vast majority of mathematical statements lack complete proofs. This represents a significant opportunity for formalization work.

### 2. Chapter Quality Variance
- **Brascamp-Lieb** (Chapter 3) has the highest quality metrics (4.88 rigor, 5.49 strategy)
- **Geometric Gas** (Chapter 2) has the lowest quality metrics (3.32 rigor, 3.95 strategy)
- This suggests prioritizing Chapter 2 for proof development

### 3. Foundational Work Needed
With only **64 theorems** having complete proofs or proven status (7.5%), there's substantial foundational work required to bring the framework to publication-ready status.

### 4. Document Priorities
Documents with highest theorem counts but low completion rates should be prioritized:
- `01_fragile_gas_framework.md`: 122 theorems (foundational)
- `03_cloning.md`: 108 theorems (core mechanism)
- `09_kl_convergence.md`: 66 theorems (key results)

## Recommendations

### Immediate Actions
1. **Resolve circular dependencies** - Review 65 cases of circular logic
2. **Add missing definitions** - 53 referenced labels need to be defined
3. **Fix self-referential cycles** - Many theorems reference themselves

### Short-term Priorities
1. **Focus on Chapter 3** - Highest quality, easiest to complete
2. **Prove foundational results** - Framework axioms and core definitions
3. **Document proof strategies** - Even sketches would improve strategy scores

### Long-term Strategy
1. **Systematic proof campaign** - Work through dependency chains
2. **Automated proof pipeline** - Use the dependency graph for scheduling
3. **Cross-chapter consistency** - Ensure geometric gas builds properly on euclidean gas

## Usage

The complete dependency graph is available in `docs/math_graph.json` with:
- Full theorem metadata (label, type, title, location)
- Dependency and reverse dependency graphs
- Quality metrics (rigor, strategy, recommendation)
- Circular and missing dependency lists
- Comprehensive statistics

### Example Queries

**Find high-priority theorems to prove:**
```python
import json
with open('docs/math_graph.json') as f:
    data = json.load(f)

# Theorems with many dependents (high impact)
reverse_deps = data['reverse_dependencies']
high_impact = sorted(
    reverse_deps.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:10]
```

**Find theorems ready for minor revision:**
```python
minor_rev = [t for t in data['theorems'] if t['recommendation'] == 'Minor revision']
```

**Analyze proof gaps by document:**
```python
from collections import defaultdict
by_doc = defaultdict(lambda: {'total': 0, 'proven': 0})

for thm in data['theorems']:
    doc = thm['document']
    by_doc[doc]['total'] += 1
    if thm['status'] in ['proven', 'has_complete_proof']:
        by_doc[doc]['proven'] += 1

for doc, stats in sorted(by_doc.items()):
    completion = 100 * stats['proven'] / stats['total']
    print(f"{doc}: {completion:.1f}% ({stats['proven']}/{stats['total']})")
```

## Next Steps

1. ✅ **Graph created** - Complete theorem dependency graph extracted
2. ⏭️ **Resolve inconsistencies** - Fix circular and missing dependencies
3. ⏭️ **Prioritize proofs** - Use graph to schedule proof development
4. ⏭️ **Automated pipeline** - Integrate with math_pipeline for systematic proving

---

**Data file**: `docs/math_graph.json`  
**Generated by**: `build_math_graph.py`  
**Last updated**: 2025-10-25
