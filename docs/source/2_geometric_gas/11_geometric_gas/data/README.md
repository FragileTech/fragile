# Deep Dependency Analysis - Quick Reference

**Document:** `docs/source/2_geometric_gas/11_geometric_gas.md`
**Title:** The Geometric Viscous Fluid Model
**Analysis Date:** 2025-10-26
**Mode:** ULTRATHINK Deep Dependency Extraction

---

## Files in This Directory

### 1. `DEEP_DEPENDENCY_REPORT.md` (PRIMARY REPORT)
**Size:** 16 KB
**Content:**
- Executive summary of the document
- Complete 13-step critical path to main convergence theorem
- Cross-document dependencies (Chapter 1: Euclidean Gas)
- Main theorems with detailed proof dependencies
- Novel contributions vs. Euclidean Gas baseline
- Gaps, conjectures, and verification requirements
- Reading guide and navigation tips

**Use this for:** Understanding the mathematical structure and proof strategy

---

### 2. `DEPENDENCY_GRAPH.md` (VISUAL DIAGRAMS)
**Size:** 12 KB
**Content:**
- Mermaid diagrams showing:
  - Critical path graph (foundation → main theorem)
  - LSI and mean-field theory dependencies
  - Keystone Principle extension (Appendix B)
  - Full internal reference graph
  - Cross-document dependency map
- Graph statistics and navigation guide

**Use this for:** Visual understanding of how theorems connect

---

### 3. `deep_dependency_analysis.json` (RAW DATA)
**Size:** 168 KB
**Content:**
- All 61 extracted MyST directives
- Complete directive content and metadata
- 456 dependency edges (41 explicit + 415 implicit)
- Line numbers, labels, math expressions
- Full extraction inventory

**Use this for:** Programmatic access to complete extraction data

---

### 4. `deep_dependency_analysis_enhanced.json` (STRUCTURED)
**Size:** 9.9 KB
**Content:**
- Cross-document dependency mapping
- 13-step critical path (structured)
- 5 main theorems with proof I/O analysis
- Key foundations from Euclidean Gas
- Novel contributions list
- Statistics summary

**Use this for:** High-level structured access to key results

---

### 5. `dependency_graph.json` (GRAPH DATA)
**Size:** 21 KB
**Content:**
- Node-edge graph format
- All 61 nodes (directives)
- All dependency edges
- Ready for graph visualization tools

**Use this for:** Import into graph analysis tools (NetworkX, Gephi, etc.)

---

### 6. `summary_report.txt` (QUICK STATS)
**Size:** 1.1 KB
**Content:**
- Directive counts by type
- Dependency type breakdown
- Standard math prerequisites
- Critical path count

**Use this for:** Quick statistics overview

---

## Quick Access Commands

### View Main Report
```bash
cat DEEP_DEPENDENCY_REPORT.md | less
```

### View Dependency Graphs
```bash
cat DEPENDENCY_GRAPH.md | less
```

### Load JSON (Python)
```python
import json

# Load enhanced analysis
with open('deep_dependency_analysis_enhanced.json') as f:
    data = json.load(f)

# Access critical path
print(data['critical_path_to_main_theorem'])

# Access main theorems
print(data['main_theorems'])

# Access cross-document dependencies
print(data['explicit_dependencies']['cross_document'])
```

### Graph Analysis (NetworkX)
```python
import json
import networkx as nx

# Load graph
with open('dependency_graph.json') as f:
    graph_data = json.load(f)

# Build NetworkX graph
G = nx.DiGraph()
for node in graph_data['nodes']:
    G.add_node(node['id'], type=node['type'], title=node['title'])
for edge in graph_data['edges']:
    G.add_edge(edge['source'], edge['target'], type=edge['type'])

# Analyze
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Most central: {nx.betweenness_centrality(G)}")
```

---

## Key Results Summary

### Main Convergence Theorem
**Label:** `thm-geometric-ergodicity` (Theorem 9.1)
**Statement:** For ε_F < ε_F*(ρ), exponential convergence to unique QSD with rate λ = 1 - κ_total(ρ)

**Critical Dependencies:**
1. `thm-foster-lyapunov` - Drift condition κ_total(ρ) > 0
2. `thm-ueph` - Uniform ellipticity by construction
3. `backbone-convergence` - Euclidean Gas stability (from 04_convergence.md)
4. `discretization-theorem` - Continuous/discrete time bridge (from 04_convergence.md)

### Core Innovation
**Label:** `thm-ueph` (Theorem 4.1)
**Statement:** c_min(ρ)I ⪯ G_reg ⪯ c_max(ρ)I for all N, all ρ > 0

**Why Critical:** Transforms probabilistic UEPH verification into algebraic eigenvalue bounds. Without this, the entire adaptive diffusion analysis would fail.

### Critical Threshold
**Formula:** ε_F*(ρ) = (κ_backbone - C_diff,1(ρ)) / (2 K_F(ρ))

**Interpretation:** Maximum safe adaptation rate. Above this threshold, adaptive perturbations overpower backbone stability and convergence fails.

---

## Cross-Document Dependencies

### From Chapter 1 (Euclidean Gas):

**03_cloning.md:**
- Theorem 8.1: Keystone Principle (variance contraction)
- Def. 5.6.1: Cloning operator structure
- Chapter 4: Foundational cloning axioms

**04_convergence.md / 06_convergence.md:**
- Axiom 1.3.1: Globally confining potential U(x)
- Theorem 1.4.2: Backbone Foster-Lyapunov convergence
- Theorem 1.7.2: Discretization theorem
- Theorem 1.4.3: Petite set property

**05_kinetic_contraction.md:**
- Hypocoercive Wasserstein contraction
- Velocity dissipation mechanism

**07_mean_field.md + 08_propagation_chaos.md:**
- Propagation of chaos
- Mean-field limit theory

---

## Novel Contributions (Not in Euclidean Gas)

1. **ρ-Parameterized Measurement Pipeline**
   - Unifies global (ρ→∞) and local (finite ρ) regimes
   - Smooth interpolation between extremes

2. **Regularized Hessian Diffusion**
   - Σ_reg = (H + ε_Σ I)^{-1/2}
   - Information-geometric adaptation to fitness landscape

3. **Uniform Ellipticity by Construction**
   - Algebraic proof via eigenvalue bounds
   - Avoids probabilistic UEPH arguments

4. **Perturbation Analysis Framework**
   - Adaptive mechanisms as bounded perturbations
   - Stable backbone + adaptive intelligence

5. **Critical Stability Threshold**
   - Explicit formula ε_F*(ρ)
   - ρ-dependent exploration/exploitation tradeoff

6. **C¹/C² Regularity Theory**
   - N-uniform gradient and Hessian bounds
   - Enables all perturbation analysis

---

## Gaps and Open Questions

### Conjectures:
- **Conjecture 9.2:** WFR convergence (formal analogy exists, rigorous proof missing)

### Partial Results:
- **Mean-Field LSI:** Requires propagation of chaos extension to Σ_reg
- **Discretization Theorem:** Verification sketched but not fully detailed
- **LSI Jump Operator:** Two sufficient conditions stated but not verified

### Verification Needed:
- Growth bounds on ∇V_total (stated, proof not shown)
- Lipschitz continuity of adaptive drift (stated, proof not shown)
- Cloning operator LSI (plausible, verification required)

---

## Statistics

| Metric | Count |
|--------|-------|
| Total Directives | 61 |
| Definitions | 12 |
| Theorems | 12 |
| Lemmas | 19 |
| Propositions | 5 |
| Axioms | 4 |
| Corollaries | 6 |
| Explicit Dependencies | 41 |
| Implicit Dependencies | 415 |
| Cross-Document Sources | 5 |
| Critical Path Length | 13 steps |
| Main Theorems | 5 |

---

## Next Steps

1. **Proof Verification:** Fill gaps in conjectures (WFR convergence, discretization details)
2. **Graph Analysis:** Import into NetworkX/Gephi for centrality metrics
3. **Cross-Document Integration:** Parse all Chapter 1 docs, build unified graph
4. **Formalization:** Export critical path to Lean4 or Coq
5. **Documentation:** Generate auto-docs, build interactive theorem browser

---

## Contact

For questions about this analysis, consult:
- Main document: `docs/source/2_geometric_gas/11_geometric_gas.md`
- Framework glossary: `docs/glossary.md`
- Project documentation: `CLAUDE.md`

**Analysis generated by:** Document Parser Agent (ULTRATHINK mode)
**Framework:** Fragile Gas (FractalAI research project)
