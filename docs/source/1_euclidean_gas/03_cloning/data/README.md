# Cloning Theory - Dependency Analysis Output

**Generated**: 2025-10-26  
**Source**: `docs/source/1_euclidean_gas/03_cloning.md` (470.5 KB)  
**Extraction Tool**: Deep Dependency Extractor v1.0

---

## Directory Contents

### Analysis Reports (Markdown)

1. **`DEPENDENCY_ANALYSIS_SUMMARY.md`** (11 KB)
   - Executive summary of extraction results
   - Directive breakdown by type
   - Dependency structure analysis
   - Most referenced definitions (top 5)
   - Keystone critical path explanation
   - Cross-document dependencies
   - Key mathematical concepts overview
   - Issues and limitations identified
   - Recommendations for framework integration
   - **Use for**: Quick overview, documentation, onboarding

2. **`KEYSTONE_CRITICAL_PATH.md`** (20 KB)
   - Detailed node-by-node analysis of 8-step critical path
   - Mermaid diagram of dependency chain
   - Mathematical statements for each node
   - N-uniformity analysis
   - Proof strategy summary
   - Bottleneck analysis
   - Validation checklist
   - **Use for**: Understanding Keystone Principle structure

3. **`README.md`** (this file)
   - Index of all output files
   - Quick reference guide
   - **Use for**: Navigation

### Data Files (JSON)

4. **`dependency_graph.json`** (40 KB, 1580 lines)
   - Complete graph structure
   - **Nodes** (116 total):
     - `id`: Label (e.g., "def-variance-conversions")
     - `type`: Directive type (definition, theorem, lemma, etc.)
     - `title`: Human-readable title
     - `line_range`: [start, end] in source document
   - **Edges** (67 total):
     - `source`: Source node label
     - `target`: Target node label
     - `type`: explicit | notation | axiom | proved_by | proves
     - `strength`: strong | medium | weak
     - `context`: Surrounding text (first 200 chars)
   - **Statistics**:
     - `total_nodes`: 116
     - `total_edges`: 67
     - `max_in_degree`: 20 (def-variance-conversions)
     - `max_out_degree`: TBD
     - `avg_in_degree`: 0.58
     - `avg_out_degree`: 0.58
   - **Most Referenced** (top 10 nodes with in-degree counts)
   - **Use for**: Graph visualization, path finding, NetworkX analysis

5. **`deep_dependency_analysis.json`** (7.6 KB, 244 lines)
   - **Directive breakdown**: Count by type (definition, theorem, lemma, etc.)
   - **Dependency breakdown**: Count by type (explicit, notation, axiom)
   - **Keystone critical path**: List of 8 node labels
   - **Cross-document dependencies**: 24 refs (14 framework, 10 drift)
   - **Isolated directives**: 67 foundational definitions (no outgoing edges)
   - **Missing references**: 67 targets (many in unlabeled proofs)
   - **Use for**: Summary statistics, gap identification, validation

6. **`directive_catalog.json`** (34 KB, 1517 lines)
   - Complete catalog of all 116 directives
   - For each directive:
     - `type`: definition | theorem | lemma | proposition | corollary | axiom | remark
     - `title`: Human-readable title
     - `line_range`: [start, end] in source
     - `explicit_refs`: List of {prf:ref} targets
     - `implicit_refs`: Heuristically detected refs
     - `notation_used`: LaTeX notation patterns
     - `axioms_invoked`: Framework axioms (EG-0 through EG-5)
   - **Use for**: Quick lookup, reference checking, notation tracking

### Legacy/Auxiliary Files

7. **`extraction_inventory.json`** (71 KB)
   - Legacy format from earlier extraction run
   - Contains raw directive content (not just metadata)
   - **Use for**: Fallback if detailed content needed

8. **`complete_dependency_graph.json`** (5.9 KB)
   - Earlier version of dependency graph
   - **Use for**: Comparison/validation

9. **`comprehensive_extraction_report.json`** (7.9 KB)
   - Earlier version of analysis report
   - **Use for**: Comparison/validation

10. **`object_dependencies.json`** (24 KB)
    - Specialized extraction of mathematical objects
    - **Use for**: Object-oriented analysis

11. **`pydantic_objects.json`** (15 KB)
    - Pydantic schema validation results
    - **Use for**: Schema compliance checking

12. **`math_object_schema.json`** (7.5 KB)
    - Schema definitions for mathematical objects
    - **Use for**: Validation, schema design

13. **`statistics.json`** (160 bytes)
    - High-level statistics
    - **Use for**: Quick metrics

### Subdirectories

14. **`proof_strategies/`**
    - Proof strategy analysis (from earlier extraction)
    - **Use for**: Proof methodology research

---

## Quick Reference

### Find a Definition

```bash
jq '.[] | select(.type == "definition") | {label, title}' directive_catalog.json
```

### Find All References to a Node

```bash
jq '.edges[] | select(.target == "def-variance-conversions")' dependency_graph.json
```

### Get Keystone Critical Path

```bash
jq '.keystone_critical_path' deep_dependency_analysis.json
```

### Count Directives by Type

```bash
jq '.directive_breakdown' deep_dependency_analysis.json
```

### Find Most Referenced Nodes

```bash
jq '.most_referenced' dependency_graph.json
```

### Get Cross-Document Dependencies

```bash
jq '.cross_document_dependencies' deep_dependency_analysis.json
```

---

## Key Statistics

- **Total Directives**: 116
  - Definitions: 36
  - Theorems: 15
  - Lemmas: 29
  - Propositions: 11
  - Corollaries: 6
  - Axioms: 6
  - Remarks: 13

- **Total Dependencies**: 67
  - Explicit (prf:ref): 2
  - Notation: 57
  - Axioms: 8

- **Critical Path Length**: 8 nodes

- **Cross-Document Refs**: 24
  - To Framework (Ch 01): 14
  - To Drift Analysis (Ch 9-12): 10

- **Most Referenced**:
  1. def-variance-conversions (20 refs)
  2. def-cloning-operator-formal (11 refs)
  3. def-structural-error-component (9 refs)
  4. def-algorithmic-distance-metric (9 refs)
  5. def-location-error-component (5 refs)

---

## Recommended Workflow

### For Understanding Document Structure

1. Read `DEPENDENCY_ANALYSIS_SUMMARY.md` for overview
2. Review `KEYSTONE_CRITICAL_PATH.md` for key results
3. Explore `dependency_graph.json` for detailed relationships

### For Finding Specific Information

1. Use `directive_catalog.json` for quick label lookup
2. Use `dependency_graph.json` for dependency queries
3. Use `deep_dependency_analysis.json` for statistics

### For Visualization

1. Load `dependency_graph.json` into NetworkX:
   ```python
   import json
   import networkx as nx
   
   with open('dependency_graph.json') as f:
       data = json.load(f)
   
   G = nx.DiGraph()
   for node in data['nodes']:
       G.add_node(node['id'], **node)
   for edge in data['edges']:
       G.add_edge(edge['source'], edge['target'], **edge)
   ```

2. Visualize with HoloViews/Bokeh (preferred) or matplotlib

### For Integration with Other Tools

- **Lean4 Export**: Use directive catalog to generate formal definitions
- **Graph Analysis**: Load dependency graph into NetworkX for path finding
- **Documentation**: Reference markdown summaries in papers/presentations
- **Automated Validation**: Use statistics.json for CI/CD checks

---

## Known Limitations

1. **Missing Proof Labels**: Many proofs don't have explicit labels
   - Impact: Only 2 explicit refs captured (8 exist in document)
   - Solution: Add labels to all proof blocks, re-run extraction

2. **Limited Implicit Dependencies**: Heuristic found 0 relationships
   - Impact: Underestimate of true dependency complexity
   - Solution: Use LLM (Gemini 2.5 Pro) for implicit dependency inference

3. **Isolated Directives**: 67 nodes have no outgoing edges
   - Status: Not a problem - these are foundational definitions
   - Explanation: Roots of dependency DAG, high in-degree expected

---

## Next Steps

1. **Label all proof blocks** in source document
2. **Re-run extraction** to capture complete graph
3. **Generate visualization** using NetworkX + HoloViews
4. **LLM-based implicit deps** using Gemini 2.5 Pro
5. **Cross-validate** with Chapters 9-12 for consistency

---

## File Format Specifications

### `dependency_graph.json`

```json
{
  "nodes": [
    {
      "id": "label-of-directive",
      "type": "definition | theorem | lemma | ...",
      "title": "Human-readable title",
      "line_range": [start_line, end_line]
    }
  ],
  "edges": [
    {
      "source": "source-label",
      "target": "target-label",
      "type": "explicit | notation | axiom | ...",
      "strength": "strong | medium | weak",
      "context": "Surrounding text snippet"
    }
  ],
  "metadata": {
    "total_nodes": 116,
    "total_edges": 67
  },
  "statistics": {
    "max_in_degree": 20,
    "max_out_degree": ...,
    "avg_in_degree": 0.58,
    "avg_out_degree": 0.58
  },
  "most_referenced": [
    {
      "label": "def-variance-conversions",
      "references": 20,
      "title": "Variance Notation Conversion Formulas"
    }
  ]
}
```

### `directive_catalog.json`

```json
{
  "label-of-directive": {
    "type": "definition",
    "title": "Human-readable title",
    "line_range": [start_line, end_line],
    "explicit_refs": ["ref-1", "ref-2"],
    "implicit_refs": ["ref-3"],
    "notation_used": ["V_{\\text{loc}}", "\\mathcal{W}_h"],
    "axioms_invoked": ["axiom-domain-regularity"]
  }
}
```

---

## Contact & Support

For issues or questions about the extraction:
- Check `DEPENDENCY_ANALYSIS_SUMMARY.md` for known issues
- Review `KEYSTONE_CRITICAL_PATH.md` for theoretical background
- Consult source document: `docs/source/1_euclidean_gas/03_cloning.md`

---

**Last Updated**: 2025-10-26  
**Extraction Tool**: `deep_dependency_extractor.py` (v1.0)  
**Total Extraction Time**: ~5 seconds  
**Document Size**: 470.5 KB, 8467 lines
