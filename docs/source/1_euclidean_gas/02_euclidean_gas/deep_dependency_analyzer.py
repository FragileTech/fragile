#!/usr/bin/env python3
"""
COMPREHENSIVE DEEP DEPENDENCY EXTRACTION FOR 02_euclidean_gas.md

This script performs ultrathink-level analysis of mathematical dependencies,
tracking both explicit cross-references and implicit assumptions/uses.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

@dataclass
class Dependency:
    """Represents a single dependency relationship"""
    target_label: str
    reference_type: str  # "prf:ref", "implicit-definition", "unstated-axiom", "standard-math"
    context: str
    line_in_proof: Optional[int] = None
    evidence: Optional[str] = None
    critical: bool = True

@dataclass
class DirectiveInfo:
    """Complete information about a MyST directive"""
    label: str
    type: str  # definition, theorem, lemma, etc.
    title: str
    content: str
    line_range: Tuple[int, int]
    math_expression_count: int
    first_math: str

    # Dependency tracking
    explicit_deps: List[Dependency] = field(default_factory=list)
    implicit_deps: List[Dependency] = field(default_factory=list)
    assumptions: List[Dependency] = field(default_factory=list)
    standard_math: List[Dependency] = field(default_factory=list)
    missing_references: List[Dict] = field(default_factory=list)

    def dependency_summary(self) -> Dict:
        total = len(self.explicit_deps) + len(self.implicit_deps) + len(self.assumptions) + len(self.standard_math)
        critical = sum(1 for d in self.explicit_deps + self.implicit_deps + self.assumptions if d.critical)
        external = sum(1 for d in self.explicit_deps + self.implicit_deps + self.assumptions
                      if d.target_label.startswith('def-') or d.target_label.startswith('axiom-'))

        return {
            "total_dependencies": total,
            "critical_dependencies": critical,
            "external_framework_dependencies": external,
            "missing_references": len(self.missing_references)
        }


class DeepDependencyAnalyzer:
    """Performs comprehensive dependency extraction from mathematical documents"""

    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        self.lines = doc_path.read_text().splitlines()
        self.directives: Dict[str, DirectiveInfo] = {}
        self.dependency_graph = {"nodes": [], "edges": []}

    def extract_all_directives(self) -> Dict:
        """Phase 1: Extract all MyST directives"""
        print("Phase 1: Extracting MyST directives...")

        current_directive = None
        directive_start = None
        in_directive = False
        in_proof = False

        for i, line in enumerate(self.lines, 1):
            # Detect directive start
            if match := re.match(r'^:::+\{prf:(definition|theorem|lemma|proposition|corollary|axiom|proof|algorithm)\}', line):
                directive_type = match.group(1)
                directive_start = i
                current_directive = {"type": directive_type, "lines": []}
                in_directive = True
                in_proof = (directive_type == "proof")

            # Collect directive content
            elif in_directive:
                current_directive["lines"].append((i, line))

                # Extract label
                if match := re.match(r'^:label:\s+(\S+)', line):
                    current_directive["label"] = match.group(1)

                # Detect directive end
                if re.match(r'^:::+\s*$', line):
                    # Process completed directive
                    if "label" in current_directive or in_proof:
                        self._process_directive(current_directive, directive_start, i)
                    in_directive = False
                    current_directive = None

        print(f"  Found {len(self.directives)} labeled directives")
        return {"total_directives": len(self.directives)}

    def _process_directive(self, directive_dict: Dict, start_line: int, end_line: int):
        """Process a single directive and extract metadata"""
        lines_content = [line for _, line in directive_dict["lines"]]
        content = "\n".join(lines_content)

        # Extract title (first non-empty, non-metadata line)
        title = ""
        for line in lines_content:
            if line.strip() and not line.startswith(":"):
                title = line.strip()
                break

        # Count math expressions
        math_count = len(re.findall(r'\$\$|\$', content))

        # Extract first math expression
        first_math = ""
        if match := re.search(r'\$\$([^$]+)\$\$|\$([^$]+)\$', content):
            first_math = (match.group(1) or match.group(2) or "").strip()[:100]

        label = directive_dict.get("label", f"unlabeled-{start_line}")

        info = DirectiveInfo(
            label=label,
            type=directive_dict["type"],
            title=title,
            content=content,
            line_range=(start_line, end_line),
            math_expression_count=math_count,
            first_math=first_math
        )

        self.directives[label] = info

    def analyze_dependencies(self):
        """Phase 2-4: Analyze all dependencies"""
        print("Phase 2-4: Analyzing dependencies...")

        for label, directive in self.directives.items():
            if directive.type == "proof":
                continue  # Proofs are analyzed with their parent theorem

            # Extract explicit cross-references
            self._extract_explicit_deps(directive)

            # Analyze proof if it exists
            self._analyze_proof_dependencies(directive)

            # Infer implicit dependencies
            self._infer_implicit_deps(directive)

        print(f"  Analyzed {len(self.directives)} directives")

    def _extract_explicit_deps(self, directive: DirectiveInfo):
        """Extract all {prf:ref}`label` cross-references"""
        content = directive.content

        for match in re.finditer(r'\{prf:ref\}`([^`]+)`', content):
            target = match.group(1)
            context_start = max(0, match.start() - 100)
            context_end = min(len(content), match.end() + 100)
            context = content[context_start:context_end]

            directive.explicit_deps.append(Dependency(
                target_label=target,
                reference_type="prf:ref",
                context=context.strip(),
                critical=True
            ))

    def _analyze_proof_dependencies(self, directive: DirectiveInfo):
        """Analyze proof blocks for dependencies"""
        # Find associated proof by looking ahead
        proof_start = directive.line_range[1]
        proof_content_lines = []
        in_proof = False

        for i in range(proof_start, min(proof_start + 500, len(self.lines))):
            line = self.lines[i]

            if re.match(r'^:::+\{prf:proof\}', line):
                in_proof = True
                continue

            if in_proof:
                if re.match(r'^:::+\s*$', line):
                    break
                proof_content_lines.append((i+1, line))

        if not proof_content_lines:
            return

        proof_content = "\n".join(line for _, line in proof_content_lines)

        # Analyze proof step by step
        self._extract_proof_steps(directive, proof_content, proof_content_lines)

    def _extract_proof_steps(self, directive: DirectiveInfo, proof_content: str, proof_lines: List[Tuple[int, str]]):
        """Extract logical flow and dependencies from proof"""

        # Pattern 1: "By [result]" or "Using [result]"
        for match in re.finditer(r'(?:By|Using|From|Since|Because|Via)\s+(?:the\s+)?([A-Z][a-z]+)\s+(\{prf:ref\}`[^`]+`)', proof_content, re.IGNORECASE):
            ref_match = re.search(r'\{prf:ref\}`([^`]+)`', match.group(0))
            if ref_match:
                target = ref_match.group(1)
                directive.explicit_deps.append(Dependency(
                    target_label=target,
                    reference_type="prf:ref",
                    context=f"Used in proof: {match.group(0)[:200]}",
                    critical=True
                ))

        # Pattern 2: Unstated assumptions
        assumption_patterns = [
            (r'by\s+assumption', "assumption"),
            (r'by\s+compactness', "compactness-assumption"),
            (r'by\s+continuity', "continuity-assumption"),
            (r'since.*bounded', "boundedness-assumption"),
        ]

        for pattern, assumption_type in assumption_patterns:
            for match in re.finditer(pattern, proof_content, re.IGNORECASE):
                context = proof_content[max(0, match.start()-100):min(len(proof_content), match.end()+100)]
                directive.assumptions.append(Dependency(
                    target_label=f"unstated-{assumption_type}",
                    reference_type="unstated-assumption",
                    context=context.strip(),
                    evidence=f"Proof mentions: {match.group(0)}",
                    critical=True
                ))

        # Pattern 3: Standard mathematical results
        standard_math_patterns = [
            "triangle inequality",
            "Cauchy-Schwarz",
            "Jensen's inequality",
            "Hölder's inequality",
            "Markov's inequality",
            "Chebyshev's inequality",
            "dominated convergence",
            "Fubini",
            "mean value theorem",
        ]

        for std_math in standard_math_patterns:
            if re.search(rf'\b{std_math}\b', proof_content, re.IGNORECASE):
                directive.standard_math.append(Dependency(
                    target_label="standard-math",
                    reference_type="standard-math",
                    context=std_math,
                    critical=False
                ))

    def _infer_implicit_deps(self, directive: DirectiveInfo):
        """Infer implicit dependencies from notation and terminology"""
        content = directive.content

        # Pattern: Uses of Sasaki metric without explicit reference
        if "d_{\\mathcal Y}^{\\mathrm{Sasaki}}" in content or "Sasaki metric" in content:
            # Check if definition is explicitly referenced
            has_explicit_ref = any(d.target_label == "def-sasaki-metric" for d in directive.explicit_deps)
            if not has_explicit_ref:
                directive.implicit_deps.append(Dependency(
                    target_label="def-sasaki-metric",
                    reference_type="implicit-definition",
                    context="Uses Sasaki metric notation",
                    evidence="Found d_Y^Sasaki notation without explicit ref",
                    critical=True
                ))

        # Pattern: Uses of squashing maps
        if "\\psi_x" in content or "\\psi_v" in content:
            has_explicit_ref = any("squashing" in d.target_label for d in directive.explicit_deps)
            if not has_explicit_ref:
                directive.implicit_deps.append(Dependency(
                    target_label="lem-squashing-properties-generic",
                    reference_type="implicit-definition",
                    context="Uses squashing map notation",
                    evidence="Found \\psi_x or \\psi_v without explicit ref",
                    critical=True
                ))

    def build_dependency_graph(self):
        """Phase 5: Build complete dependency graph"""
        print("Phase 5: Building dependency graph...")

        # Create nodes
        for label, directive in self.directives.items():
            self.dependency_graph["nodes"].append({
                "id": label,
                "type": directive.type,
                "title": directive.title[:100]
            })

        # Create edges
        for label, directive in self.directives.items():
            all_deps = directive.explicit_deps + directive.implicit_deps + directive.assumptions

            for dep in all_deps:
                if dep.target_label in self.directives or dep.target_label.startswith("def-") or dep.target_label.startswith("axiom-"):
                    self.dependency_graph["edges"].append({
                        "source": label,
                        "target": dep.target_label,
                        "edge_type": "requires" if dep.critical else "uses",
                        "critical": dep.critical,
                        "reference_type": dep.reference_type
                    })

        print(f"  Created {len(self.dependency_graph['nodes'])} nodes and {len(self.dependency_graph['edges'])} edges")

    def generate_outputs(self, output_dir: Path):
        """Generate all output files"""
        print("Generating output files...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main analysis file
        analysis = {
            "document": str(self.doc_path),
            "extraction_mode": "deep-dependency-analysis",
            "directives": [
                {
                    "label": d.label,
                    "type": d.type,
                    "title": d.title,
                    "line_range": d.line_range,
                    "dependencies": {
                        "explicit": [asdict(dep) for dep in d.explicit_deps],
                        "implicit": [asdict(dep) for dep in d.implicit_deps],
                        "assumptions": [asdict(dep) for dep in d.assumptions],
                        "standard_math": [asdict(dep) for dep in d.standard_math],
                        "missing_references": d.missing_references
                    },
                    "dependency_summary": d.dependency_summary()
                }
                for d in self.directives.values()
            ],
            "dependency_graph": self.dependency_graph,
            "analysis": self._compute_analysis_stats()
        }

        (output_dir / "deep_dependency_analysis.json").write_text(json.dumps(analysis, indent=2))

        # Separate graph file
        (output_dir / "dependency_graph.json").write_text(json.dumps(self.dependency_graph, indent=2))

        # Human-readable reports
        self._generate_missing_references_report(output_dir)
        self._generate_critical_path_analysis(output_dir)

        print(f"✓ Generated outputs in {output_dir}")

    def _compute_analysis_stats(self) -> Dict:
        """Compute summary statistics"""
        all_targets = set()
        dependency_counts = defaultdict(int)

        for directive in self.directives.values():
            for dep in directive.explicit_deps + directive.implicit_deps + directive.assumptions:
                all_targets.add(dep.target_label)
                dependency_counts[dep.target_label] += 1

        # Find most depended-on results
        most_depended = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Find leaf results (not depended on by anything)
        all_sources = set(self.directives.keys())
        leaf_results = [label for label in all_sources if dependency_counts[label] == 0]

        return {
            "most_depended_on": [{"label": label, "count": count} for label, count in most_depended],
            "leaf_results": leaf_results[:20],
            "total_unique_dependencies": len(all_targets),
            "validation_issues": []
        }

    def _generate_missing_references_report(self, output_dir: Path):
        """Generate human-readable report of missing references"""
        with open(output_dir / "missing_references_report.txt", "w") as f:
            f.write("MISSING REFERENCES REPORT\n")
            f.write("=" * 80 + "\n\n")

            for directive in self.directives.values():
                if directive.missing_references:
                    f.write(f"{directive.label} ({directive.type})\n")
                    f.write(f"  Line {directive.line_range[0]}-{directive.line_range[1]}\n")
                    for issue in directive.missing_references:
                        f.write(f"  - {issue['description']} (line {issue['line']})\n")
                    f.write("\n")

    def _generate_critical_path_analysis(self, output_dir: Path):
        """Generate critical dependency paths"""
        with open(output_dir / "critical_path_analysis.txt", "w") as f:
            f.write("CRITICAL PATH ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Find main theorems (likely endpoints)
            main_theorems = [d for d in self.directives.values() if d.type == "theorem"]

            for thm in main_theorems[:5]:  # Top 5 theorems
                f.write(f"\nDependency chain for: {thm.label}\n")
                f.write(f"  {thm.title}\n")
                f.write("-" * 80 + "\n")

                # List direct critical dependencies
                critical_deps = [d for d in thm.explicit_deps + thm.implicit_deps if d.critical]
                for dep in critical_deps:
                    f.write(f"  → {dep.target_label} ({dep.reference_type})\n")
                    if dep.context:
                        f.write(f"     Context: {dep.context[:150]}...\n")


def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas.md")
    output_dir = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas/data")

    print("="*80)
    print("ULTRATHINK DEEP DEPENDENCY EXTRACTION")
    print("="*80)
    print(f"Document: {doc_path}")
    print(f"Output: {output_dir}")
    print()

    analyzer = DeepDependencyAnalyzer(doc_path)

    # Execute analysis pipeline
    analyzer.extract_all_directives()
    analyzer.analyze_dependencies()
    analyzer.build_dependency_graph()
    analyzer.generate_outputs(output_dir)

    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
