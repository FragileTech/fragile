#!/usr/bin/env python3
"""
ENHANCED COMPREHENSIVE DEEP DEPENDENCY EXTRACTION

This version performs ultra-detailed proof analysis to capture ALL dependencies,
including subtle implicit uses, notation dependencies, and logical prerequisites.
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
    reference_type: str
    context: str
    line_in_proof: Optional[int] = None
    evidence: Optional[str] = None
    critical: bool = True

@dataclass
class ProofStep:
    """A single step in a proof"""
    step_number: int
    content: str
    inputs_used: List[str] = field(default_factory=list)
    output_established: str = ""
    justification: str = ""

@dataclass
class DirectiveInfo:
    """Complete information about a MyST directive"""
    label: str
    type: str
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

    # Proof analysis
    proof_steps: List[ProofStep] = field(default_factory=list)
    proof_inputs: List[str] = field(default_factory=list)
    proof_outputs: List[str] = field(default_factory=list)

    def dependency_summary(self) -> Dict:
        total = len(self.explicit_deps) + len(self.implicit_deps) + len(self.assumptions) + len(self.standard_math)
        critical = sum(1 for d in self.explicit_deps + self.implicit_deps + self.assumptions if d.critical)
        external = sum(1 for d in self.explicit_deps + self.implicit_deps + self.assumptions
                      if "framework" in d.target_label or d.target_label.startswith('def-') or d.target_label.startswith('axiom-'))

        return {
            "total_dependencies": total,
            "critical_dependencies": critical,
            "external_framework_dependencies": external,
            "missing_references": len(self.missing_references),
            "proof_steps_analyzed": len(self.proof_steps)
        }


class EnhancedDependencyAnalyzer:
    """Enhanced analyzer with deep proof analysis"""

    # Comprehensive mapping of implicit notation to definitions
    NOTATION_TO_DEFINITION = {
        r'd_\{\\mathcal Y\}\^\{\\mathrm\{Sasaki\}\}': 'sasaki-metric-definition',
        r'\\psi_x': 'lem-squashing-properties-generic',
        r'\\psi_v': 'lem-squashing-properties-generic',
        r'\\varphi': 'lem-projection-lipschitz',
        r'd_\{\\text\{alg\}\}': 'algorithmic-distance-definition',
        r'R_\{\\mathrm\{pos\}\}': 'reward-function-definition',
        r'V_\{\\text\{fit\}\}': 'potential-vector-definition',
        r"\\sigma'_\{\\min,\\mathrm\{patch\}\}": 'def-sasaki-standardization-constants',
        r'L_R\^\{\\mathrm\{Sasaki\}\}': 'lem-euclidean-reward-regularity',
        r'L_\{\\varphi\}': 'lem-projection-lipschitz',
        r'L_\{\\mathrm\{flow\}\}': 'lem-sasaki-kinetic-lipschitz',
        r'\\kappa_\{\\mathrm\{drift\}\}': 'lem-euclidean-geometric-consistency',
        r'\\kappa_\{\\mathrm\{anisotropy\}\}': 'lem-euclidean-geometric-consistency',
        r'C_x\^\{\\(\\mathrm\{pert\}\\)\}': 'lem-euclidean-perturb-moment',
        r'F_\{d,ms\}': 'thm-sasaki-distance-ms',
    }

    # Common proof phrases and their implicit dependencies
    PROOF_PHRASES = {
        r'Lemma.*shows': 'references-previous-lemma',
        r'Theorem.*gives': 'references-previous-theorem',
        r'Definition.*supplies': 'references-definition',
        r'(?:By|From|Using) (?:the )?Axiom of ([A-Z][a-z]+(?: [A-Z][a-z]+)*)': 'axiom',
        r'(?:by|from) compactness': 'compactness-axiom',
        r'(?:by|since).*bounded': 'boundedness-assumption',
        r'(?:by|from) continuity': 'continuity-assumption',
        r'Lipschitz (?:continuity|constant|bound)': 'lipschitz-property',
        r'(?:mean|apply|use|invoke) the triangle inequality': 'triangle-inequality',
        r'Cauchy-?Schwarz': 'cauchy-schwarz',
        r"Jensen'?s inequality": 'jensen-inequality',
        r"H[öo]lder'?s inequality": 'holder-inequality',
        r"Markov'?s inequality": 'markov-inequality',
        r'dominated convergence': 'dominated-convergence-theorem',
        r'Fubini': 'fubini-theorem',
        r'mean[- ]value (?:theorem|inequality)': 'mean-value-theorem',
        r'(?:by|from) definition': 'definition-usage',
        r'(?:as|by) construction': 'construction-usage',
        r'combining.*(bound|inequality|estimate)': 'bound-composition',
        r'substituting': 'substitution-step',
        r'taking expectations': 'expectation-operation',
        r'(?:apply|using) (?:the )?chain rule': 'chain-rule',
    }

    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        self.lines = doc_path.read_text().splitlines()
        self.directives: Dict[str, DirectiveInfo] = {}
        self.dependency_graph = {"nodes": [], "edges": []}
        self.all_labels = set()  # Track all labels in document

    def extract_all_directives(self) -> Dict:
        """Phase 1: Extract all MyST directives with enhanced parsing"""
        print("Phase 1: Extracting MyST directives...")

        current_directive = None
        directive_start = None
        in_directive = False
        brace_depth = 0

        for i, line in enumerate(self.lines, 1):
            # Detect directive start
            if match := re.match(r'^:::+\{prf:(definition|theorem|lemma|proposition|corollary|axiom|proof|algorithm)\}', line):
                directive_type = match.group(1)
                directive_start = i
                current_directive = {"type": directive_type, "lines": [], "content_lines": []}
                in_directive = True
                brace_depth = 0

            # Collect directive content
            elif in_directive:
                current_directive["lines"].append((i, line))

                # Track content (skip dropdown wrappers)
                if not re.match(r'```\{dropdown\}', line) and not line.strip() == '```':
                    current_directive["content_lines"].append(line)

                # Extract label
                if match := re.match(r'^:label:\s+(\S+)', line):
                    current_directive["label"] = match.group(1)
                    self.all_labels.add(match.group(1))

                # Detect directive end (properly handle nested braces)
                if '```{' in line or '::::{' in line:
                    brace_depth += 1
                elif brace_depth > 0 and ('```' in line or '::::' in line):
                    brace_depth -= 1
                elif brace_depth == 0 and re.match(r'^:::+\s*$', line):
                    # Process completed directive
                    if "label" in current_directive or current_directive["type"] == "proof":
                        self._process_directive(current_directive, directive_start, i)
                    in_directive = False
                    current_directive = None

        print(f"  Found {len(self.directives)} labeled directives")
        print(f"  Tracked {len(self.all_labels)} total labels")
        return {"total_directives": len(self.directives), "all_labels": len(self.all_labels)}

    def _process_directive(self, directive_dict: Dict, start_line: int, end_line: int):
        """Process a single directive with enhanced metadata extraction"""
        content_lines = directive_dict["content_lines"]
        content = "\n".join(content_lines)

        # Extract title (improved heuristic)
        title = ""
        for line in content_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith(":") and not line_stripped.startswith("$"):
                # Remove markdown formatting
                title = re.sub(r'\*\*([^*]+)\*\*', r'\1', line_stripped)
                title = re.sub(r'\$([^$]+)\$', r'[math]', title)
                break

        # Count math expressions (better pattern)
        display_math = len(re.findall(r'\$\$[^$]+\$\$', content, re.DOTALL))
        inline_math = len(re.findall(r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)', content))
        math_count = display_math + inline_math

        # Extract first non-trivial math expression
        first_math = ""
        if match := re.search(r'\$\$\s*([^$]+?)\s*\$\$', content, re.DOTALL):
            first_math = match.group(1).strip()[:150]
        elif match := re.search(r'(?<!\$)\$([^$]+?)\$(?!\$)', content):
            first_math = match.group(1).strip()[:100]

        label = directive_dict.get("label", f"unlabeled-{start_line}")

        info = DirectiveInfo(
            label=label,
            type=directive_dict["type"],
            title=title[:200],
            content=content,
            line_range=(start_line, end_line),
            math_expression_count=math_count,
            first_math=first_math
        )

        self.directives[label] = info

    def analyze_dependencies(self):
        """Phase 2-4: Comprehensive dependency analysis"""
        print("Phase 2-4: Analyzing dependencies...")
        print("  Extracting explicit cross-references...")

        for directive in self.directives.values():
            self._extract_explicit_deps(directive)

        print("  Analyzing proofs...")
        for directive in self.directives.values():
            if directive.type != "proof":
                self._analyze_associated_proof(directive)

        print("  Inferring implicit dependencies...")
        for directive in self.directives.values():
            self._infer_implicit_deps_comprehensive(directive)
            self._detect_notation_dependencies(directive)

        print(f"  Analyzed {len(self.directives)} directives")

        # Print statistics
        total_explicit = sum(len(d.explicit_deps) for d in self.directives.values())
        total_implicit = sum(len(d.implicit_deps) for d in self.directives.values())
        total_assumptions = sum(len(d.assumptions) for d in self.directives.values())
        total_standard = sum(len(d.standard_math) for d in self.directives.values())

        print(f"    Explicit deps: {total_explicit}")
        print(f"    Implicit deps: {total_implicit}")
        print(f"    Assumptions: {total_assumptions}")
        print(f"    Standard math: {total_standard}")

    def _extract_explicit_deps(self, directive: DirectiveInfo):
        """Extract all {prf:ref}`label` cross-references"""
        content = directive.content

        for match in re.finditer(r'\{prf:ref\}`([^`]+)`', content):
            target = match.group(1)

            # Get surrounding context (within sentence)
            context_start = max(0, content.rfind('.', 0, match.start()) + 1)
            context_end = content.find('.', match.end())
            if context_end == -1:
                context_end = min(len(content), match.end() + 200)
            context = content[context_start:context_end].strip()

            # Determine usage type from context
            usage_type = "general"
            if any(word in context.lower() for word in ["lemma", "theorem", "shows", "proves", "establishes"]):
                usage_type = "proof-dependency"
            elif any(word in context.lower() for word in ["definition", "defined", "denote"]):
                usage_type = "definition-use"
            elif any(word in context.lower() for word in ["axiom", "assumption", "hypothesis"]):
                usage_type = "axiom-use"

            directive.explicit_deps.append(Dependency(
                target_label=target,
                reference_type=f"prf:ref ({usage_type})",
                context=context[:300],
                critical=True
            ))

    def _analyze_associated_proof(self, directive: DirectiveInfo):
        """Find and analyze the proof associated with a theorem/lemma"""
        # Look for proof in next ~500 lines
        proof_start_line = directive.line_range[1]
        proof_content_lines = []
        in_proof = False
        proof_line_start = 0

        for i in range(proof_start_line, min(proof_start_line + 500, len(self.lines))):
            line = self.lines[i]

            if re.match(r'^:::+\{prf:proof\}', line):
                in_proof = True
                proof_line_start = i + 1
                continue

            if in_proof:
                if re.match(r'^:::+\s*$', line):
                    break
                # Skip dropdown wrappers
                if not re.match(r'```\{dropdown\}', line) and line.strip() != '```':
                    proof_content_lines.append((i + 1, line))

        if not proof_content_lines:
            return

        proof_content = "\n".join(line for _, line in proof_content_lines)

        # Deep proof analysis
        self._analyze_proof_structure(directive, proof_content, proof_content_lines)
        self._extract_proof_phrases(directive, proof_content)
        self._extract_proof_dependencies(directive, proof_content, proof_content_lines)

    def _analyze_proof_structure(self, directive: DirectiveInfo, proof_content: str, proof_lines: List[Tuple[int, str]]):
        """Analyze proof structure and extract steps"""

        # Pattern 1: Numbered steps (Step 1:, 1., etc.)
        step_pattern = r'(?:^|\n)\s*(?:\*\*)?(?:Step\s+)?(\d+)(?:\.|\:)?\s*(?:\*\*)?\s*([^\n]+)'

        for match in re.finditer(step_pattern, proof_content, re.MULTILINE):
            step_num = int(match.group(1))
            step_title = match.group(2).strip()

            # Find step content (until next step or end)
            step_start = match.end()
            next_step = re.search(step_pattern, proof_content[step_start:])
            step_end = step_start + next_step.start() if next_step else len(proof_content)
            step_content = proof_content[step_start:step_end].strip()

            # Analyze step for dependencies
            step_deps = []
            for ref_match in re.finditer(r'\{prf:ref\}`([^`]+)`', step_content):
                step_deps.append(ref_match.group(1))

            directive.proof_steps.append(ProofStep(
                step_number=step_num,
                content=step_title + " " + step_content[:200],
                inputs_used=step_deps,
                justification=f"Line {proof_lines[0][0] + match.start() // 80}"
            ))

    def _extract_proof_phrases(self, directive: DirectiveInfo, proof_content: str):
        """Extract dependencies from proof phrases"""

        for pattern, dep_type in self.PROOF_PHRASES.items():
            for match in re.finditer(pattern, proof_content, re.IGNORECASE):
                # Get context
                context_start = max(0, match.start() - 100)
                context_end = min(len(proof_content), match.end() + 100)
                context = proof_content[context_start:context_end]

                if dep_type == 'axiom':
                    # Extract axiom name
                    axiom_name = match.group(1)
                    target = f"axiom-{axiom_name.lower().replace(' ', '-')}"
                else:
                    target = dep_type

                directive.assumptions.append(Dependency(
                    target_label=target,
                    reference_type=dep_type,
                    context=context.strip(),
                    evidence=f"Proof phrase: {match.group(0)}",
                    critical=(dep_type in ['axiom', 'references-previous-lemma', 'references-previous-theorem'])
                ))

    def _extract_proof_dependencies(self, directive: DirectiveInfo, proof_content: str, proof_lines: List[Tuple[int, str]]):
        """Extract all types of dependencies from proof"""

        # Pattern 1: "Lemma X.Y.Z" or "Theorem X.Y.Z" references
        for match in re.finditer(r'(Lemma|Theorem|Proposition|Corollary)\s+\{prf:ref\}`([^`]+)`', proof_content):
            result_type = match.group(1).lower()
            target = match.group(2)

            directive.explicit_deps.append(Dependency(
                target_label=target,
                reference_type=f"proof-uses-{result_type}",
                context=f"Proof invokes {result_type}: {match.group(0)}",
                critical=True
            ))

        # Pattern 2: Constants from other results
        constant_patterns = [
            (r'L_\{\\varphi\}', 'lem-projection-lipschitz', 'Lipschitz constant from projection lemma'),
            (r'L_R', 'lem-euclidean-reward-regularity', 'Lipschitz constant from reward lemma'),
            (r'L_\{\\mathrm\{flow\}\}', 'lem-sasaki-kinetic-lipschitz', 'Flow Lipschitz constant'),
            (r'\\kappa_\{\\mathrm\{drift\}\}', 'lem-euclidean-geometric-consistency', 'Drift constant'),
            (r'\\sigma_\{\\min,\\mathrm\{patch\}\}', 'def-sasaki-standardization-constants', 'Patched std dev constant'),
        ]

        for pattern, source_label, description in constant_patterns:
            if re.search(pattern, proof_content):
                # Check if already explicitly referenced
                if not any(d.target_label == source_label for d in directive.explicit_deps):
                    directive.implicit_deps.append(Dependency(
                        target_label=source_label,
                        reference_type="constant-usage",
                        context=description,
                        evidence=f"Uses constant defined in {source_label}",
                        critical=True
                    ))

        # Pattern 3: Standard math used
        standard_math_items = [
            ("triangle inequality", "standard-math"),
            ("Cauchy-Schwarz", "standard-math"),
            ("Jensen", "standard-math"),
            ("Hölder", "standard-math"),
            ("Markov", "standard-math"),
            ("dominated convergence", "standard-math"),
            ("Fubini", "standard-math"),
        ]

        for math_name, math_type in standard_math_items:
            if re.search(rf'\b{math_name}\b', proof_content, re.IGNORECASE):
                directive.standard_math.append(Dependency(
                    target_label=math_type,
                    reference_type="standard-result",
                    context=f"Uses {math_name}",
                    critical=False
                ))

    def _infer_implicit_deps_comprehensive(self, directive: DirectiveInfo):
        """Comprehensive inference of implicit dependencies"""
        content = directive.content

        # Check for framework axiom uses
        framework_axioms = [
            ("Bounded Algorithmic Diameter", "def-axiom-bounded-algorithmic-diameter"),
            ("Reward Regularity", "def-axiom-reward-regularity"),
            ("Environmental Richness", "def-axiom-environmental-richness"),
            ("Geometric Consistency", "def-axiom-geometric-consistency"),
            ("Guaranteed Revival", "def-axiom-guaranteed-revival"),
            ("Boundary Regularity", "def-axiom-boundary-regularity"),
            ("Sufficient Amplification", "def-axiom-sufficient-amplification"),
        ]

        for axiom_name, axiom_label in framework_axioms:
            if axiom_name.lower() in content.lower():
                if not any(d.target_label == axiom_label for d in directive.explicit_deps):
                    directive.implicit_deps.append(Dependency(
                        target_label=axiom_label,
                        reference_type="framework-axiom",
                        context=f"References {axiom_name}",
                        evidence=f"Found mention of '{axiom_name}' in content",
                        critical=True
                    ))

        # Check for framework definitions
        framework_defs = [
            ("Fragile Gas Algorithm", "def-fragile-gas-algorithm"),
            ("Fragile Swarm", "def-fragile-swarm-instantiation"),
            ("patched standardis", "def-statistical-properties-measurement"),
            ("logistic rescale", "def-canonical-logistic-rescale-function-example"),
        ]

        for def_phrase, def_label in framework_defs:
            if def_phrase.lower() in content.lower():
                if not any(d.target_label == def_label for d in directive.explicit_deps):
                    directive.implicit_deps.append(Dependency(
                        target_label=def_label,
                        reference_type="framework-definition",
                        context=f"Uses {def_phrase}",
                        evidence=f"Found reference to framework definition",
                        critical=True
                    ))

    def _detect_notation_dependencies(self, directive: DirectiveInfo):
        """Detect dependencies based on mathematical notation"""
        content = directive.content

        for notation_pattern, source_label in self.NOTATION_TO_DEFINITION.items():
            if re.search(notation_pattern, content):
                # Check if already explicitly referenced
                if not any(d.target_label == source_label for d in directive.explicit_deps):
                    directive.implicit_deps.append(Dependency(
                        target_label=source_label,
                        reference_type="notation-use",
                        context=f"Uses notation from {source_label}",
                        evidence=f"Found notation pattern: {notation_pattern}",
                        critical=True
                    ))

    def build_dependency_graph(self):
        """Build complete dependency graph with enhanced edge detection"""
        print("Phase 5: Building dependency graph...")

        # Create nodes
        for label, directive in self.directives.items():
            node_type = directive.type
            if directive.type in ["lemma", "theorem", "proposition"]:
                # Classify by usage
                num_dependents = sum(1 for d in self.directives.values()
                                   if any(dep.target_label == label for dep in d.explicit_deps + d.implicit_deps))
                if num_dependents >= 5:
                    node_type = f"{directive.type} (foundational)"
                elif num_dependents == 0:
                    node_type = f"{directive.type} (leaf)"

            self.dependency_graph["nodes"].append({
                "id": label,
                "type": node_type,
                "title": directive.title,
                "line_range": directive.line_range,
                "dependency_count": len(directive.explicit_deps) + len(directive.implicit_deps)
            })

        # Create edges with deduplication
        edge_set = set()

        for label, directive in self.directives.items():
            all_deps = (directive.explicit_deps + directive.implicit_deps +
                       directive.assumptions)

            for dep in all_deps:
                target = dep.target_label

                # Skip self-references
                if target == label:
                    continue

                # Create edge key for deduplication
                edge_key = (label, target, dep.reference_type)

                if edge_key not in edge_set:
                    edge_set.add(edge_key)

                    # Determine edge type
                    edge_type = "requires" if dep.critical else "references"
                    if "framework" in dep.reference_type:
                        edge_type = "framework-depends"
                    elif "axiom" in dep.reference_type:
                        edge_type = "axiom-depends"
                    elif dep.reference_type == "notation-use":
                        edge_type = "notation-from"

                    self.dependency_graph["edges"].append({
                        "source": label,
                        "target": target,
                        "edge_type": edge_type,
                        "critical": dep.critical,
                        "reference_type": dep.reference_type,
                        "context": dep.context[:100] if dep.context else ""
                    })

        print(f"  Created {len(self.dependency_graph['nodes'])} nodes")
        print(f"  Created {len(self.dependency_graph['edges'])} edges")

    def generate_outputs(self, output_dir: Path):
        """Generate all output files"""
        print("Generating output files...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main analysis file
        analysis = {
            "document": str(self.doc_path),
            "extraction_mode": "enhanced-deep-dependency-analysis",
            "metadata": {
                "total_lines": len(self.lines),
                "total_directives": len(self.directives),
                "directive_types": self._count_directive_types()
            },
            "directives": [
                self._serialize_directive(d) for d in self.directives.values()
            ],
            "dependency_graph": self.dependency_graph,
            "analysis": self._compute_analysis_stats()
        }

        (output_dir / "deep_dependency_analysis.json").write_text(
            json.dumps(analysis, indent=2), encoding='utf-8'
        )

        # Separate graph file
        (output_dir / "dependency_graph.json").write_text(
            json.dumps(self.dependency_graph, indent=2), encoding='utf-8'
        )

        # Human-readable reports
        self._generate_missing_references_report(output_dir)
        self._generate_critical_path_analysis(output_dir)
        self._generate_dependency_summary(output_dir)

        print(f"✓ Generated outputs in {output_dir}")

    def _serialize_directive(self, directive: DirectiveInfo) -> Dict:
        """Serialize directive to dict with all information"""
        return {
            "label": directive.label,
            "type": directive.type,
            "title": directive.title,
            "line_range": directive.line_range,
            "math_expression_count": directive.math_expression_count,
            "first_math": directive.first_math,
            "dependencies": {
                "explicit": [asdict(d) for d in directive.explicit_deps],
                "implicit": [asdict(d) for d in directive.implicit_deps],
                "assumptions": [asdict(d) for d in directive.assumptions],
                "standard_math": [asdict(d) for d in directive.standard_math],
                "missing_references": directive.missing_references
            },
            "proof_analysis": {
                "steps": [asdict(step) for step in directive.proof_steps],
                "inputs": directive.proof_inputs,
                "outputs": directive.proof_outputs
            },
            "dependency_summary": directive.dependency_summary()
        }

    def _count_directive_types(self) -> Dict[str, int]:
        """Count directives by type"""
        counts = defaultdict(int)
        for d in self.directives.values():
            counts[d.type] += 1
        return dict(counts)

    def _compute_analysis_stats(self) -> Dict:
        """Compute comprehensive analysis statistics"""
        all_targets = defaultdict(int)

        for directive in self.directives.values():
            for dep in directive.explicit_deps + directive.implicit_deps + directive.assumptions:
                all_targets[dep.target_label] += 1

        # Find most depended-on results
        most_depended = sorted(all_targets.items(), key=lambda x: x[1], reverse=True)[:15]

        # Find leaf results (not depended on by anything)
        all_sources = set(self.directives.keys())
        leaf_results = [label for label in all_sources if all_targets[label] == 0]

        # Find foundational results (many dependencies)
        foundational = [label for label, count in most_depended if count >= 5]

        # External dependencies (framework axioms/defs)
        external_targets = set()
        for directive in self.directives.values():
            for dep in directive.explicit_deps + directive.implicit_deps + directive.assumptions:
                if dep.target_label not in self.directives:
                    external_targets.add(dep.target_label)

        return {
            "most_depended_on": [{"label": label, "count": count} for label, count in most_depended],
            "leaf_results": leaf_results[:20],
            "foundational_results": foundational,
            "total_unique_dependencies": len(all_targets),
            "external_dependencies": list(external_targets),
            "validation_issues": []
        }

    def _generate_missing_references_report(self, output_dir: Path):
        """Generate human-readable report of missing references"""
        with open(output_dir / "missing_references_report.txt", "w", encoding='utf-8') as f:
            f.write("MISSING REFERENCES REPORT\n")
            f.write("=" * 80 + "\n\n")

            has_issues = False
            for directive in self.directives.values():
                if directive.missing_references:
                    has_issues = True
                    f.write(f"{directive.label} ({directive.type})\n")
                    f.write(f"  Line {directive.line_range[0]}-{directive.line_range[1]}\n")
                    for issue in directive.missing_references:
                        f.write(f"  - {issue['description']} (line {issue['line']})\n")
                    f.write("\n")

            if not has_issues:
                f.write("No missing references detected.\n")

    def _generate_critical_path_analysis(self, output_dir: Path):
        """Generate critical dependency paths to main theorems"""
        with open(output_dir / "critical_path_analysis.txt", "w", encoding='utf-8') as f:
            f.write("CRITICAL PATH ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Find main theorems
            main_theorems = [d for d in self.directives.values() if d.type == "theorem"]
            main_theorems.sort(key=lambda x: len(x.explicit_deps) + len(x.implicit_deps), reverse=True)

            for thm in main_theorems[:10]:
                f.write(f"\n{thm.label}\n")
                f.write(f"  {thm.title}\n")
                f.write(f"  Lines {thm.line_range[0]}-{thm.line_range[1]}\n")
                f.write("-" * 80 + "\n")

                # List direct critical dependencies
                critical_deps = [d for d in thm.explicit_deps + thm.implicit_deps if d.critical]
                f.write(f"  Direct dependencies ({len(critical_deps)}):\n")
                for dep in critical_deps[:20]:
                    f.write(f"    → {dep.target_label} ({dep.reference_type})\n")
                    if dep.context:
                        f.write(f"       {dep.context[:120]}...\n")

                f.write("\n")

    def _generate_dependency_summary(self, output_dir: Path):
        """Generate a concise dependency summary"""
        with open(output_dir / "dependency_summary.txt", "w", encoding='utf-8') as f:
            f.write("DEPENDENCY SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            total_explicit = sum(len(d.explicit_deps) for d in self.directives.values())
            total_implicit = sum(len(d.implicit_deps) for d in self.directives.values())
            total_assumptions = sum(len(d.assumptions) for d in self.directives.values())
            total_standard = sum(len(d.standard_math) for d in self.directives.values())

            f.write(f"Total directives analyzed: {len(self.directives)}\n")
            f.write(f"Total explicit dependencies: {total_explicit}\n")
            f.write(f"Total implicit dependencies: {total_implicit}\n")
            f.write(f"Total assumptions: {total_assumptions}\n")
            f.write(f"Total standard math uses: {total_standard}\n")
            f.write(f"\nTotal edges in graph: {len(self.dependency_graph['edges'])}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Top 10 most depended-on results:\n\n")

            all_targets = defaultdict(int)
            for directive in self.directives.values():
                for dep in directive.explicit_deps + directive.implicit_deps:
                    all_targets[dep.target_label] += 1

            for label, count in sorted(all_targets.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"  {label}: {count} dependencies\n")


def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas.md")
    output_dir = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas/data")

    print("="*80)
    print("ENHANCED ULTRATHINK DEEP DEPENDENCY EXTRACTION")
    print("="*80)
    print(f"Document: {doc_path.name}")
    print(f"Output: {output_dir}")
    print()

    analyzer = EnhancedDependencyAnalyzer(doc_path)

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
