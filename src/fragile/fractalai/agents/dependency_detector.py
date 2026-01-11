"""
Dependency Detector - LLM-Based Implicit Dependency Extraction.

This module uses LLM analysis to discover hidden mathematical dependencies
in theorem statements that are not explicitly marked with {prf:ref} tags.

Detects:
- Mathematical symbols and their definitions (e.g., $E_{S,ms}^2$ → def-structural-error)
- Concepts referenced but not explicitly cited (e.g., "swarm state" → obj-swarm)
- Axioms implicitly assumed (e.g., "bounded domain" → axiom-bounded-domain)
- Parameters used (e.g., $n_c$, $V_{max}$)
- Required properties (e.g., "Lipschitz" → attr-lipschitz)

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fragile.proofs import MathematicalRegistry


@dataclass
class ObjectDependency:
    """Dependency on a mathematical object."""

    label: str
    role: str  # "primary structure", "partition component", "operator", etc.
    context: str  # How the object is used in the theorem


@dataclass
class AxiomDependency:
    """Dependency on an axiom."""

    label: str
    role: str  # "foundational", "constraint", "regularity", etc.
    context: str  # Why the axiom is needed


@dataclass
class ParameterDependency:
    """Dependency on a parameter."""

    label: str
    description: str
    appears_as: str  # LaTeX representation (e.g., "$n_c(S_1, S_2)$")


@dataclass
class PropertyRequirement:
    """Property required on an object."""

    object_label: str
    property_labels: list[str]  # e.g., ["attr-lipschitz", "attr-bounded"]
    context: str  # Why these properties are needed


@dataclass
class ImplicitRelationship:
    """Implicit relationship between objects."""

    source: str
    target: str
    relationship_type: str  # EQUIVALENCE, EMBEDDING, APPROXIMATION, etc.
    context: str
    attributes: dict[str, str]  # e.g., {"error-rate": "O(N^{-1/d})"}


@dataclass
class DependencyReport:
    """Complete dependency analysis for a theorem."""

    theorem_label: str
    input_objects: list[ObjectDependency]
    input_axioms: list[AxiomDependency]
    input_parameters: list[ParameterDependency]
    attributes_required: dict[str, list[str]]  # object_label → property_labels
    implicit_relationships: list[ImplicitRelationship]
    confidence: str  # "high", "medium", "low"
    notes: list[str]  # Any caveats or ambiguities


class DependencyDetector:
    """
    LLM-based implicit dependency detector.

    Uses Gemini 2.5 Pro (and optionally Codex) to analyze theorem content
    and discover mathematical dependencies not explicitly marked with refs.
    """

    def __init__(
        self,
        registry: MathematicalRegistry,
        glossary_path: Path | None = None,
        use_dual_ai: bool = False,
    ):
        """
        Initialize dependency detector.

        Args:
            registry: Registry of all mathematical objects in the framework
            glossary_path: Path to docs/glossary.md for framework context
            use_dual_ai: Whether to use dual AI analysis (Gemini + Codex)
        """
        self.registry = registry
        self.glossary_path = glossary_path or Path("docs/glossary.md")
        self.use_dual_ai = use_dual_ai

        # Build framework context
        self._build_framework_context()

    def _build_framework_context(self) -> None:
        """Build compact framework context for LLM prompts."""
        # Get all available objects
        objects = self.registry.get_all_objects()
        axioms = self.registry.get_all_axioms()

        # Create compact summary
        self.object_summary = "\n".join([
            f"- {obj.label}: {obj.name}"
            for obj in objects[:100]  # Limit for token budget
        ])

        self.axiom_summary = "\n".join([
            f"- {axiom.label}: {axiom.statement}" for axiom in axioms[:50]
        ])

        # Load glossary tags if available
        self.glossary_tags: set[str] = set()
        if self.glossary_path.exists():
            with open(self.glossary_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("- **"):
                        # Extract label from markdown
                        parts = line.split("`")
                        if len(parts) >= 2:
                            self.glossary_tags.add(parts[1])

    def analyze_theorem(
        self, theorem_label: str, theorem_content: str, source_document: Path | None = None
    ) -> DependencyReport:
        """
        Analyze a theorem to discover implicit dependencies.

        Args:
            theorem_label: Label of the theorem being analyzed
            theorem_content: Full theorem statement (and proof if available)
            source_document: Path to source document for additional context

        Returns:
            DependencyReport with discovered dependencies
        """
        # Query Gemini
        gemini_result = self._query_gemini(theorem_label, theorem_content)

        # Optionally query Codex for dual analysis
        codex_result = None
        if self.use_dual_ai:
            codex_result = self._query_codex(theorem_label, theorem_content)

        # Synthesize results
        return self._synthesize_analysis(theorem_label, gemini_result, codex_result)

    def _query_gemini(self, theorem_label: str, theorem_content: str) -> dict:
        """
        Query Gemini 2.5 Pro for dependency analysis.

        Args:
            theorem_label: Label of the theorem
            theorem_content: Full theorem content

        Returns:
            Parsed JSON response from Gemini
        """
        self._construct_dependency_prompt(theorem_label, theorem_content)

        # This would be replaced with actual MCP call in production
        # For now, return structure template
        return {
            "input_objects": [],
            "input_axioms": [],
            "input_parameters": [],
            "attributes_required": {},
            "implicit_relationships": [],
            "confidence": "medium",
            "notes": [],
        }

    def _query_codex(self, theorem_label: str, theorem_content: str) -> dict:
        """
        Query Codex for independent dependency analysis.

        Args:
            theorem_label: Label of the theorem
            theorem_content: Full theorem content

        Returns:
            Parsed JSON response from Codex
        """
        self._construct_dependency_prompt(theorem_label, theorem_content)

        # This would be replaced with actual MCP call in production
        return {
            "input_objects": [],
            "input_axioms": [],
            "input_parameters": [],
            "attributes_required": {},
            "implicit_relationships": [],
            "confidence": "medium",
            "notes": [],
        }

    def _construct_dependency_prompt(self, theorem_label: str, theorem_content: str) -> str:
        """
        Construct LLM prompt for dependency detection.

        Args:
            theorem_label: Label of the theorem
            theorem_content: Full theorem content

        Returns:
            Formatted prompt string
        """
        return f"""You are analyzing a mathematical theorem to discover ALL dependencies.

THEOREM LABEL: {theorem_label}

THEOREM CONTENT:
{theorem_content}

AVAILABLE FRAMEWORK OBJECTS:
{self.object_summary}

AVAILABLE AXIOMS:
{self.axiom_summary}

TASK: Identify every mathematical entity this theorem depends on.

OUTPUT FORMAT (JSON):
{{
  "input_objects": [
    {{"label": "obj-swarm", "role": "primary structure", "context": "State S referenced throughout"}},
    {{"label": "obj-alive-set", "role": "partition component", "context": "A(S) in line 3"}}
  ],
  "input_axioms": [
    {{"label": "axiom-bounded-domain", "role": "foundational", "context": "Assumes compact state space"}}
  ],
  "input_parameters": [
    {{"label": "n_c", "description": "number of status changes", "appears_as": "$n_c(S_1, S_2)$"}}
  ],
  "attributes_required": {{
    "obj-reward-function": ["attr-lipschitz", "attr-bounded"]
  }},
  "implicit_relationships": [
    {{
      "source": "obj-discrete-swarm",
      "target": "obj-continuous-measure",
      "type": "APPROXIMATION",
      "context": "Discrete approx continuous with O(1/N) error",
      "attributes": {{"error-rate": "O(N^{{-1/d}})"}}
    }}
  ],
  "confidence": "high",
  "notes": ["Any caveats or ambiguities"]
}}

CRITICAL RULES:
1. Use ONLY labels that appear in AVAILABLE FRAMEWORK OBJECTS or AVAILABLE AXIOMS
2. For mathematical symbols (e.g., $E_{{S,ms}}^2$, $n_c$), trace them to their definitions
3. Identify concepts mentioned implicitly (e.g., "swarm state" → obj-swarm)
4. Look for axiom assumptions (e.g., "bounded domain", "Lipschitz function")
5. Mark confidence as "low" if unsure about any dependency

Return ONLY valid JSON, no additional text."""

    def _synthesize_analysis(
        self, theorem_label: str, gemini_result: dict, codex_result: dict | None
    ) -> DependencyReport:
        """
        Synthesize dependency reports from AI responses.

        If dual AI is used, compare results and resolve conflicts.

        Args:
            theorem_label: Label of the theorem
            gemini_result: Result from Gemini
            codex_result: Result from Codex (optional)

        Returns:
            Synthesized DependencyReport
        """
        if codex_result is None:
            # Single AI analysis
            return self._parse_ai_result(theorem_label, gemini_result)

        # Dual AI analysis - merge and resolve conflicts
        gemini_report = self._parse_ai_result(theorem_label, gemini_result)
        codex_report = self._parse_ai_result(theorem_label, codex_result)

        return self._merge_reports(gemini_report, codex_report)

    def _parse_ai_result(self, theorem_label: str, ai_result: dict) -> DependencyReport:
        """Parse AI JSON result into DependencyReport."""
        return DependencyReport(
            theorem_label=theorem_label,
            input_objects=[
                ObjectDependency(label=obj["label"], role=obj["role"], context=obj["context"])
                for obj in ai_result.get("input_objects", [])
            ],
            input_axioms=[
                AxiomDependency(label=ax["label"], role=ax["role"], context=ax["context"])
                for ax in ai_result.get("input_axioms", [])
            ],
            input_parameters=[
                ParameterDependency(
                    label=param["label"],
                    description=param["description"],
                    appears_as=param["appears_as"],
                )
                for param in ai_result.get("input_parameters", [])
            ],
            attributes_required=ai_result.get("attributes_required", {}),
            implicit_relationships=[
                ImplicitRelationship(
                    source=rel["source"],
                    target=rel["target"],
                    relationship_type=rel["type"],
                    context=rel["context"],
                    attributes=rel.get("attributes", {}),
                )
                for rel in ai_result.get("implicit_relationships", [])
            ],
            confidence=ai_result.get("confidence", "medium"),
            notes=ai_result.get("notes", []),
        )

    def _merge_reports(
        self, report1: DependencyReport, report2: DependencyReport
    ) -> DependencyReport:
        """
        Merge two dependency reports from dual AI analysis.

        Strategy:
        - Include dependencies that both AIs agree on (high confidence)
        - Include dependencies only one AI found (medium confidence)
        - Note conflicts in the notes field
        """
        # Merge objects (union of both sets)
        merged_objects = self._merge_object_deps(report1.input_objects, report2.input_objects)

        # Merge axioms
        merged_axioms = self._merge_axiom_deps(report1.input_axioms, report2.input_axioms)

        # Merge parameters
        merged_parameters = self._merge_parameter_deps(
            report1.input_parameters, report2.input_parameters
        )

        # Merge attributes_required
        merged_attrs = self._merge_dicts(report1.attributes_required, report2.attributes_required)

        # Merge relationships
        merged_rels = self._merge_relationships(
            report1.implicit_relationships, report2.implicit_relationships
        )

        # Determine overall confidence
        confidence = (
            "high" if report1.confidence == "high" and report2.confidence == "high" else "medium"
        )

        # Collect notes
        notes = list(set(report1.notes + report2.notes))
        notes.append("Synthesized from dual AI analysis (Gemini + Codex)")

        return DependencyReport(
            theorem_label=report1.theorem_label,
            input_objects=merged_objects,
            input_axioms=merged_axioms,
            input_parameters=merged_parameters,
            attributes_required=merged_attrs,
            implicit_relationships=merged_rels,
            confidence=confidence,
            notes=notes,
        )

    def _merge_object_deps(
        self, deps1: list[ObjectDependency], deps2: list[ObjectDependency]
    ) -> list[ObjectDependency]:
        """Merge object dependencies from two reports."""
        # Use dict to deduplicate by label
        merged = {}
        for dep in deps1 + deps2:
            if dep.label not in merged:
                merged[dep.label] = dep
            else:
                # Merge contexts if different
                existing = merged[dep.label]
                if dep.context not in existing.context:
                    merged[dep.label] = ObjectDependency(
                        label=dep.label,
                        role=existing.role,
                        context=f"{existing.context}; {dep.context}",
                    )
        return list(merged.values())

    def _merge_axiom_deps(
        self, deps1: list[AxiomDependency], deps2: list[AxiomDependency]
    ) -> list[AxiomDependency]:
        """Merge axiom dependencies from two reports."""
        merged = {}
        for dep in deps1 + deps2:
            if dep.label not in merged:
                merged[dep.label] = dep
            else:
                existing = merged[dep.label]
                if dep.context not in existing.context:
                    merged[dep.label] = AxiomDependency(
                        label=dep.label,
                        role=existing.role,
                        context=f"{existing.context}; {dep.context}",
                    )
        return list(merged.values())

    def _merge_parameter_deps(
        self, deps1: list[ParameterDependency], deps2: list[ParameterDependency]
    ) -> list[ParameterDependency]:
        """Merge parameter dependencies from two reports."""
        merged = {}
        for dep in deps1 + deps2:
            if dep.label not in merged:
                merged[dep.label] = dep
        return list(merged.values())

    def _merge_relationships(
        self, rels1: list[ImplicitRelationship], rels2: list[ImplicitRelationship]
    ) -> list[ImplicitRelationship]:
        """Merge implicit relationships from two reports."""
        merged = {}
        for rel in rels1 + rels2:
            key = (rel.source, rel.target, rel.relationship_type)
            if key not in merged:
                merged[key] = rel
        return list(merged.values())

    def _merge_dicts(self, dict1: dict, dict2: dict) -> dict:
        """Merge two dictionaries, combining lists for shared keys."""
        merged = dict1.copy()
        for key, val in dict2.items():
            if key in merged:
                # Merge lists, removing duplicates
                merged[key] = list(set(merged[key] + val))
            else:
                merged[key] = val
        return merged
