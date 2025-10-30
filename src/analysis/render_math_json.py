#!/usr/bin/env python3
"""
Render mathematical document JSON to human-readable markdown.

This script converts JSON files conforming to math_schema.json into beautifully
formatted markdown documents suitable for human reading or further processing.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


@dataclass
class RenderOptions:
    """Configuration options for rendering."""

    jupyter_book: bool = False  # Output Jupyter Book format
    include_graph: bool = True  # Include dependency graph
    graph_format: str = "mermaid"  # "mermaid" or "table"
    filter_type: str | None = None  # Filter by directive type
    filter_label: str | None = None  # Filter by label


class MathDocumentRenderer:
    """Renders a mathematical document JSON to markdown."""

    def __init__(self, doc: dict[str, Any], options: RenderOptions = None):
        """Initialize renderer with document and options."""
        self.doc = doc
        self.options = options or RenderOptions()
        self.metadata = doc.get("metadata", {})
        self.directives = doc.get("directives", [])
        self.dependency_graph = doc.get("dependency_graph", {})
        self.constants_glossary = doc.get("constants_glossary", {})
        self.notation_index = doc.get("notation_index", {})

        # Apply filters if specified
        if self.options.filter_type:
            self.directives = [
                d for d in self.directives if d.get("type") == self.options.filter_type
            ]
        if self.options.filter_label:
            self.directives = [
                d for d in self.directives if d.get("label") == self.options.filter_label
            ]

    def render(self) -> str:
        """Render the full document to markdown."""
        sections = [
            self._render_metadata(),
            self._render_toc(),
            self._render_directives(),
        ]

        # Add appendices
        if self.options.include_graph and self.dependency_graph:
            sections.append(self._render_dependency_graph())
        if self.notation_index:
            sections.append(self._render_notation_index())
        if self.constants_glossary:
            sections.append(self._render_constants_glossary())

        return "\n\n".join(filter(None, sections))

    # ==================== Metadata and TOC ====================

    def _render_metadata(self) -> str:
        """Render document metadata."""
        lines = []

        # Title
        title = self.metadata.get("title", "Untitled Document")
        lines.extend((f"# {title}", ""))

        # Metadata line
        meta_parts = []
        if version := self.metadata.get("version"):
            meta_parts.append(f"**Version:** {version}")
        if chapter := self.metadata.get("chapter"):
            meta_parts.append(f"**Chapter:** {chapter}")
        if rigor := self.metadata.get("rigor_level"):
            meta_parts.append(f"**Rigor:** {rigor}")

        if meta_parts:
            lines.extend((" | ".join(meta_parts), ""))

        # Authors
        if authors := self.metadata.get("authors"):
            lines.extend((f"**Authors:** {', '.join(authors)}", ""))

        # Dates
        if date_created := self.metadata.get("date_created"):
            lines.append(f"**Created:** {date_created}")
        if date_modified := self.metadata.get("date_modified"):
            lines.append(f"**Modified:** {date_modified}")
        if date_created or date_modified:
            lines.append("")

        # Document ID
        if doc_id := self.metadata.get("document_id"):
            lines.extend((f"**Document ID:** `{doc_id}`", ""))

        # Abstract
        if abstract := self.metadata.get("abstract"):
            lines.extend(("## Abstract", "", abstract, ""))

        # Peer review status
        if peer_review := self.metadata.get("peer_review_status"):
            lines.extend(("## Peer Review Status", ""))
            if gemini := peer_review.get("gemini_review"):
                status = gemini.get("status", "unknown")
                lines.append(f"- **Gemini Review:** {status}")
                if review_file := gemini.get("review_file"):
                    lines.append(f"  - Review: `{review_file}`")
            if codex := peer_review.get("codex_review"):
                status = codex.get("status", "unknown")
                lines.append(f"- **Codex Review:** {status}")
                if review_file := codex.get("review_file"):
                    lines.append(f"  - Review: `{review_file}`")
            if consensus := peer_review.get("dual_review_consensus"):
                lines.append(f"- **Consensus:** {'✓' if consensus else '✗'}")
            lines.append("")

        # Publication readiness aggregate
        if readiness := self.metadata.get("publication_readiness_aggregate"):
            lines.extend(("## Publication Readiness", ""))

            # Overall verdict with badge
            if verdict := readiness.get("overall_verdict"):
                verdict_badge = self._verdict_badge(verdict)
                lines.extend((f"**Overall Status:** {verdict_badge}", ""))

            # Aggregate scores
            if scores := readiness.get("aggregate_scores"):
                lines.extend(("**Aggregate Scores:**", ""))
                if rigor := scores.get("rigor"):
                    lines.append(f"- Mathematical Rigor: {rigor:.1f}/10")
                if soundness := scores.get("soundness"):
                    lines.append(f"- Logical Soundness: {soundness:.1f}/10")
                if consistency := scores.get("consistency"):
                    lines.append(f"- Framework Consistency: {consistency:.1f}/10")
                lines.append("")

            # Directive summary
            if summary := readiness.get("directive_summary"):
                total = summary.get("total_directives", 0)
                reviewed = summary.get("reviewed_directives", 0)
                ready = summary.get("ready_count", 0)
                minor = summary.get("minor_revisions_count", 0)
                major = summary.get("major_revisions_count", 0)
                reject = summary.get("reject_count", 0)

                lines.extend((f"**Directive Summary:** {reviewed}/{total} reviewed", ""))
                if ready > 0:
                    lines.append(f"- ✅ Ready: {ready}")
                if minor > 0:
                    lines.append(f"- ⚠️ Minor revisions: {minor}")
                if major > 0:
                    lines.append(f"- 🔧 Major revisions: {major}")
                if reject > 0:
                    lines.append(f"- ❌ Rejected: {reject}")
                lines.append("")

            # Blocking issues
            if blocking := readiness.get("blocking_issues"):
                if blocking:
                    lines.extend((
                        f"**🚫 Blocking Issues:** {len(blocking)} critical issues must be resolved",
                        "",
                    ))
                    for block in blocking[:3]:  # Show first 3
                        directive_label = block.get("directive_label", "unknown")
                        issue = block.get("issue", {})
                        title = issue.get("title", "Untitled issue")
                        lines.append(f"- `{directive_label}`: {title}")
                    if len(blocking) > 3:
                        lines.append(f"- ... and {len(blocking) - 3} more")
                    lines.append("")

            # Development summary
            if dev_summary := readiness.get("development_summary"):
                avg_completeness = dev_summary.get("average_completeness", 0)
                lines.extend((
                    f"**Development Maturity:** {avg_completeness:.0f}% average completeness",
                    "",
                ))

                stages = []
                if sketch := dev_summary.get("sketch_count"):
                    stages.append(f"Sketch: {sketch}")
                if partial := dev_summary.get("partial_count"):
                    stages.append(f"Partial: {partial}")
                if complete := dev_summary.get("complete_count"):
                    stages.append(f"Complete: {complete}")
                if verified := dev_summary.get("verified_count"):
                    stages.append(f"Verified: {verified}")
                if published := dev_summary.get("published_count"):
                    stages.append(f"Published: {published}")

                if stages:
                    lines.extend(("- " + " | ".join(stages), ""))

        # Dependencies
        if deps := self.metadata.get("dependencies"):
            lines.extend(("## Prerequisites", ""))
            for dep in deps:
                lines.append(f"- `{dep}`")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_toc(self) -> str:
        """Render table of contents."""
        if not self.directives:
            return ""

        lines = ["## Table of Contents", ""]

        # Group by type
        by_type: dict[str, list[dict]] = {}
        for directive in self.directives:
            dtype = directive.get("type", "unknown")
            by_type.setdefault(dtype, []).append(directive)

        # Render each type group
        type_order = [
            "definition",
            "axiom",
            "theorem",
            "lemma",
            "proposition",
            "corollary",
            "proof",
            "algorithm",
            "remark",
            "observation",
            "conjecture",
            "example",
            "property",
        ]

        for dtype in type_order:
            if dtype not in by_type:
                continue

            # Section header
            lines.extend((f"### {dtype.title()}s", ""))

            # List items
            for i, directive in enumerate(by_type[dtype], 1):
                label = directive.get("label", "unlabeled")
                title = directive.get("title", "Untitled")
                importance = ""
                if dtype in {"theorem", "lemma", "proposition"}:
                    imp = directive.get("importance", "")
                    if imp == "foundational":
                        importance = " ⭐"
                    elif imp == "main-result":
                        importance = " ★"

                lines.append(f"{i}. [{title}](#{label}){importance}")

            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    # ==================== Directive Rendering ====================

    def _render_directives(self) -> str:
        """Render all directives."""
        if not self.directives:
            return ""

        # Group by type for section headers
        by_type: dict[str, list[dict]] = {}
        for directive in self.directives:
            dtype = directive.get("type", "unknown")
            by_type.setdefault(dtype, []).append(directive)

        sections = []
        type_order = [
            "definition",
            "axiom",
            "theorem",
            "lemma",
            "proposition",
            "corollary",
            "proof",
            "algorithm",
            "remark",
            "observation",
            "conjecture",
            "example",
            "property",
        ]

        for dtype in type_order:
            if dtype not in by_type:
                continue

            # Section header
            sections.extend((f"## {dtype.title()}s", ""))

            # Render each directive
            for directive in by_type[dtype]:
                renderer_map = {
                    "definition": self._render_definition,
                    "axiom": self._render_axiom,
                    "theorem": self._render_theorem,
                    "lemma": self._render_lemma,
                    "proposition": self._render_proposition,
                    "corollary": self._render_corollary,
                    "proof": self._render_proof,
                    "algorithm": self._render_algorithm,
                    "remark": self._render_remark,
                    "observation": self._render_observation,
                    "conjecture": self._render_conjecture,
                    "example": self._render_example,
                    "property": self._render_property,
                }

                renderer = renderer_map.get(dtype)
                if renderer:
                    sections.extend((renderer(directive), ""))

        return "\n".join(sections)

    def _render_definition(self, d: dict[str, Any]) -> str:
        """Render a definition directive."""
        lines = []

        # Header
        lines.append(f"### Definition: {d.get('title', 'Untitled')}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Defined objects
        if defined_objects := d.get("defined_objects"):
            lines.extend(("**Defined Objects:**", ""))
            for obj in defined_objects:
                name = obj.get("name", "")
                symbol = obj.get("symbol", "")
                math_def = obj.get("mathematical_definition", "")
                obj_type = obj.get("type", "")

                header = f"- **{name}"
                if symbol:
                    header += f" ({symbol})"
                header += f"** [{obj_type}]"

                lines.extend((header, f"  - {math_def}"))

                # Properties
                if properties := obj.get("properties"):
                    lines.append("  - **Properties:**")
                    for prop in properties:
                        # Handle both string and object formats
                        if isinstance(prop, str):
                            lines.append(f"    - {prop}")
                        else:
                            prop_str = self._render_math_property(prop, indent="    ")
                            lines.append(prop_str)

                lines.append("")

        # Examples
        if examples := d.get("examples"):
            lines.extend(("**Examples:**", ""))
            for ex in examples:
                desc = ex.get("description", "")
                instance = ex.get("instance", "")
                lines.append(f"- {desc}")
                if instance:
                    lines.append(f"  - {instance}")
                lines.append("")

        # Counterexamples
        if counterexamples := d.get("counterexamples"):
            lines.extend(("**Counterexamples:**", ""))
            for cex in counterexamples:
                desc = cex.get("description", "")
                reason = cex.get("reason", "")
                lines.extend((f"- {desc}", f"  - Fails because: {reason}", ""))

        # Related concepts
        if related := d.get("related_concepts"):
            lines.extend(("**Related Concepts:**", ""))
            for ref in related:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_axiom(self, d: dict[str, Any]) -> str:
        """Render an axiom directive."""
        lines = []

        # Header
        lines.append(f"### Axiom: {d.get('title', 'Untitled')}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Axiomatic parameters
        if params := d.get("axiomatic_parameters"):
            lines.extend((
                "**Axiomatic Parameters:**",
                "",
                "| Symbol | Description | Type | Conditions | Sensitivity |",
                "|--------|-------------|------|------------|-------------|",
            ))

            for param in params:
                symbol = param.get("symbol", "")
                desc = param.get("description", "")
                ptype = param.get("type", "")

                # Handle conditions: can be string, list, or None
                conds_raw = param.get("conditions", [])
                if isinstance(conds_raw, str):
                    conds_str = conds_raw
                elif isinstance(conds_raw, list):
                    conds_str = ", ".join(str(c) for c in conds_raw) if conds_raw else "-"
                else:
                    conds_str = "-"

                sens = param.get("sensitivity", "-")

                lines.append(f"| {symbol} | {desc} | {ptype} | {conds_str} | {sens} |")

            lines.append("")

        # Category
        if category := d.get("category"):
            lines.extend((f"**Category:** {category}", ""))

        # Failure modes
        if failure_modes := d.get("failure_modes"):
            lines.extend(("**Failure Modes:**", ""))
            for fm in failure_modes:
                condition = fm.get("condition", "")
                consequence = fm.get("consequence", "")
                diagnostic = fm.get("diagnostic", "")

                lines.extend((f"- **When:** {condition}", f"  - **Consequence:** {consequence}"))
                if diagnostic:
                    lines.append(f"  - **Diagnostic:** {diagnostic}")
                lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_theorem(self, d: dict[str, Any]) -> str:
        """Render a theorem directive."""
        return self._render_theorem_like(d, "Theorem")

    def _render_lemma(self, d: dict[str, Any]) -> str:
        """Render a lemma directive."""
        return self._render_theorem_like(d, "Lemma")

    def _render_proposition(self, d: dict[str, Any]) -> str:
        """Render a proposition directive."""
        return self._render_theorem_like(d, "Proposition")

    def _render_theorem_like(self, d: dict[str, Any], dtype: str) -> str:
        """Render theorem/lemma/proposition (common structure)."""
        lines = []

        # Header with importance
        importance = d.get("importance", "")
        importance_marker = ""
        if importance == "foundational":
            importance_marker = " ⭐"
        elif importance == "main-result":
            importance_marker = " ★"

        lines.extend((
            f"### {dtype}: {d.get('title', 'Untitled')}{importance_marker}",
            self._render_directive_metadata(d),
        ))
        if importance:
            lines.append(f"**Importance:** {importance}")
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Hypotheses
        if hypotheses := d.get("hypotheses"):
            lines.extend(("**Hypotheses:**", ""))
            for i, hyp in enumerate(hypotheses, 1):
                hyp_str = self._render_assumption(hyp, number=i)
                lines.append(hyp_str)
            lines.append("")

        # Conclusion
        if conclusion := d.get("conclusion"):
            lines.extend(("**Conclusion:**", ""))
            statement = conclusion.get("statement", "")
            lines.extend((statement, ""))

            # Properties established
            if props := conclusion.get("properties_established"):
                lines.extend(("**Properties Established:**", ""))
                for prop in props:
                    # Handle both string and object formats
                    if isinstance(prop, str):
                        lines.append(f"- {prop}")
                    else:
                        prop_str = self._render_math_property(prop)
                        lines.append(prop_str)
                lines.append("")

            # Quantitative bounds
            if bounds := conclusion.get("quantitative_bounds"):
                lines.extend(("**Quantitative Bounds:**", ""))

                # Check if this is schema format (dict of bound objects) or custom format (flat dict)
                # Schema format has keys that map to dicts with "bound", "type", "tightness"
                # Custom format is a flat dict with any keys
                is_schema_format = False
                if bounds:
                    first_value = next(iter(bounds.values()))
                    if isinstance(first_value, dict) and "bound" in first_value:
                        is_schema_format = True

                if is_schema_format:
                    # Schema format: dict of bound objects
                    for name, bound_info in bounds.items():
                        if isinstance(bound_info, str):
                            lines.append(f"- **{name}**: {bound_info}")
                        else:
                            bound = bound_info.get("bound", "")
                            btype = bound_info.get("type", "")
                            tightness = bound_info.get("tightness", "")
                            lines.append(f"- **{name}** [{btype}]: {bound}")
                            if tightness:
                                lines.append(f"  - Tightness: {tightness}")
                else:
                    # Custom format: flat dict with arbitrary keys
                    for key, value in bounds.items():
                        if isinstance(value, list):
                            value_str = ", ".join(value)
                            lines.append(f"- **{key}**: {value_str}")
                        elif isinstance(value, dict):
                            lines.append(f"- **{key}**:")
                            for subkey, subvalue in value.items():
                                lines.append(f"  - {subkey}: {subvalue}")
                        else:
                            lines.append(f"- **{key}**: {value}")

                lines.append("")

        # Proof reference
        if proof_ref := d.get("proof_reference"):
            lines.append("**Proof:**")
            if proof_ref.get("inline"):
                lines.append("See proof immediately below.")
            elif label := proof_ref.get("label"):
                location = proof_ref.get("location", "")
                lines.append(f"[Proof: {label}](#{label})")
                if location:
                    lines.append(f"(Located in: {location})")
            elif proof_ref.get("deferred"):
                reason = proof_ref.get("reason", "")
                reference = proof_ref.get("reference", "")
                lines.append(f"Deferred: {reason}")
                if reference:
                    lines.append(f"See: {reference}")
            lines.append("")

        # Computational verification
        if comp_ver := d.get("computational_verification"):
            lines.extend((self._render_computational_verification(comp_ver), ""))

        lines.append("---")
        return "\n".join(lines)

    def _render_corollary(self, d: dict[str, Any]) -> str:
        """Render a corollary directive."""
        lines = []

        # Header
        lines.append(f"### Corollary: {d.get('title', 'Untitled')}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Follows from
        if follows_from := d.get("follows_from"):
            lines.extend(("**Follows From:**", ""))
            for ref in follows_from:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        # Conclusion
        if conclusion := d.get("conclusion"):
            statement = conclusion.get("statement", "")
            lines.extend(("**Statement:**", "", statement, ""))

        # Proof reference
        if proof_ref := d.get("proof_reference"):
            lines.append("**Proof:**")
            if proof_ref.get("immediate"):
                just = proof_ref.get("justification", "Follows immediately")
                lines.append(f"{just}")
            elif proof_ref.get("inline"):
                lines.append("See proof below.")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_proof(self, d: dict[str, Any]) -> str:
        """Render a proof directive."""
        lines = []

        # Header
        lines.extend((
            f"### Proof: {d.get('title', 'Untitled')}",
            self._render_directive_metadata(d),
            "",
        ))

        # Proves
        if proves := d.get("proves"):
            lines.extend((f"**Proves:** {self._render_cross_reference(proves)}", ""))

        # Proof type and difficulty
        meta_parts = []
        if proof_type := d.get("proof_type"):
            meta_parts.append(f"**Type:** {proof_type}")
        if difficulty := d.get("difficulty"):
            meta_parts.append(f"**Difficulty:** {difficulty}")
        if rigor := d.get("rigor_level"):
            meta_parts.append(f"**Rigor:** {rigor}/10")

        if meta_parts:
            lines.extend((" | ".join(meta_parts), ""))

        # Strategy
        if strategy := d.get("strategy"):
            lines.extend(("**Strategy:**", "", strategy, ""))

        # Prerequisites
        if prereqs := d.get("prerequisites"):
            lines.extend(("**Prerequisites:**", ""))
            for ref in prereqs:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        # Key insights
        if insights := d.get("key_insights"):
            lines.extend(("**Key Insights:**", ""))
            for insight in insights:
                lines.append(f"- {insight}")
            lines.append("")

        # Proof steps
        if steps := d.get("steps"):
            lines.extend(("**Proof:**", "", self._render_proof_steps(steps, level=0), ""))

        # Alternative approaches
        if alternatives := d.get("alternative_approaches"):
            lines.extend(("**Alternative Approaches:**", ""))
            for alt in alternatives:
                desc = alt.get("description", "")
                adv = alt.get("advantages", "")
                dis = alt.get("disadvantages", "")
                lines.append(f"- **{desc}**")
                if adv:
                    lines.append(f"  - Advantages: {adv}")
                if dis:
                    lines.append(f"  - Disadvantages: {dis}")
                lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_proof_steps(self, steps: list[dict[str, Any]], level: int = 0) -> str:
        """Recursively render proof steps with substeps."""
        lines = []
        indent = "  " * level

        for step in steps:
            step_id = step.get("id", "")
            title = step.get("title", "")
            content = step.get("content", "")
            step.get("type", "")
            techniques = step.get("techniques", [])

            # Handle justification: can be string, list, or None
            justification_raw = step.get("justification", [])
            if isinstance(justification_raw, str):
                justification = [justification_raw]
            elif justification_raw is None:
                justification = []
            else:
                justification = justification_raw

            intermediate = step.get("intermediate_result", "")
            substeps = step.get("substeps", [])

            # Step header
            header = f"{indent}**{step_id}"
            if title:
                header += f": {title}"
            header += "**"
            lines.extend((header, ""))

            # Content
            for line in content.split("\n"):
                lines.append(f"{indent}{line}")
            lines.append("")

            # Techniques
            if techniques:
                tech_str = ", ".join(techniques)
                lines.append(f"{indent}*Techniques:* {tech_str}")

            # Justification
            if justification:
                just_strs = []
                for ref in justification:
                    if isinstance(ref, str):
                        just_strs.append(ref)
                    else:
                        just_strs.append(self._render_cross_reference(ref))
                lines.append(f"{indent}*Justification:* {', '.join(just_strs)}")

            # Intermediate result
            if intermediate:
                lines.append(f"{indent}*Result:* {intermediate}")

            if techniques or justification or intermediate:
                lines.append("")

            # Recursive substeps
            if substeps:
                lines.append(self._render_proof_steps(substeps, level=level + 1))

        return "\n".join(lines)

    def _render_algorithm(self, d: dict[str, Any]) -> str:
        """Render an algorithm directive."""
        lines = []

        # Header
        lines.append(f"### Algorithm: {d.get('title', 'Untitled')}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Inputs
        if inputs := d.get("inputs"):
            lines.extend((
                "**Inputs:**",
                "",
                "| Name | Type | Description | Constraints |",
                "|------|------|-------------|-------------|",
            ))

            for inp in inputs:
                name = inp.get("name", "")
                itype = inp.get("type", "")
                desc = inp.get("description", "")
                constraints = inp.get("constraints", [])
                const_str = ", ".join(constraints) if constraints else "-"

                lines.append(f"| {name} | {itype} | {desc} | {const_str} |")

            lines.append("")

        # Outputs
        if outputs := d.get("outputs"):
            lines.extend((
                "**Outputs:**",
                "",
                "| Name | Type | Description | Guarantees |",
                "|------|------|-------------|------------|",
            ))

            for out in outputs:
                name = out.get("name", "")
                otype = out.get("type", "")
                desc = out.get("description", "")
                guarantees = out.get("guarantees", [])
                guar_str = ", ".join(guarantees) if guarantees else "-"

                lines.append(f"| {name} | {otype} | {desc} | {guar_str} |")

            lines.append("")

        # Steps
        if steps := d.get("steps"):
            lines.extend(("**Steps:**", ""))

            for step in steps:
                step_num = step.get("step_number", "")
                desc = step.get("description", "")
                pseudocode = step.get("pseudocode", "")
                math_op = step.get("mathematical_operation", "")
                complexity = step.get("complexity", "")

                if step_num:
                    lines.append(f"{step_num}. {desc}")
                else:
                    lines.append(f"- {desc}")

                if pseudocode:
                    lines.extend(("   ```", f"   {pseudocode}", "   ```"))

                if math_op:
                    lines.append(f"   - Math: {math_op}")

                if complexity:
                    lines.append(f"   - Complexity: {complexity}")

                lines.append("")

        # Overall complexity
        if complexity := d.get("complexity"):
            lines.extend(("**Complexity:**", ""))
            if time := complexity.get("time"):
                lines.append(f"- **Time:** {time}")
            if space := complexity.get("space"):
                lines.append(f"- **Space:** {space}")
            if worst := complexity.get("worst_case"):
                lines.append(f"- **Worst Case:** {worst}")
            if avg := complexity.get("average_case"):
                lines.append(f"- **Average Case:** {avg}")
            lines.append("")

        # Implementation
        if impl := d.get("implementation"):
            lines.extend(("**Implementation:**", ""))
            if path := impl.get("path"):
                lines.append(f"- **Code:** `{path}`")
            if lang := impl.get("language"):
                lines.append(f"- **Language:** {lang}")
            if entry := impl.get("entry_point"):
                lines.append(f"- **Entry Point:** `{entry}`")
            if tests := impl.get("tests"):
                lines.append(f"- **Tests:** `{tests}`")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_remark(self, d: dict[str, Any]) -> str:
        """Render a remark directive."""
        lines = []

        # Header
        remark_type = d.get("remark_type", "")
        type_str = f" [{remark_type}]" if remark_type else ""
        lines.append(f"### Remark: {d.get('title', 'Untitled')}{type_str}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Relates to
        if relates_to := d.get("relates_to"):
            lines.extend(("**Relates To:**", ""))
            for ref in relates_to:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_observation(self, d: dict[str, Any]) -> str:
        """Render an observation directive."""
        lines = []

        # Header
        empirical = " [empirical]" if d.get("empirical") else ""
        lines.append(f"### Observation: {d.get('title', 'Untitled')}{empirical}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Evidence
        if evidence := d.get("evidence"):
            lines.extend(("**Evidence:**", ""))
            for ev in evidence:
                ev_type = ev.get("type", "")
                desc = ev.get("description", "")
                ref = ev.get("reference", "")
                lines.append(f"- **[{ev_type}]** {desc}")
                if ref:
                    lines.append(f"  - Reference: {ref}")
                lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_conjecture(self, d: dict[str, Any]) -> str:
        """Render a conjecture directive."""
        lines = []

        # Header
        confidence = d.get("confidence", "")
        lines.append(f"### Conjecture: {d.get('title', 'Untitled')} [{confidence} confidence]")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Evidence
        if evidence := d.get("evidence"):
            lines.extend(("**Evidence:**", ""))
            for ev in evidence:
                ev_type = ev.get("type", "")
                desc = ev.get("description", "")
                lines.append(f"- **[{ev_type}]** {desc}")
            lines.append("")

        # Partial results
        if partial := d.get("partial_results"):
            lines.extend(("**Partial Results:**", ""))
            for ref in partial:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        # Obstacles
        if obstacles := d.get("obstacles"):
            lines.extend(("**Known Obstacles:**", ""))
            for obs in obstacles:
                lines.append(f"- {obs}")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _render_example(self, d: dict[str, Any]) -> str:
        """Render an example directive."""
        lines = []

        # Header
        lines.append(f"### Example: {d.get('title', 'Untitled')}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Demonstrates
        if demonstrates := d.get("demonstrates"):
            lines.extend(("**Demonstrates:**", ""))
            for ref in demonstrates:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        # Setup
        if setup := d.get("setup"):
            lines.extend(("**Setup:**", "", setup, ""))

        # Calculation
        if calc := d.get("calculation"):
            lines.extend(("**Calculation:**", "", calc, ""))

        # Conclusion
        if conclusion := d.get("conclusion"):
            lines.extend(("**Conclusion:**", "", conclusion, ""))

        lines.append("---")
        return "\n".join(lines)

    def _render_property(self, d: dict[str, Any]) -> str:
        """Render a property directive."""
        lines = []

        # Header
        prop_type = d.get("property_type", "")
        type_str = f" [{prop_type}]" if prop_type else ""
        lines.append(f"### Property: {d.get('title', 'Untitled')}{type_str}")
        lines.append(self._render_directive_metadata(d))
        lines.append("")

        # Statement
        lines.append(d.get("statement", ""))
        lines.append("")

        # Applies to
        if applies_to := d.get("applies_to"):
            lines.extend(("**Applies To:**", ""))
            for ref in applies_to:
                lines.append(f"- {self._render_cross_reference(ref)}")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    # ==================== Utility Renderers ====================

    def _render_directive_metadata(self, d: dict[str, Any]) -> str:
        """Render common directive metadata (label, tags, source, review status)."""
        lines = []

        # First line: label, tags, source
        parts = []
        if label := d.get("label"):
            parts.append(f"**Label:** `{label}`")
        if tags := d.get("tags"):
            tag_str = ", ".join(tags)
            parts.append(f"**Tags:** {tag_str}")
        if source := d.get("source"):
            doc = source.get("document", "")
            section = source.get("section", "")
            if doc or section:
                src_str = f"{doc} § {section}" if section else doc
                parts.append(f"**Source:** {src_str}")

        if parts:
            lines.append(" | ".join(parts))

        # Review scores (if present)
        if peer_review := d.get("peer_review"):
            lines.extend(("", self._render_review_scores(peer_review)))

        # Development status (if present)
        if dev_status := d.get("development_status"):
            lines.extend(("", self._render_development_status(dev_status)))

        # Sketch linkage (if present)
        if sketch_link := d.get("sketch_linkage"):
            lines.extend(("", self._render_sketch_linkage(sketch_link)))

        return "\n".join(lines) if lines else ""

    def _render_cross_reference(self, ref: dict[str, Any]) -> str:
        """Render a cross-reference."""
        label = ref.get("label", "unknown")
        ref_type = ref.get("type", "item")
        role = ref.get("role", "")
        desc = ref.get("description", "")
        location = ref.get("location", "")

        # Base reference
        result = f"[{ref_type.title()}: {label}](#{label})"

        # Add role/description
        if role:
            result += f" ({role}"
            if desc:
                result += f": {desc}"
            result += ")"
        elif desc:
            result += f" ({desc})"

        # Add location
        if location:
            result += f" @ {location}"

        return result

    def _render_math_property(self, prop: dict[str, Any], indent: str = "") -> str:
        """Render a mathematical property."""
        name = prop.get("name", "")
        statement = prop.get("statement", "")
        quantitative = prop.get("quantitative", False)
        scope = prop.get("scope", "")

        result = f"{indent}- **{name}"
        if quantitative:
            result += " [quantitative]"
        result += f":** {statement}"

        if scope:
            result += f" (scope: {scope})"

        return result

    def _render_assumption(self, assumption: dict[str, Any], number: int | None = None) -> str:
        """Render a mathematical assumption."""
        statement = assumption.get("statement", "")
        ass_type = assumption.get("type", "")
        justification = assumption.get("justification", "")

        prefix = f"{number}. " if number else "- "
        result = f"{prefix}**[{ass_type}]** {statement}"

        if justification:
            result += f"\n   - Justification: {justification}"

        return result

    def _render_computational_verification(self, comp_ver: dict[str, Any]) -> str:
        """Render computational verification info."""
        lines = ["**Computational Verification:**", ""]

        ver_type = comp_ver.get("type", "")
        status = comp_ver.get("status", "")
        lines.extend((f"- **Type:** {ver_type}", f"- **Status:** {status}"))

        if script := comp_ver.get("script_path"):
            lines.append(f"- **Script:** `{script}`")

        if notebook := comp_ver.get("notebook_path"):
            lines.append(f"- **Notebook:** `{notebook}`")

        if desc := comp_ver.get("description"):
            lines.append(f"- **Description:** {desc}")

        if results := comp_ver.get("results"):
            if summary := results.get("summary"):
                lines.append(f"- **Results:** {summary}")

        return "\n".join(lines)

    # ==================== Appendices ====================

    def _render_dependency_graph(self) -> str:
        """Render dependency graph."""
        lines = ["## Dependency Graph", ""]

        nodes = self.dependency_graph.get("nodes", [])
        edges = self.dependency_graph.get("edges", [])

        if not nodes and not edges:
            return ""

        if self.options.graph_format == "mermaid":
            lines.extend(("```mermaid", "graph TD"))

            for edge in edges:
                from_label = edge.get("from", "")
                to_label = edge.get("to", "")
                rel = edge.get("relationship", "")
                critical = edge.get("critical", False)

                arrow = "==>" if critical else "-->"
                lines.append(f"  {from_label} {arrow}|{rel}| {to_label}")

            lines.append("```")

        else:  # table format
            lines.extend((
                "| From | To | Relationship | Critical |",
                "|------|-----|--------------|----------|",
            ))

            for edge in edges:
                from_label = edge.get("from", "")
                to_label = edge.get("to", "")
                rel = edge.get("relationship", "")
                critical = "✓" if edge.get("critical", False) else ""

                lines.append(f"| {from_label} | {to_label} | {rel} | {critical} |")

        lines.extend(("", "---"))
        return "\n".join(lines)

    def _render_notation_index(self) -> str:
        """Render notation index."""
        lines = ["## Notation Index", ""]

        if not self.notation_index:
            return ""

        # Check if this is schema format (objects) or simple format (strings)
        first_value = next(iter(self.notation_index.values()))
        is_simple_format = isinstance(first_value, str)

        if is_simple_format:
            # Simple format: symbol -> description string
            lines.extend(("| Symbol | Description |", "|--------|-------------|"))

            for symbol, desc in sorted(self.notation_index.items()):
                lines.append(f"| {symbol} | {desc} |")
        else:
            # Schema format: symbol -> object with description, first_use, scope
            lines.extend((
                "| Symbol | Description | First Use | Scope |",
                "|--------|-------------|-----------|-------|",
            ))

            for key, notation in sorted(self.notation_index.items()):
                symbol = notation.get("symbol", "")
                desc = notation.get("description", "")
                first_use = notation.get("first_use", "")
                scope = notation.get("scope", "")

                lines.append(f"| {symbol} | {desc} | {first_use} | {scope} |")

        lines.extend(("", "---"))
        return "\n".join(lines)

    def _render_constants_glossary(self) -> str:
        """Render constants glossary."""
        lines = ["## Constants Glossary", ""]

        if not self.constants_glossary:
            return ""

        lines.extend((
            "| Symbol | Value | Description | Dependencies | Used In |",
            "|--------|-------|-------------|--------------|---------|",
        ))

        for key, constant in sorted(self.constants_glossary.items()):
            symbol = constant.get("symbol", "")
            value = constant.get("value", "")
            desc = constant.get("description", "")
            deps = constant.get("depends_on", [])
            deps_str = ", ".join(deps) if deps else "-"
            used_in = constant.get("used_in", [])
            used_str = ", ".join(used_in[:3])  # Limit to 3 for brevity
            if len(used_in) > 3:
                used_str += "..."

            lines.append(f"| {symbol} | {value} | {desc} | {deps_str} | {used_str} |")

        lines.extend(("", "---"))
        return "\n".join(lines)

    # ==================== Review Rendering Utilities ====================

    def _verdict_badge(self, verdict: str) -> str:
        """Return a visual badge for publication verdict."""
        badges = {
            "ready": "✅ **READY** for publication",
            "minor-revisions": "⚠️ **MINOR REVISIONS** needed",
            "major-revisions": "🔧 **MAJOR REVISIONS** required",
            "reject": "❌ **REJECTED** - fundamental issues",
            "not-reviewed": "⏳ **NOT REVIEWED**",
        }
        return badges.get(verdict, verdict)

    def _render_review_scores(self, peer_review: dict[str, Any]) -> str:
        """Render dual review analysis section."""
        lines = []

        # Gemini review
        if gemini := peer_review.get("gemini_review"):
            lines.append("**Gemini 2.5 Pro Review:**")
            rigor = gemini.get("rigor", 0)
            soundness = gemini.get("soundness", 0)
            consistency = gemini.get("consistency", 0)
            verdict = gemini.get("verdict", "not-reviewed")
            lines.extend((
                f"- Rigor: {rigor}/10 | Soundness: {soundness}/10 | Consistency: {consistency}/10",
                f"- Verdict: {self._verdict_badge(verdict)}",
            ))
            if issues := gemini.get("issues_identified"):
                critical = [i for i in issues if i.get("severity") == "critical"]
                major = [i for i in issues if i.get("severity") == "major"]
                if critical:
                    lines.append(f"- 🚫 {len(critical)} CRITICAL issues")
                if major:
                    lines.append(f"- ⚠️ {len(major)} MAJOR issues")
            lines.append("")

        # Codex review
        if codex := peer_review.get("codex_review"):
            lines.append("**Codex Review:**")
            rigor = codex.get("rigor", 0)
            soundness = codex.get("soundness", 0)
            consistency = codex.get("consistency", 0)
            verdict = codex.get("verdict", "not-reviewed")
            lines.extend((
                f"- Rigor: {rigor}/10 | Soundness: {soundness}/10 | Consistency: {consistency}/10",
                f"- Verdict: {self._verdict_badge(verdict)}",
            ))
            if issues := codex.get("issues_identified"):
                critical = [i for i in issues if i.get("severity") == "critical"]
                major = [i for i in issues if i.get("severity") == "major"]
                if critical:
                    lines.append(f"- 🚫 {len(critical)} CRITICAL issues")
                if major:
                    lines.append(f"- ⚠️ {len(major)} MAJOR issues")
            lines.append("")

        # Aggregate and consensus
        if aggregate := peer_review.get("aggregate_score"):
            lines.append("**Aggregate Score:**")
            rigor = aggregate.get("rigor", 0)
            soundness = aggregate.get("soundness", 0)
            consistency = aggregate.get("consistency", 0)
            lines.extend((
                f"- Rigor: {rigor:.1f}/10 | Soundness: {soundness:.1f}/10 | Consistency: {consistency:.1f}/10",
                "",
            ))

        if final_verdict := peer_review.get("final_verdict"):
            lines.extend((f"**Final Verdict:** {self._verdict_badge(final_verdict)}", ""))

        # Consensus vs discrepancies
        if consensus := peer_review.get("consensus_issues"):
            if consensus:
                lines.append(
                    f"**Consensus Issues:** {len(consensus)} issues identified by both reviewers"
                )
        if discrepancies := peer_review.get("discrepancies"):
            if discrepancies:
                lines.append(
                    f"⚠️ **Discrepancies:** {len(discrepancies)} contradictory findings (requires manual verification)"
                )

        # Blocking issues
        if blocking := peer_review.get("blocking_issues"):
            if blocking:
                lines.extend((
                    "",
                    f"**🚫 Blocking Issues:** {len(blocking)} critical issues prevent publication",
                ))
                for issue in blocking[:3]:  # Show first 3
                    title = issue.get("title", "Untitled")
                    severity = issue.get("severity", "unknown")
                    lines.append(f"- [{severity.upper()}] {title}")
                if len(blocking) > 3:
                    lines.append(f"- ... and {len(blocking) - 3} more")

        return "\n".join(lines)

    def _render_development_status(self, dev_status: dict[str, Any]) -> str:
        """Render development status section."""
        lines = []

        stage = dev_status.get("stage", "unknown")
        completeness = dev_status.get("completeness", 0)

        stage_emoji = {
            "sketch": "📝",
            "partial": "🚧",
            "complete": "✅",
            "verified": "🔒",
            "published": "📚",
        }
        emoji = stage_emoji.get(stage, "❓")

        lines.append(f"**Development:** {emoji} {stage.upper()} ({completeness}% complete)")

        if verification := dev_status.get("verification_status"):
            checks = []
            if verification.get("logic_verified"):
                checks.append("✓ Logic")
            if verification.get("computation_verified"):
                checks.append("✓ Computation")
            if verification.get("framework_consistent"):
                checks.append("✓ Framework")
            if checks:
                lines.append(f"- Verification: {' | '.join(checks)}")

        if quality := dev_status.get("quality_metrics"):
            if rigor_level := quality.get("rigor_level"):
                lines.append(f"- Quality: Rigor {rigor_level}/10")

        return "\n".join(lines)

    def _render_sketch_linkage(self, sketch_link: dict[str, Any]) -> str:
        """Render sketch-to-proof linkage section."""
        lines = []

        if source := sketch_link.get("source_sketch"):
            file_path = source.get("file_path", "")
            label = source.get("label", "")
            agent = source.get("agent", "unknown")
            lines.append(f"**Source Sketch:** `{file_path}` (by {agent})")
            if label:
                lines.append(f"- Label: `{label}`")

        if coverage := sketch_link.get("sketch_coverage"):
            pct = coverage.get("coverage_percentage", 0)
            lines.append(f"- Coverage: {pct}% of sketch items addressed")
            if uncovered := coverage.get("uncovered_items"):
                if uncovered:
                    lines.append(f"- ⚠️ {len(uncovered)} sketch items not yet covered")

        if history := sketch_link.get("expansion_history"):
            if history:
                lines.append(f"- Expansion history: {len(history)} revisions")

        return "\n".join(lines)


# ==================== CLI ====================


def validate_document(doc: dict[str, Any], schema_path: Path) -> None:
    """Validate document against JSON schema."""
    try:
        import jsonschema
    except ImportError:
        print(
            "Warning: jsonschema not installed. Install with: pip install jsonschema",
            file=sys.stderr,
        )
        return

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    try:
        jsonschema.validate(doc, schema)
        print("✓ Document is valid according to schema", file=sys.stderr)
    except jsonschema.ValidationError as e:
        print(f"✗ Validation error: {e.message}", file=sys.stderr)
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render mathematical document JSON to human-readable markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render to stdout
  python render_math_json.py example_document.json

  # Render to file
  python render_math_json.py example_document.json -o output.md

  # With validation
  python render_math_json.py example_document.json --validate

  # Filter by type
  python render_math_json.py example_document.json --filter-type theorem

  # Jupyter Book format
  python render_math_json.py example_document.json --jupyter-book
        """,
    )

    parser.add_argument("input_json", type=Path, help="Input JSON file path")

    parser.add_argument("-o", "--output", type=Path, help="Output markdown file (default: stdout)")

    parser.add_argument(
        "--validate", action="store_true", help="Validate JSON against schema before rendering"
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("math_schema.json"),
        help="Path to JSON schema file (default: math_schema.json)",
    )

    parser.add_argument(
        "--jupyter-book",
        action="store_true",
        help="Output Jupyter Book format (future feature)",
    )

    parser.add_argument(
        "--no-graph", action="store_true", help="Exclude dependency graph from output"
    )

    parser.add_argument(
        "--graph-format",
        choices=["mermaid", "table"],
        default="mermaid",
        help="Dependency graph format (default: mermaid)",
    )

    parser.add_argument(
        "--filter-type",
        choices=[
            "definition",
            "axiom",
            "theorem",
            "lemma",
            "proposition",
            "corollary",
            "proof",
            "algorithm",
            "remark",
            "observation",
            "conjecture",
            "example",
            "property",
        ],
        help="Render only specific directive type",
    )

    parser.add_argument("--filter-label", help="Render only specific label")

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_json.exists():
        print(f"Error: Input file '{args.input_json}' not found", file=sys.stderr)
        sys.exit(1)

    # Load JSON
    try:
        with open(args.input_json, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)

    # Optional validation
    if args.validate:
        if not args.schema.exists():
            print(f"Error: Schema file '{args.schema}' not found", file=sys.stderr)
            sys.exit(1)
        validate_document(doc, args.schema)

    # Configure rendering options
    options = RenderOptions(
        jupyter_book=args.jupyter_book,
        include_graph=not args.no_graph,
        graph_format=args.graph_format,
        filter_type=args.filter_type,
        filter_label=args.filter_label,
    )

    # Render
    renderer = MathDocumentRenderer(doc, options)
    markdown = renderer.render()

    # Output
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"✓ Rendered to {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
