"""
Relationship Builder - Construct and Validate Relationship Objects.

This module constructs Relationship objects from discovered dependencies,
performing relationship type inference, attribute extraction, and validation
against Pydantic schemas.

Version: 1.0.0
"""

from __future__ import annotations

import re

from fragile.proofs import Relationship, RelationshipAttribute, RelationType


class RelationshipBuilder:
    """
    Constructs Relationship objects with proper typing and validation.

    Infers relationship types from context using keyword matching and
    mathematical pattern recognition.
    """

    def __init__(self):
        """Initialize relationship builder with pattern matchers."""
        self._build_pattern_matchers()

    def _build_pattern_matchers(self) -> None:
        """Build regex patterns for relationship type inference."""
        # Relationship type keywords
        self.type_patterns = {
            RelationType.EQUIVALENCE: [
                r"\bequivalent\s+to\b",
                r"\bif\s+and\s+only\s+if\b",
                r"\biff\b",
                r"≡",
                r"⟺",
                r"\bsame\s+as\b",
                r"\bidentical\s+to\b",
            ],
            RelationType.EMBEDDING: [
                r"\bembeds?\s+(into|in)\b",
                r"\binjection\b",
                r"\binjective\b",
                r"↪",
                r"\bstructure[- ]preserving\b",
                r"\bisomorphic\s+to\s+subspace\b",
            ],
            RelationType.APPROXIMATION: [
                r"\bapproximate[sd]?\s+(to|by)\b",
                r"≈",
                r"\berror\s+bound\b",
                r"\bconverge[sd]?\s+to\b",
                r"O\(",
                r"\\mathcal\{O\}",
                r"\basymptotic(ally)?\b",
            ],
            RelationType.REDUCTION: [
                r"\breduce[sd]?\s+to\b",
                r"→",
                r"\bsimplif(y|ies)\s+to\b",
                r"\bcollapse[sd]?\s+to\b",
            ],
            RelationType.EXTENSION: [
                r"\bextend[sd]?\b",
                r"\bgeneralize[sd]?\b",
                r"\bbroaden[sed]?\b",
            ],
            RelationType.GENERALIZATION: [
                r"\bgeneralization\s+of\b",
                r"\bmore\s+general\s+than\b",
                r"\bbroader\s+than\b",
                r"\bsubsumes\b",
            ],
            RelationType.SPECIALIZATION: [
                r"\bspecial\s+case\s+of\b",
                r"\brestrict(s|ed|ion)\s+to\b",
                r"\bparticular\s+case\b",
                r"\bnarrow[sed]?\s+down\b",
            ],
        }

        # Compile patterns
        self.compiled_patterns: dict[RelationType, list[re.Pattern]] = {}
        for rel_type, patterns in self.type_patterns.items():
            self.compiled_patterns[rel_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def build_relationship(
        self,
        source_label: str,
        target_label: str,
        established_by: str,
        context: str,
        chapter: str | None = None,
        document: str | None = None,
    ) -> Relationship:
        """
        Build a Relationship object from dependency information.

        Args:
            source_label: Source object label (must start with obj-)
            target_label: Target object label (must start with obj-)
            established_by: Theorem/lemma/proposition label
            context: Text describing the relationship
            chapter: Chapter identifier
            document: Document identifier

        Returns:
            Validated Relationship object
        """
        # Infer relationship type from context
        rel_type, bidirectional = self._infer_relationship_type(context)

        # Extract relationship attributes (e.g., error bounds)
        attributes = self._extract_attributes(context)

        # Generate relationship label
        label = self._generate_label(source_label, target_label, rel_type)

        # Create relationship
        return Relationship(
            label=label,
            relationship_type=rel_type,
            bidirectional=bidirectional,
            source_object=source_label,
            target_object=target_label,
            established_by=established_by,
            expression=context[:500],  # Truncate to reasonable length
            attributes=attributes,
            tags=self._extract_tags(context),
            chapter=chapter,
            document=document,
        )

    def build_simple_uses_relationship(
        self,
        source_label: str,
        target_label: str,
        established_by: str,
        chapter: str | None = None,
        document: str | None = None,
    ) -> Relationship:
        """
        Build a simple "uses" relationship for explicit cross-refs.

        Args:
            source_label: Source theorem/lemma label
            target_label: Target object/theorem label
            established_by: Same as source_label
            chapter: Chapter identifier
            document: Document identifier

        Returns:
            Relationship with OTHER type
        """
        # Convert theorem labels to object labels if needed
        source_obj = self._ensure_object_label(source_label)
        target_obj = self._ensure_object_label(target_label)

        label = f"rel-{source_obj[4:]}-{target_obj[4:]}-other"

        return Relationship(
            label=label,
            relationship_type=RelationType.OTHER,
            bidirectional=False,
            source_object=source_obj,
            target_object=target_obj,
            established_by=established_by,
            expression=f"{source_label} uses {target_label}",
            attributes=[],
            tags=["explicit-ref"],
            chapter=chapter,
            document=document,
        )

    def _infer_relationship_type(self, context: str) -> tuple[RelationType, bool]:
        """
        Infer relationship type from context.

        Args:
            context: Text describing the relationship

        Returns:
            Tuple of (RelationType, bidirectional)
        """
        # Check each pattern
        for rel_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(context):
                    # Determine bidirectionality
                    bidirectional = rel_type == RelationType.EQUIVALENCE
                    return rel_type, bidirectional

        # Default to OTHER
        return RelationType.OTHER, False

    def _extract_attributes(self, context: str) -> list[RelationshipAttribute]:
        """
        Extract relationship attributes from context.

        Looks for:
        - Error bounds: O(...), $O(...)$, order of
        - Rates: rate, convergence
        - Constants: Lipschitz constant, bound

        Args:
            context: Text describing the relationship

        Returns:
            List of RelationshipAttribute objects
        """
        attributes = []

        # Pattern: O(...) or $O(...)$
        big_o_pattern = re.compile(r"[O$]\\?\{?O\\?\}?\s*\(([^)]+)\)\$?")
        for match in big_o_pattern.finditer(context):
            attributes.append(
                RelationshipAttribute(
                    label="error-rate",
                    expression=f"O({match.group(1)})",
                    description="Approximation error rate",
                )
            )

        # Pattern: "error bound" followed by expression
        error_pattern = re.compile(r"error\s+bound[:\s]+(\$?[^\$]+\$?)", re.IGNORECASE)
        for match in error_pattern.finditer(context):
            expr = match.group(1).strip("$")
            if expr not in [attr.expression for attr in attributes]:
                attributes.append(
                    RelationshipAttribute(
                        label="error-bound", expression=expr, description="Error bound"
                    )
                )

        # Pattern: "convergence rate" followed by expression
        rate_pattern = re.compile(r"(convergence\s+)?rate[:\s]+(\$?[^\$]+\$?)", re.IGNORECASE)
        for match in rate_pattern.finditer(context):
            expr = match.group(2).strip("$")
            if expr not in [attr.expression for attr in attributes]:
                attributes.append(
                    RelationshipAttribute(
                        label="convergence-rate", expression=expr, description="Convergence rate"
                    )
                )

        return attributes

    def _generate_label(self, source_label: str, target_label: str, rel_type: RelationType) -> str:
        """
        Generate relationship label following naming convention.

        Format: rel-{source}-{target}-{type}

        Args:
            source_label: Source object label
            target_label: Target object label
            rel_type: Relationship type

        Returns:
            Valid relationship label
        """
        # Extract object IDs (remove obj- prefix)
        source_id = source_label.replace("obj-", "")
        target_id = target_label.replace("obj-", "")

        # Convert RelationType to lowercase with hyphens
        type_str = rel_type.value.lower().replace("_", "-")

        return f"rel-{source_id}-{target_id}-{type_str}"

    def _extract_tags(self, context: str) -> list[str]:
        """
        Extract tags from context for categorization.

        Args:
            context: Text describing the relationship

        Returns:
            List of tags
        """
        tags = []

        # Common mathematical tags
        tag_keywords = {
            "mean-field": ["mean field", "mean-field", "mckean"],
            "discrete-continuous": ["discrete", "continuous", "discretization"],
            "convergence": ["converge", "convergence", "limit"],
            "approximation": ["approximate", "approximation"],
            "stability": ["stable", "stability"],
            "contraction": ["contractive", "contraction"],
        }

        context_lower = context.lower()
        for tag, keywords in tag_keywords.items():
            if any(kw in context_lower for kw in keywords):
                tags.append(tag)

        return tags

    def _ensure_object_label(self, label: str) -> str:
        """
        Ensure label starts with obj- prefix.

        Converts theorem/lemma labels to pseudo-object labels for relationships.

        Args:
            label: Original label

        Returns:
            Label with obj- prefix
        """
        if label.startswith("obj-"):
            return label
        if label.startswith(("thm-", "lem-", "prop-")):
            # Convert to object label (pseudo-object representing the theorem)
            return "obj-" + label[4:]
        if label.startswith("def-"):
            return "obj-" + label[4:]
        # Already in correct format or unknown type
        return f"obj-{label}" if not label.startswith("obj-") else label


class RelationshipValidator:
    """
    Validates relationships for consistency and correctness.

    Checks:
    - Label format correctness
    - Source/target existence in registry
    - Established_by existence
    - Bidirectional consistency
    """

    def __init__(self, registry):
        """
        Initialize validator with registry.

        Args:
            registry: MathematicalRegistry to check against
        """
        self.registry = registry

    def validate_relationship(self, rel: Relationship) -> tuple[bool, list[str]]:
        """
        Validate a relationship.

        Args:
            rel: Relationship to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check source object exists
        if not self.registry.get(rel.source_object):
            errors.append(f"Source object not found: {rel.source_object}")

        # Check target object exists
        if not self.registry.get(rel.target_object):
            errors.append(f"Target object not found: {rel.target_object}")

        # Check established_by exists
        if not self.registry.get(rel.established_by):
            errors.append(f"Establishing theorem not found: {rel.established_by}")

        # Check bidirectional consistency
        if rel.bidirectional and rel.relationship_type != RelationType.EQUIVALENCE:
            errors.append(f"Relationship type {rel.relationship_type} should not be bidirectional")

        return len(errors) == 0, errors
