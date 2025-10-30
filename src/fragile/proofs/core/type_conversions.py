"""
Type Conversion Utilities: Enriched Types → Math Types.

This module provides conversion functions to transform document-extracted types
(enriched_types.py) into framework computational types (math_types.py).

Use Cases:
- Scale testing: Extract parameters from papers, convert to framework types, test theorems
- Document processing: Parse papers, then create computational entities
- Round-trip workflows: Extraction → Enrichment → Conversion → Computation

Primary Conversions:
- ParameterBox → Parameter (direct mapping)
- EquationBox, RemarkBox → semantic linking (updates theorem/object metadata)

Maps to Lean:
    namespace TypeConversions
      def parameterBoxToParameter : ParameterBox → Parameter := ...
      def convertParameterBoxes : List ParameterBox → Result (List Parameter) := ...
    end TypeConversions

Version: 1.0.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fragile.proofs.core.enriched_types import EquationBox, ParameterBox, RemarkBox
from fragile.proofs.core.math_types import (
    MathematicalObject,
    Parameter,
    TheoremBox,
)


# =============================================================================
# REPORTING TYPES
# =============================================================================


class ConversionReport(BaseModel):
    """
    Report on batch conversion results.

    Tracks success/failure statistics and provides detailed diagnostic information.

    Maps to Lean:
        structure ConversionReport where
          total_items : Nat
          successful : Nat
          failed : Nat
          warnings : List String
          errors : List String
    """

    model_config = ConfigDict(frozen=True)

    total_items: int = Field(..., ge=0, description="Total items attempted")
    successful: int = Field(..., ge=0, description="Successfully converted items")
    failed: int = Field(..., ge=0, description="Failed conversions")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues")
    errors: list[str] = Field(default_factory=list, description="Fatal errors")

    def summary(self) -> str:
        """Get human-readable summary."""
        success_rate = (self.successful / self.total_items * 100) if self.total_items > 0 else 0
        lines = [
            f"Conversion Report: {self.successful}/{self.total_items} successful ({success_rate:.1f}%)",
            f"  Failed: {self.failed}",
        ]

        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:3]:  # Show first 3
                lines.append(f"    - {warning}")
            if len(self.warnings) > 3:
                lines.append(f"    ... and {len(self.warnings) - 3} more")

        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for error in self.errors[:3]:  # Show first 3
                lines.append(f"    - {error}")
            if len(self.errors) > 3:
                lines.append(f"    ... and {len(self.errors) - 3} more")

        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def extract_chapter_document(
    source: SourceLocation | None,  # type: ignore
) -> tuple[str | None, str | None]:
    """
    Extract chapter and document identifiers from SourceLocation.

    Parses file paths like:
    - "docs/source/1_euclidean_gas/04_convergence/theorem.md" → ("1_euclidean_gas", "04_convergence")
    - "docs/source/2_geometric_gas/11_geometric_gas.md" → ("2_geometric_gas", "11_geometric_gas")
    - "standalone.md" → (None, None)

    Args:
        source: Source location with file_path attribute

    Returns:
        Tuple of (chapter, document) or (None, None) if cannot parse

    Maps to Lean:
        def extract_chapter_document (source : Option SourceLocation) : (Option String × Option String) :=
          match source with
          | none => (none, none)
          | some s =>
              let parts := s.file_path.split '/'
              if parts.length >= 2 then
                (some parts[0], some parts[1])
              else
                (none, none)
    """
    if source is None:
        return (None, None)

    # Parse file path (use file_path attribute, not file)
    parts = source.file_path.split("/")

    # Look for chapter/document pattern in path
    # Typical structure: docs/source/1_euclidean_gas/04_convergence/file.md
    # We want to extract: 1_euclidean_gas and 04_convergence
    chapter = None
    document = None

    for i, part in enumerate(parts):
        # Look for chapter pattern (starts with digit_)
        if part and part[0].isdigit() and "_" in part:
            chapter = part
            # Next part might be document
            if i + 1 < len(parts):
                next_part = parts[i + 1]
                # Document might be a file or subdirectory
                if next_part.endswith(".md"):
                    document = next_part.replace(".md", "")
                elif next_part and next_part[0].isdigit() and "_" in next_part:
                    document = next_part
            break

    return (chapter, document)


def merge_constraints(constraints: list[str]) -> str | None:
    """
    Merge list of constraints into single string.

    Args:
        constraints: List of constraint strings (e.g., ["γ > 0", "γ < 1"])

    Returns:
        Comma-separated string or None if empty

    Examples:
        >>> merge_constraints(["γ > 0", "γ < 1"])
        "γ > 0, γ < 1"

        >>> merge_constraints([])
        None

    Maps to Lean:
        def merge_constraints (cs : List String) : Option String :=
          if cs.isEmpty then none
          else some (String.intercalate ", " cs)
    """
    if not constraints:
        return None
    return ", ".join(constraints)


# =============================================================================
# CORE CONVERSIONS
# =============================================================================


def parameter_box_to_parameter(param_box: ParameterBox) -> Parameter:
    """
    Convert ParameterBox (document-extracted) to Parameter (framework).

    Transformation:
    - label: Preserved (already in param- format)
    - name: Use meaning field (more descriptive than symbol)
    - symbol: Preserved
    - parameter_type: Preserved (already ParameterType enum)
    - constraints: Merge list into comma-separated string
    - default_value: Preserved
    - chapter/document: Extract from source location

    Args:
        param_box: ParameterBox from document extraction

    Returns:
        Parameter suitable for framework use

    Examples:
        >>> param_box = ParameterBox(
        ...     label="param-gamma",
        ...     symbol="γ",
        ...     latex="\\gamma",
        ...     domain=ParameterType.REAL,
        ...     meaning="friction coefficient",
        ...     scope=ParameterScope.GLOBAL,
        ...     constraints=["γ > 0"],
        ...     source=SourceLocation(file="1_euclidean_gas/04_convergence/file.md"),
        ... )
        >>> param = parameter_box_to_parameter(param_box)
        >>> param.name
        "friction coefficient"
        >>> param.chapter
        "1_euclidean_gas"
        >>> param.document
        "04_convergence"

    Maps to Lean:
        def parameterBoxToParameter (pb : ParameterBox) : Parameter :=
          let (chapter, document) := extract_chapter_document pb.source
          { label := pb.label,
            name := pb.meaning,
            symbol := pb.symbol,
            parameter_type := pb.domain,
            constraints := merge_constraints pb.constraints,
            default_value := pb.default_value,
            chapter := chapter,
            document := document }
    """
    # Extract chapter/document from source
    chapter, document = extract_chapter_document(param_box.source)

    # Merge constraints
    constraints_str = merge_constraints(param_box.constraints)

    return Parameter(
        label=param_box.label,
        name=param_box.meaning,  # Use meaning (more descriptive than symbol)
        symbol=param_box.symbol,
        parameter_type=param_box.domain,  # ParameterType enum already shared
        constraints=constraints_str,
        default_value=param_box.default_value,
        chapter=chapter,
        document=document,
    )


def convert_parameter_boxes(
    param_boxes: dict[str, ParameterBox],
) -> tuple[dict[str, Parameter], ConversionReport]:
    """
    Batch convert ParameterBox collection to Parameter collection.

    Continues on individual failures, collecting all errors for reporting.

    Args:
        param_boxes: Dictionary mapping label → ParameterBox

    Returns:
        Tuple of (converted_parameters, conversion_report)

    Examples:
        >>> param_boxes = {
        ...     "param-gamma": ParameterBox(...),
        ...     "param-N": ParameterBox(...),
        ... }
        >>> params, report = convert_parameter_boxes(param_boxes)
        >>> print(report.summary())
        Conversion Report: 2/2 successful (100.0%)
          Failed: 0

    Maps to Lean:
        def convertParameterBoxes (pbs : HashMap String ParameterBox) :
          (HashMap String Parameter × ConversionReport) :=
          let results := pbs.foldl (fun acc (label, pb) =>
            match parameterBoxToParameter pb with
            | Except.ok p => { acc with successful := acc.successful.insert label p }
            | Except.error e => { acc with failed := acc.failed ++ [e] })
            { successful := HashMap.empty, failed := [] }
          (results.successful, { total := pbs.size, ... })
    """
    converted: dict[str, Parameter] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for label, param_box in param_boxes.items():
        try:
            param = parameter_box_to_parameter(param_box)
            converted[label] = param

            # Add warnings for missing data
            if param_box.source is None:
                warnings.append(f"{label}: No source location available")

        except Exception as e:
            errors.append(f"{label}: {e!s}")

    report = ConversionReport(
        total_items=len(param_boxes),
        successful=len(converted),
        failed=len(errors),
        warnings=warnings,
        errors=errors,
    )

    return converted, report


# =============================================================================
# SEMANTIC LINKING (for types without direct conversion)
# =============================================================================


def link_equation_to_theorems(
    eq_box: EquationBox, theorems: dict[str, TheoremBox]
) -> dict[str, TheoremBox]:
    """
    Update theorems with equation references.

    Since EquationBox has no direct math_types equivalent, we instead
    update theorem metadata to reference the equation.

    Args:
        eq_box: Equation from document extraction
        theorems: Existing theorem collection

    Returns:
        Updated theorems with equation references in uses_definitions

    Examples:
        >>> eq_box = EquationBox(
        ...     label="eq-langevin",
        ...     appears_in_theorems=["thm-convergence"],
        ...     ...
        ... )
        >>> updated = link_equation_to_theorems(eq_box, theorems)
        >>> "eq-langevin" in updated["thm-convergence"].uses_definitions
        True

    Maps to Lean:
        def linkEquationToTheorems (eq : EquationBox) (thms : HashMap String TheoremBox) :
          HashMap String TheoremBox :=
          eq.appears_in_theorems.foldl (fun acc thm_label =>
            match acc.find? thm_label with
            | none => acc
            | some thm =>
                let updated := { thm with uses_definitions := thm.uses_definitions ++ [eq.label] }
                acc.insert thm_label updated)
            thms
    """
    updated_theorems = dict(theorems)  # Copy

    for thm_label in eq_box.appears_in_theorems:
        if thm_label in updated_theorems:
            thm = updated_theorems[thm_label]

            # Add equation reference if not already present
            if eq_box.label not in thm.uses_definitions:
                updated_theorems[thm_label] = thm.model_copy(
                    update={"uses_definitions": [*thm.uses_definitions, eq_box.label]}
                )

    return updated_theorems


def link_remark_to_entities(
    remark_box: RemarkBox,
    theorems: dict[str, TheoremBox],
    objects: dict[str, MathematicalObject],
) -> tuple[dict[str, TheoremBox], dict[str, MathematicalObject]]:
    """
    Update objects with remark references.

    Since RemarkBox has no direct math_types equivalent, we add tags to
    related MathematicalObjects to preserve the semantic link.

    Note: TheoremBox does not have a tags field, so remarks are not
    linked to theorems. Remarks can be separately stored and referenced.

    Args:
        remark_box: Remark from document extraction
        theorems: Existing theorem collection (returned unchanged)
        objects: Existing object collection

    Returns:
        Tuple of (unchanged_theorems, updated_objects)

    Examples:
        >>> remark_box = RemarkBox(
        ...     label="remark-particle-interpretation",
        ...     relates_to=["obj-walker"],
        ...     ...
        ... )
        >>> thms, objs = link_remark_to_entities(remark_box, theorems, objects)
        >>> "remark-particle-interpretation" in objs["obj-walker"].tags
        True

    Maps to Lean:
        def linkRemarkToEntities (r : RemarkBox)
          (thms : HashMap String TheoremBox)
          (objs : HashMap String MathematicalObject) :
          (HashMap String TheoremBox × HashMap String MathematicalObject) :=
          -- Add remark label as tag to related objects only
          (thms, objs.mapValues (fun obj =>
            if obj.label ∈ r.relates_to then
              { obj with tags := obj.tags ++ [r.label] }
            else obj))
    """
    updated_theorems = dict(theorems)  # Return unchanged
    updated_objects = dict(objects)

    # Link to objects only (TheoremBox doesn't have tags field)
    for entity_label in remark_box.relates_to + remark_box.provides_intuition_for:
        if entity_label in updated_objects:
            obj = updated_objects[entity_label]
            if remark_box.label not in obj.tags:
                updated_objects[entity_label] = obj.model_copy(
                    update={"tags": [*obj.tags, remark_box.label]}
                )
        # Skip theorem labels - TheoremBox doesn't have tags field

    return updated_theorems, updated_objects


# =============================================================================
# BATCH LINKING
# =============================================================================


def link_all_enriched_types(
    param_boxes: dict[str, ParameterBox],
    equation_boxes: dict[str, EquationBox],
    remark_boxes: dict[str, RemarkBox],
    theorems: dict[str, TheoremBox],
    objects: dict[str, MathematicalObject],
) -> tuple[
    dict[str, Parameter],
    dict[str, TheoremBox],
    dict[str, MathematicalObject],
    ConversionReport,
]:
    """
    Comprehensive conversion and linking of all enriched types.

    This is the main entry point for full document processing:
    1. Convert ParameterBox → Parameter
    2. Link EquationBox to theorems
    3. Link RemarkBox to theorems/objects

    Args:
        param_boxes: Extracted parameters
        equation_boxes: Extracted equations
        remark_boxes: Extracted remarks
        theorems: Existing theorems (may be updated)
        objects: Existing objects (may be updated)

    Returns:
        Tuple of (parameters, updated_theorems, updated_objects, report)

    Examples:
        >>> params, thms, objs, report = link_all_enriched_types(
        ...     param_boxes=extracted_params,
        ...     equation_boxes=extracted_equations,
        ...     remark_boxes=extracted_remarks,
        ...     theorems=existing_theorems,
        ...     objects=existing_objects,
        ... )
        >>> print(report.summary())
        Conversion Report: 10/10 successful (100.0%)
    """
    # Convert parameters
    parameters, report = convert_parameter_boxes(param_boxes)

    # Start with copies of input theorems/objects
    updated_theorems = dict(theorems)
    updated_objects = dict(objects)

    # Link equations
    for eq_box in equation_boxes.values():
        updated_theorems = link_equation_to_theorems(eq_box, updated_theorems)

    # Link remarks
    for remark_box in remark_boxes.values():
        updated_theorems, updated_objects = link_remark_to_entities(
            remark_box, updated_theorems, updated_objects
        )

    return parameters, updated_theorems, updated_objects, report


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Reporting
    "ConversionReport",
    "convert_parameter_boxes",
    # Helpers
    "extract_chapter_document",
    "link_all_enriched_types",
    # Semantic linking
    "link_equation_to_theorems",
    "link_remark_to_entities",
    "merge_constraints",
    # Core conversions
    "parameter_box_to_parameter",
]
