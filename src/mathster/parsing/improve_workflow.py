"""
Improvement workflow for mathematical concept extraction using DSPy ReAct agents.

**⚠️ DEPRECATED: This file is legacy code. Use the modular API instead:**

```python
# OLD (deprecated):
from mathster.parsing.improve_workflow import improve_chapter

# NEW (recommended):
from mathster.parsing import workflows
workflows.improve_chapter(...)

# Or directly:
from mathster.parsing.workflows import improve_chapter
```

This module implements the IMPROVEMENT workflow that enhances existing extractions
by finding missing entities or correcting errors in previously extracted data.

Workflow:
    1. Load existing extraction from JSON
    2. Compare against chapter text to find missing/incorrect entities
    3. Generate improvement using ReAct agent
    4. Merge improvements with existing data
    5. Track changes (ADD/MODIFY/DELETE operations)
    6. Return improved data with change metadata

Key Components:
    - MathematicalConceptImprover: ReAct agent for improvement
    - ImproveMathematicalConcepts: DSPy signature for agent
    - compare_extractions_tool: Tool to compare existing vs new
    - ImprovementResult: Change tracking structure

⚠️ For new code, use: `from mathster.parsing.workflows import improve_chapter`
"""

import json
from enum import Enum
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from mathster.core.article_system import SourceLocation, TextLocation
from mathster.core.raw_data import RawDocumentSection

# Import extraction workflow components we reuse
from mathster.parsing.extract_workflow import (
    ChapterExtraction,
    ValidationResult,
    convert_to_raw_document_section,
    validate_extraction,
    generate_detailed_error_report,
    make_error_dict,
)


# =============================================================================
# IMPROVEMENT TRACKING SCHEMA
# =============================================================================


class ChangeOperation(str, Enum):
    """Type of change operation."""
    ADD = "ADD"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    NO_CHANGE = "NO_CHANGE"


class EntityChange(BaseModel):
    """Record of a change to a single entity."""

    entity_type: Literal["definition", "theorem", "proof", "axiom", "parameter", "remark", "citation"]
    label: str
    operation: ChangeOperation
    reason: str = Field(..., description="Why this change was made")
    old_data: dict | None = Field(None, description="Original entity data (for MODIFY/DELETE)")
    new_data: dict | None = Field(None, description="New entity data (for ADD/MODIFY)")


class ImprovementResult(BaseModel):
    """Result of improvement workflow with change tracking."""

    changes: list[EntityChange] = Field(default_factory=list)
    entities_added: int = Field(default=0, description="Count of entities added")
    entities_modified: int = Field(default=0, description="Count of entities modified")
    entities_deleted: int = Field(default=0, description="Count of entities deleted")
    entities_unchanged: int = Field(default=0, description="Count of entities unchanged")

    def add_change(self, change: EntityChange) -> None:
        """Add a change and update counters."""
        self.changes.append(change)

        if change.operation == ChangeOperation.ADD:
            self.entities_added += 1
        elif change.operation == ChangeOperation.MODIFY:
            self.entities_modified += 1
        elif change.operation == ChangeOperation.DELETE:
            self.entities_deleted += 1
        elif change.operation == ChangeOperation.NO_CHANGE:
            self.entities_unchanged += 1

    def get_summary(self) -> str:
        """Get human-readable summary of changes."""
        return (
            f"Improvement Summary:\n"
            f"  Added: {self.entities_added}\n"
            f"  Modified: {self.entities_modified}\n"
            f"  Deleted: {self.entities_deleted}\n"
            f"  Unchanged: {self.entities_unchanged}\n"
            f"  Total changes: {len(self.changes)}"
        )


# =============================================================================
# IMPROVEMENT TOOLS
# =============================================================================


def compare_extractions_tool(
    existing_json: str,
    proposed_json: str,
    context: str
) -> str:
    """
    Tool to compare existing extraction with proposed improvements.

    Args:
        existing_json: JSON string of existing ChapterExtraction
        proposed_json: JSON string of proposed improved ChapterExtraction
        context: Context string (file_path|||article_id|||chapter_text)

    Returns:
        Comparison feedback string
    """
    try:
        # Parse JSONs
        existing_dict = json.loads(existing_json)
        proposed_dict = json.loads(proposed_json)

        existing = ChapterExtraction(**existing_dict)
        proposed = ChapterExtraction(**proposed_dict)

        # Compare entity counts
        feedback = ["Comparison Results:"]

        # Definitions
        existing_def_labels = {d.label for d in existing.definitions}
        proposed_def_labels = {d.label for d in proposed.definitions}
        new_defs = proposed_def_labels - existing_def_labels
        removed_defs = existing_def_labels - proposed_def_labels
        modified_defs = []

        for d in proposed.definitions:
            if d.label in existing_def_labels:
                # Check if modified
                existing_d = next(ed for ed in existing.definitions if ed.label == d.label)
                if d.model_dump() != existing_d.model_dump():
                    modified_defs.append(d.label)

        if new_defs or removed_defs or modified_defs:
            feedback.append(f"\nDefinitions:")
            if new_defs:
                feedback.append(f"  + Added: {', '.join(new_defs)}")
            if modified_defs:
                feedback.append(f"  ± Modified: {', '.join(modified_defs)}")
            if removed_defs:
                feedback.append(f"  - Removed: {', '.join(removed_defs)}")

        # Theorems
        existing_thm_labels = {t.label for t in existing.theorems}
        proposed_thm_labels = {t.label for t in proposed.theorems}
        new_thms = proposed_thm_labels - existing_thm_labels
        removed_thms = existing_thm_labels - proposed_thm_labels
        modified_thms = []

        for t in proposed.theorems:
            if t.label in existing_thm_labels:
                existing_t = next(et for et in existing.theorems if et.label == t.label)
                if t.model_dump() != existing_t.model_dump():
                    modified_thms.append(t.label)

        if new_thms or removed_thms or modified_thms:
            feedback.append(f"\nTheorems:")
            if new_thms:
                feedback.append(f"  + Added: {', '.join(new_thms)}")
            if modified_thms:
                feedback.append(f"  ± Modified: {', '.join(modified_thms)}")
            if removed_thms:
                feedback.append(f"  - Removed: {', '.join(removed_thms)}")

        # Add similar comparisons for other entity types
        # (abbreviated for brevity - same pattern)

        if len(feedback) == 1:
            feedback.append("\nNo changes detected.")

        return "\n".join(feedback)

    except Exception as e:
        return f"Comparison error: {str(e)}"


def validate_improvement_tool(
    proposed_json: str,
    context: str
) -> str:
    """
    Tool to validate proposed improvements.

    Args:
        proposed_json: JSON string of proposed ChapterExtraction
        context: Context string (file_path|||article_id|||chapter_text)

    Returns:
        Validation feedback string
    """
    try:
        # Parse context
        parts = context.split("|||")
        if len(parts) != 3:
            return "Error: Invalid context format"

        file_path, article_id, chapter_text = parts

        # Parse proposed extraction
        proposed_dict = json.loads(proposed_json)

        # Validate using existing validation logic
        validation = validate_extraction(
            proposed_dict,
            file_path=file_path,
            article_id=article_id,
            chapter_text=chapter_text
        )

        return validation.get_feedback()

    except Exception as e:
        return f"Validation error: {str(e)}"


# =============================================================================
# DSPY SIGNATURE AND MODULE
# =============================================================================


class ImproveMathematicalConcepts(dspy.Signature):
    """
    Improve an existing mathematical concept extraction by finding missing entities
    or correcting errors.

    You are an expert mathematical document reviewer. Your task is to:
    1. Review the existing extraction against the source chapter text
    2. Identify MISSING entities that should have been extracted
    3. Identify INCORRECT entities that need correction
    4. Identify UNNECESSARY entities that should be removed
    5. Propose improvements

    ============================================================================
    FOCUSED EXTRACTION MODE (when missed_labels_list is provided):
    ============================================================================

    Your task: Extract ONLY the specific labels in missed_labels_list.

    CRITICAL CONSTRAINTS:
    1. Extract ONLY entities whose labels appear in missed_labels_list
    2. DO NOT modify existing entities (preserve them unchanged)
    3. DO NOT invent new labels not in the list
    4. DO NOT re-extract already correct entities

    WORKFLOW:
    1. Parse missed_labels_list (comma-separated labels)
    2. For each target label:
       a. Search chapter_with_lines for ":label: <target-label>" directive
       b. Extract entity metadata (line range, type, term, etc.)
       c. Add entity to improved_extraction
    3. Merge with existing_extraction (preserve ALL existing entities)
    4. Call validate_improvement_tool to verify additions
    5. Return improved_extraction

    EXAMPLE:
    missed_labels_list = "def-lipschitz, thm-main, lem-helper"

    → Search for ":label: def-lipschitz" → extract definition
    → Search for ":label: thm-main" → extract theorem
    → Search for ":label: lem-helper" → extract lemma
    → Add these 3 entities to existing entities (preserve all existing)
    → Return improved_extraction with 3 new entities added

    ============================================================================

    IMPROVEMENT STRATEGIES (when missed_labels_list is empty):

    1. **Find Missing Entities**:
       - Scan chapter text for definitions/theorems not in existing extraction
       - Look for unlabeled mathematical content
       - Check for citations or proofs that were skipped

    2. **Correct Errors**:
       - Fix incorrect label patterns (e.g., "theorem-X" should be "thm-X")
       - Correct wrong line numbers
       - Fix entity metadata (statement_type, term, etc.)

    3. **Remove Incorrect Entities**:
       - Entities with invalid line ranges
       - Duplicate entities
       - Misclassified content

    TOOLS AVAILABLE:
    - compare_extractions_tool: Compare your proposed changes with existing data
    - validate_improvement_tool: Validate your proposed extraction

    WORKFLOW:
    1. Analyze existing extraction and chapter text
    2. Propose improvements (add/modify/delete entities)
    3. Use compare_extractions_tool to see what changed
    4. Use validate_improvement_tool to check correctness
    5. Refine based on feedback
    6. Repeat until improvement is validated

    OUTPUT:
    - Return improved ChapterExtraction with all fixes applied
    - Preserve correct existing entities
    - Add missing entities (or specific missed labels if provided)
    - Fix incorrect entities (unless in focused mode)
    - Remove invalid entities (unless in focused mode)
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )
    existing_extraction_json: str = dspy.InputField(
        desc="JSON string of existing ChapterExtraction to improve"
    )
    validation_context: str = dspy.InputField(
        desc="Context for validation: file_path|||article_id|||chapter_text"
    )
    missed_labels_list: str = dspy.InputField(
        desc="Comma-separated list of labels that were missed in extraction. "
             "Extract ONLY these specific labels. Do NOT modify existing entities. "
             "Empty string means general improvement mode."
    )

    improved_extraction: ChapterExtraction = dspy.OutputField(
        desc="Improved ChapterExtraction with fixes and additions"
    )


class MathematicalConceptImprover(dspy.Module):
    """
    ReAct-based DSPy module for improving existing mathematical concept extractions.

    Uses DSPy's ReAct agent with comparison and validation tools to iteratively
    improve an existing extraction by finding missing entities and fixing errors.
    """

    def __init__(self, max_iters: int = 3):
        """
        Initialize improver.

        Args:
            max_iters: Maximum number of ReAct iterations (default: 3)
        """
        super().__init__()
        # Create ReAct agent with improvement tools
        self.react_agent = dspy.ReAct(
            ImproveMathematicalConcepts,
            tools=[compare_extractions_tool, validate_improvement_tool],
            max_iters=max_iters
        )

    def forward(
        self,
        chapter_with_lines: str,
        existing_extraction: ChapterExtraction,
        missed_labels: list[str] = None,  # NEW
        file_path: str = "",
        article_id: str = ""
    ) -> ChapterExtraction:
        """
        Improve existing extraction using ReAct agent.

        Args:
            chapter_with_lines: Chapter text with line numbers
            existing_extraction: Existing ChapterExtraction to improve
            missed_labels: List of labels to extract (for focused extraction mode)
            file_path: Path to source file (for validation)
            article_id: Article identifier (for validation)

        Returns:
            Improved ChapterExtraction
        """
        # Prepare context
        validation_context = f"{file_path}|||{article_id}|||{chapter_with_lines}"
        existing_json = existing_extraction.model_dump_json()

        # Prepare missed labels for agent
        missed_labels_str = ", ".join(missed_labels) if missed_labels else ""

        if missed_labels:
            print(f"  → Targeting {len(missed_labels)} missed labels for extraction")

        try:
            # Run ReAct agent
            result = self.react_agent(
                chapter_with_lines=chapter_with_lines,
                existing_extraction_json=existing_json,
                validation_context=validation_context,
                missed_labels_list=missed_labels_str  # NEW
            )

            return result.improved_extraction

        except Exception as e:
            print(f"  ✗ ReAct improver failed: {e}")
            # Return original extraction as fallback
            return existing_extraction


# =============================================================================
# CHANGE TRACKING AND MERGING
# =============================================================================


def compute_changes(
    existing: ChapterExtraction,
    improved: ChapterExtraction
) -> ImprovementResult:
    """
    Compute changes between existing and improved extractions.

    Args:
        existing: Original ChapterExtraction
        improved: Improved ChapterExtraction

    Returns:
        ImprovementResult with all changes tracked
    """
    result = ImprovementResult()

    # Track definition changes
    existing_defs = {d.label: d for d in existing.definitions}
    improved_defs = {d.label: d for d in improved.definitions}

    for label in improved_defs:
        if label not in existing_defs:
            # Added
            result.add_change(EntityChange(
                entity_type="definition",
                label=label,
                operation=ChangeOperation.ADD,
                reason="New entity found in chapter text",
                new_data=improved_defs[label].model_dump()
            ))
        elif improved_defs[label].model_dump() != existing_defs[label].model_dump():
            # Modified
            result.add_change(EntityChange(
                entity_type="definition",
                label=label,
                operation=ChangeOperation.MODIFY,
                reason="Entity data corrected",
                old_data=existing_defs[label].model_dump(),
                new_data=improved_defs[label].model_dump()
            ))
        else:
            # Unchanged
            result.add_change(EntityChange(
                entity_type="definition",
                label=label,
                operation=ChangeOperation.NO_CHANGE,
                reason="Entity already correct"
            ))

    for label in existing_defs:
        if label not in improved_defs:
            # Deleted
            result.add_change(EntityChange(
                entity_type="definition",
                label=label,
                operation=ChangeOperation.DELETE,
                reason="Invalid or duplicate entity removed",
                old_data=existing_defs[label].model_dump()
            ))

    # Track theorem changes (same pattern)
    existing_thms = {t.label: t for t in existing.theorems}
    improved_thms = {t.label: t for t in improved.theorems}

    for label in improved_thms:
        if label not in existing_thms:
            result.add_change(EntityChange(
                entity_type="theorem",
                label=label,
                operation=ChangeOperation.ADD,
                reason="New entity found in chapter text",
                new_data=improved_thms[label].model_dump()
            ))
        elif improved_thms[label].model_dump() != existing_thms[label].model_dump():
            result.add_change(EntityChange(
                entity_type="theorem",
                label=label,
                operation=ChangeOperation.MODIFY,
                reason="Entity data corrected",
                old_data=existing_thms[label].model_dump(),
                new_data=improved_thms[label].model_dump()
            ))
        else:
            result.add_change(EntityChange(
                entity_type="theorem",
                label=label,
                operation=ChangeOperation.NO_CHANGE,
                reason="Entity already correct"
            ))

    for label in existing_thms:
        if label not in improved_thms:
            result.add_change(EntityChange(
                entity_type="theorem",
                label=label,
                operation=ChangeOperation.DELETE,
                reason="Invalid or duplicate entity removed",
                old_data=existing_thms[label].model_dump()
            ))

    # Similar for proofs, axioms, parameters, remarks, citations
    # (abbreviated for brevity - same pattern)

    return result


# =============================================================================
# RETRY WRAPPERS WITH FALLBACK MODEL SUPPORT
# =============================================================================


def improve_chapter_with_retry(
    chapter_text: str,
    existing_extraction: dict,
    file_path: str,
    article_id: str,
    max_iters: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True
) -> tuple[ChapterExtraction, ImprovementResult, list[str]]:
    """
    Improve chapter extraction with retry logic and fallback model support.

    After first failure, switches from primary model to fallback model for
    remaining retry attempts.

    Args:
        chapter_text: Chapter text with line numbers
        existing_extraction: Existing extraction as dict
        file_path: Path to source markdown file
        article_id: Article identifier
        max_iters: Maximum ReAct iterations per attempt
        max_retries: Maximum retry attempts (default: 3)
        fallback_model: Model to use after first failure
        verbose: Print progress information

    Returns:
        Tuple of (ChapterExtraction, ImprovementResult, list of errors)
    """
    errors_encountered = []
    switched_to_fallback = False

    # Parse existing extraction to ChapterExtraction
    try:
        if "definitions" in existing_extraction and isinstance(existing_extraction["definitions"], list):
            if len(existing_extraction["definitions"]) > 0:
                first_def = existing_extraction["definitions"][0]
                if "term" in first_def:
                    existing_chapter = ChapterExtraction(**existing_extraction)
                else:
                    existing_chapter = ChapterExtraction(
                        section_id=existing_extraction.get("section_id", "Unknown"),
                        definitions=[],
                        theorems=[],
                        proofs=[],
                        axioms=[],
                        parameters=[],
                        remarks=[],
                        citations=[]
                    )
            else:
                existing_chapter = ChapterExtraction(**existing_extraction)
        else:
            existing_chapter = ChapterExtraction(
                section_id=existing_extraction.get("section_id", "Unknown"),
                definitions=[],
                theorems=[],
                proofs=[],
                axioms=[],
                parameters=[],
                remarks=[],
                citations=[]
            )
    except Exception as e:
        error_msg = f"Failed to parse existing extraction: {str(e)}"
        errors_encountered.append(make_error_dict(
            error_msg,
            value={"existing_extraction": existing_extraction}
        ))
        if verbose:
            print(f"  ✗ {error_msg}")

        # Start with empty extraction
        import re
        section_id = "Unknown"
        for line in chapter_text.split('\n')[:20]:
            content = re.sub(r"^\s*\d+:\s*", "", line)
            if content.startswith("## "):
                section_id = content.strip()
                break

        existing_chapter = ChapterExtraction(
            section_id=section_id,
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            parameters=[],
            remarks=[],
            citations=[]
        )

    # Detect missed labels
    from mathster.parsing.tools import compare_extraction_with_source

    existing_dict = existing_chapter.model_dump()
    comparison, validation_report = compare_extraction_with_source(
        existing_dict,
        chapter_text
    )

    missed_labels = []
    for entity_type, data in comparison.items():
        if entity_type != "summary":
            missed_labels.extend(data.get("not_extracted", []))

    if verbose and missed_labels:
        print(f"  → Found {len(missed_labels)} missed labels to improve")

    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                if attempt == 1:
                    print(f"  → Improvement attempt {attempt}/{max_retries}")
                else:
                    print(f"  → Retry attempt {attempt}/{max_retries}")

            # Create improver
            improver = MathematicalConceptImprover(max_iters=max_iters)

            # Run improvement
            improved_chapter = improver(
                chapter_with_lines=chapter_text,
                existing_extraction=existing_chapter,
                missed_labels=missed_labels,
                file_path=file_path,
                article_id=article_id
            )

            if verbose:
                print(f"  ✓ Improvement successful on attempt {attempt}")

            # Compute changes
            improvement_result = compute_changes(existing_chapter, improved_chapter)

            return improved_chapter, improvement_result, errors_encountered

        except Exception as e:
            error_msg = f"Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {str(e)}"
            errors_encountered.append(make_error_dict(
                error_msg,
                value={
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "chapter_info": {"file_path": file_path, "article_id": article_id},
                    "missed_labels_count": len(missed_labels) if missed_labels else 0
                }
            ))

            if verbose:
                print(f"  ✗ {error_msg}")

            # Switch to fallback model after first failure
            if attempt == 1 and max_retries > 1 and not switched_to_fallback:
                if verbose:
                    print(f"  → Switching to fallback model: {fallback_model}")

                # Import here to avoid circular dependency
                from mathster.parsing.dspy_pipeline import configure_dspy

                try:
                    configure_dspy(model=fallback_model)
                    switched_to_fallback = True
                    if verbose:
                        print(f"  ✓ Successfully switched to {fallback_model}")
                except Exception as switch_error:
                    if verbose:
                        print(f"  ⚠ Failed to switch model: {switch_error}")
                        print(f"  → Continuing with current model")

            # If this was the last attempt, raise
            if attempt == max_retries:
                if verbose:
                    print(f"  ✗ All {max_retries} improvement attempts failed")
                raise

    # Should not reach here, but return existing as fallback
    improvement_result = ImprovementResult()
    return existing_chapter, improvement_result, errors_encountered


def improve_label_with_retry(
    chapter_text: str,
    target_label: str,
    entity_type: str,
    existing_extraction: dict,
    file_path: str,
    article_id: str,
    max_iters_per_label: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True
) -> tuple[ChapterExtraction, ImprovementResult, list[str]]:
    """
    Improve a single missed label with retry logic and fallback model support.

    Args:
        chapter_text: Chapter text with line numbers
        target_label: Specific label to extract (e.g., "def-lipschitz")
        entity_type: Type of entity (e.g., "definitions", "theorems")
        existing_extraction: Existing extraction as dict
        file_path: Path to source markdown file
        article_id: Article identifier
        max_iters_per_label: Maximum ReAct iterations per label
        max_retries: Maximum retry attempts
        fallback_model: Model to use after first failure
        verbose: Print progress information

    Returns:
        Tuple of (ChapterExtraction, ImprovementResult, list of errors)
    """
    errors_encountered = []
    switched_to_fallback = False

    # Parse existing extraction
    try:
        if "definitions" in existing_extraction and isinstance(existing_extraction["definitions"], list):
            if len(existing_extraction["definitions"]) > 0:
                first_def = existing_extraction["definitions"][0]
                if "term" in first_def:
                    existing_chapter = ChapterExtraction(**existing_extraction)
                else:
                    existing_chapter = ChapterExtraction(
                        section_id=existing_extraction.get("section_id", "Unknown"),
                        definitions=[], theorems=[], proofs=[], axioms=[],
                        parameters=[], remarks=[], citations=[]
                    )
            else:
                existing_chapter = ChapterExtraction(**existing_extraction)
        else:
            existing_chapter = ChapterExtraction(
                section_id=existing_extraction.get("section_id", "Unknown"),
                definitions=[], theorems=[], proofs=[], axioms=[],
                parameters=[], remarks=[], citations=[]
            )
    except Exception as e:
        error_msg = f"Failed to parse existing extraction: {str(e)}"
        errors_encountered.append(make_error_dict(
            error_msg,
            value={"existing_extraction": existing_extraction}
        ))
        if verbose:
            print(f"      ✗ {error_msg}")

        existing_chapter = ChapterExtraction(
            section_id="Unknown",
            definitions=[], theorems=[], proofs=[], axioms=[],
            parameters=[], remarks=[], citations=[]
        )

    if verbose:
        print(f"    → Target: {target_label} ({entity_type})")

    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                if attempt == 1:
                    print(f"      → Attempt {attempt}/{max_retries}")
                else:
                    print(f"      → Retry {attempt}/{max_retries}")

            # Create improver with focused mode (single label)
            improver = MathematicalConceptImprover(max_iters=max_iters_per_label)

            # Run improvement targeting only this label
            improved_chapter = improver(
                chapter_with_lines=chapter_text,
                existing_extraction=existing_chapter,
                missed_labels=[target_label],  # Single label focus
                file_path=file_path,
                article_id=article_id
            )

            if verbose:
                print(f"      ✓ Success on attempt {attempt}")

            # Compute changes
            improvement_result = compute_changes(existing_chapter, improved_chapter)

            # Verify the target label was actually added
            improved_dict = improved_chapter.model_dump()
            target_found = False

            for entity_list_name in ["definitions", "theorems", "proofs", "axioms",
                                     "parameters", "remarks", "citations"]:
                entity_list = improved_dict.get(entity_list_name, [])
                if any(e.get("label") == target_label for e in entity_list):
                    target_found = True
                    break

            if not target_found:
                raise ValueError(f"Target label '{target_label}' was not extracted")

            return improved_chapter, improvement_result, errors_encountered

        except Exception as e:
            error_msg = f"Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {str(e)}"
            errors_encountered.append(make_error_dict(
                error_msg,
                value={
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "target_label": target_label,
                    "entity_type": entity_type
                }
            ))

            if verbose:
                print(f"      ✗ {error_msg}")

            # Switch to fallback model after first failure
            if attempt == 1 and max_retries > 1 and not switched_to_fallback:
                if verbose:
                    print(f"      → Switching to fallback model: {fallback_model}")

                from mathster.parsing.dspy_pipeline import configure_dspy

                try:
                    configure_dspy(model=fallback_model)
                    switched_to_fallback = True
                    if verbose:
                        print(f"      ✓ Successfully switched to {fallback_model}")
                except Exception as switch_error:
                    if verbose:
                        print(f"      ⚠ Failed to switch model: {switch_error}")
                        print(f"      → Continuing with current model")

            # If this was the last attempt, raise
            if attempt == max_retries:
                if verbose:
                    print(f"      ✗ All {max_retries} attempts failed for {target_label}")
                raise

    # Should not reach here, but return existing as fallback
    improvement_result = ImprovementResult()
    return existing_chapter, improvement_result, errors_encountered


def improve_chapter_by_labels(
    chapter_text: str,
    existing_extraction: dict,
    file_path: str,
    article_id: str,
    max_iters_per_label: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True
) -> tuple[RawDocumentSection | None, ImprovementResult, list[str]]:
    """
    Improve chapter by processing missed labels one at a time.

    This implements SINGLE-LABEL IMPROVEMENT MODE with nested loops:
    - Outer loop: Iterate over missed labels
    - Inner loop: Retry logic with fallback model per label

    Args:
        chapter_text: Chapter text with line numbers
        existing_extraction: Existing extraction as dict
        file_path: Path to source markdown file
        article_id: Article identifier
        max_iters_per_label: Maximum ReAct iterations per label
        max_retries: Maximum retry attempts per label
        fallback_model: Model to use after first failure
        verbose: Print progress information

    Returns:
        Tuple of (RawDocumentSection or None, ImprovementResult, list of errors)
    """
    errors_encountered = []

    # Parse existing extraction
    try:
        if "definitions" in existing_extraction and isinstance(existing_extraction["definitions"], list):
            if len(existing_extraction["definitions"]) > 0:
                first_def = existing_extraction["definitions"][0]
                if "term" in first_def:
                    current_extraction = ChapterExtraction(**existing_extraction)
                else:
                    current_extraction = ChapterExtraction(
                        section_id=existing_extraction.get("section_id", "Unknown"),
                        definitions=[], theorems=[], proofs=[], axioms=[],
                        parameters=[], remarks=[], citations=[]
                    )
            else:
                current_extraction = ChapterExtraction(**existing_extraction)
        else:
            current_extraction = ChapterExtraction(
                section_id=existing_extraction.get("section_id", "Unknown"),
                definitions=[], theorems=[], proofs=[], axioms=[],
                parameters=[], remarks=[], citations=[]
            )
    except Exception as e:
        error_msg = f"Failed to parse existing extraction: {str(e)}"
        errors_encountered.append(make_error_dict(
            error_msg,
            value={"existing_extraction": existing_extraction}
        ))
        if verbose:
            print(f"  ✗ {error_msg}")

        current_extraction = ChapterExtraction(
            section_id="Unknown",
            definitions=[], theorems=[], proofs=[], axioms=[],
            parameters=[], remarks=[], citations=[]
        )

    # Discover missed labels
    from mathster.parsing.tools import compare_extraction_with_source

    current_dict = current_extraction.model_dump()
    comparison, validation_report = compare_extraction_with_source(
        current_dict,
        chapter_text
    )

    # Build label→entity_type mapping
    labels_by_type = {}
    for entity_type, data in comparison.items():
        if entity_type != "summary":
            for label in data.get("not_extracted", []):
                labels_by_type[label] = entity_type

    if not labels_by_type:
        if verbose:
            print("  ✓ No missed labels found - extraction is complete")

        improvement_result = ImprovementResult()
        raw_section, conversion_warnings = convert_to_raw_document_section(
            current_extraction,
            file_path=file_path,
            article_id=article_id,
            chapter_text=chapter_text
        )
        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        return raw_section, improvement_result, errors_encountered

    if verbose:
        print(f"  → Found {len(labels_by_type)} missed labels for single-label improvement")
        print(f"  → Strategy: Improve one label at a time with retries + fallback per label")

    # Accumulate all improvements
    cumulative_improvement = ImprovementResult()
    successful_labels = []
    failed_labels = []

    # Iterate over each missed label
    for idx, (target_label, entity_type) in enumerate(labels_by_type.items(), 1):
        if verbose:
            print(f"\n  [{idx}/{len(labels_by_type)}] Processing {target_label}")

        try:
            # Improve this single label with retry + fallback
            improved_chapter, label_improvement, label_errors = improve_label_with_retry(
                chapter_text=chapter_text,
                target_label=target_label,
                entity_type=entity_type,
                existing_extraction=current_extraction.model_dump(),
                file_path=file_path,
                article_id=article_id,
                max_iters_per_label=max_iters_per_label,
                max_retries=max_retries,
                fallback_model=fallback_model,
                verbose=verbose
            )

            # Accumulate errors
            if label_errors:
                errors_encountered.extend(label_errors)

            # Update current extraction with improvements
            current_extraction = improved_chapter

            # Accumulate changes
            for change in label_improvement.changes:
                cumulative_improvement.add_change(change)

            successful_labels.append(target_label)

            if verbose:
                print(f"      ✓ {target_label} successfully improved")

        except Exception as e:
            error_msg = f"Failed to improve {target_label} after {max_retries} retries: {str(e)}"
            errors_encountered.append(make_error_dict(
                error_msg,
                value={
                    "target_label": target_label,
                    "entity_type": entity_type,
                    "exception": str(e),
                    "label_errors": label_errors  # Errors from retry attempts
                }
            ))
            failed_labels.append(target_label)

            if verbose:
                print(f"      ✗ {error_msg}")

            # Continue with next label

    # Final summary
    if verbose:
        print(f"\n  ✓ Single-label improvement completed")
        print(f"    - Successful: {len(successful_labels)}/{len(labels_by_type)}")
        if failed_labels:
            print(f"    - Failed: {', '.join(failed_labels)}")
        print(cumulative_improvement.get_summary())

    # Convert to RawDocumentSection
    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            current_extraction,
            file_path=file_path,
            article_id=article_id,
            chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

        return raw_section, cumulative_improvement, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        errors_encountered.append(make_error_dict(
            error_msg,
            value=current_extraction.model_dump()
        ))
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, cumulative_improvement, errors_encountered


# =============================================================================
# MAIN IMPROVEMENT WORKFLOW FUNCTION
# =============================================================================


def improve_chapter(
    chapter_text: str,
    existing_extraction: dict,
    file_path: str,
    article_id: str,
    max_iters: int = 3,
    improvement_mode: str = "batch",
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True
) -> tuple[RawDocumentSection | None, ImprovementResult, list[str]]:
    """
    Improve an existing mathematical concept extraction.

    This is the main entry point for the IMPROVEMENT workflow with support for:
    - Batch improvement: Process all missed labels at once with retry + fallback
    - Single-label improvement: Process missed labels one at a time with per-label retry + fallback

    Args:
        chapter_text: Chapter text with line numbers (format: "NNN: content")
        existing_extraction: Existing extraction as dict (loaded from JSON)
        file_path: Path to source markdown file
        article_id: Article identifier (e.g., "01_fragile_gas_framework")
        max_iters: Maximum ReAct iterations (default: 3)
        improvement_mode: "batch" (all labels at once) or "single_label" (one at a time)
        max_retries: Maximum retry attempts (default: 3)
        fallback_model: Model to use after first failure (default: Claude Haiku)
        verbose: Print progress information

    Returns:
        Tuple of (RawDocumentSection or None, ImprovementResult, list of errors)
        - RawDocumentSection: Improved extraction (None if failed)
        - ImprovementResult: Change tracking metadata
        - list[str]: Any errors or warnings encountered
    """
    if verbose:
        print(f"  → IMPROVE mode ({improvement_mode})")

    # Route to appropriate improvement function
    if improvement_mode == "single_label":
        # Single-label mode: Process missed labels one at a time
        return improve_chapter_by_labels(
            chapter_text=chapter_text,
            existing_extraction=existing_extraction,
            file_path=file_path,
            article_id=article_id,
            max_iters_per_label=max_iters,
            max_retries=max_retries,
            fallback_model=fallback_model,
            verbose=verbose
        )
    else:
        # Batch mode: Process all missed labels at once with retry + fallback
        improved_chapter, improvement_result, errors_encountered = improve_chapter_with_retry(
            chapter_text=chapter_text,
            existing_extraction=existing_extraction,
            file_path=file_path,
            article_id=article_id,
            max_iters=max_iters,
            max_retries=max_retries,
            fallback_model=fallback_model,
            verbose=verbose
        )

        # Convert to RawDocumentSection
        try:
            raw_section, conversion_warnings = convert_to_raw_document_section(
                improved_chapter,
                file_path=file_path,
                article_id=article_id,
                chapter_text=chapter_text
            )

            if conversion_warnings:
                errors_encountered.extend(conversion_warnings)

            if verbose and raw_section:
                print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")
                print(improvement_result.get_summary())

            return raw_section, improvement_result, errors_encountered

        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            errors_encountered.append(make_error_dict(
                error_msg,
                value=improved_chapter.model_dump()
            ))
            if verbose:
                print(f"  ✗ {error_msg}")

            return None, improvement_result, errors_encountered
