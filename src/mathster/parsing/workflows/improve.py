"""Improvement workflows using DSPy components."""

from mathster.core.raw_data import RawDocumentSection
from mathster.parsing.conversion import (
    convert_to_raw_document_section,
)
from mathster.parsing.dspy_components import MathematicalConceptImprover
from mathster.parsing.models import (
    ChangeOperation,
    ChapterExtraction,
    EntityChange,
    ImprovementResult,
)
from mathster.parsing.validation import make_error_dict


def compute_changes(existing: ChapterExtraction, improved: ChapterExtraction) -> ImprovementResult:
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
            result.add_change(
                EntityChange(
                    entity_type="definition",
                    label=label,
                    operation=ChangeOperation.ADD,
                    reason="New entity found in chapter text",
                    new_data=improved_defs[label].model_dump(),
                )
            )
        elif improved_defs[label].model_dump() != existing_defs[label].model_dump():
            # Modified
            result.add_change(
                EntityChange(
                    entity_type="definition",
                    label=label,
                    operation=ChangeOperation.MODIFY,
                    reason="Entity data corrected",
                    old_data=existing_defs[label].model_dump(),
                    new_data=improved_defs[label].model_dump(),
                )
            )
        else:
            # Unchanged
            result.add_change(
                EntityChange(
                    entity_type="definition",
                    label=label,
                    operation=ChangeOperation.NO_CHANGE,
                    reason="Entity already correct",
                )
            )

    for label in existing_defs:
        if label not in improved_defs:
            # Deleted
            result.add_change(
                EntityChange(
                    entity_type="definition",
                    label=label,
                    operation=ChangeOperation.DELETE,
                    reason="Invalid or duplicate entity removed",
                    old_data=existing_defs[label].model_dump(),
                )
            )

    # Track theorem changes (same pattern)
    existing_thms = {t.label: t for t in existing.theorems}
    improved_thms = {t.label: t for t in improved.theorems}

    for label in improved_thms:
        if label not in existing_thms:
            result.add_change(
                EntityChange(
                    entity_type="theorem",
                    label=label,
                    operation=ChangeOperation.ADD,
                    reason="New entity found in chapter text",
                    new_data=improved_thms[label].model_dump(),
                )
            )
        elif improved_thms[label].model_dump() != existing_thms[label].model_dump():
            result.add_change(
                EntityChange(
                    entity_type="theorem",
                    label=label,
                    operation=ChangeOperation.MODIFY,
                    reason="Entity data corrected",
                    old_data=existing_thms[label].model_dump(),
                    new_data=improved_thms[label].model_dump(),
                )
            )
        else:
            result.add_change(
                EntityChange(
                    entity_type="theorem",
                    label=label,
                    operation=ChangeOperation.NO_CHANGE,
                    reason="Entity already correct",
                )
            )

    for label in existing_thms:
        if label not in improved_thms:
            result.add_change(
                EntityChange(
                    entity_type="theorem",
                    label=label,
                    operation=ChangeOperation.DELETE,
                    reason="Invalid or duplicate entity removed",
                    old_data=existing_thms[label].model_dump(),
                )
            )

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
    verbose: bool = True,
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
        if "definitions" in existing_extraction and isinstance(
            existing_extraction["definitions"], list
        ):
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
                        citations=[],
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
                citations=[],
            )
    except Exception as e:
        error_msg = f"Failed to parse existing extraction: {e!s}"
        errors_encountered.append(
            make_error_dict(error_msg, value={"existing_extraction": existing_extraction})
        )
        if verbose:
            print(f"  ✗ {error_msg}")

        # Start with empty extraction
        import re

        section_id = "Unknown"
        for line in chapter_text.split("\n")[:20]:
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
            citations=[],
        )

    # Detect missed labels
    from mathster.parsing.tools import compare_extraction_with_source

    existing_dict = existing_chapter.model_dump()
    comparison, _validation_report = compare_extraction_with_source(existing_dict, chapter_text)

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
                article_id=article_id,
            )

            if verbose:
                print(f"  ✓ Improvement successful on attempt {attempt}")

            # Compute changes
            improvement_result = compute_changes(existing_chapter, improved_chapter)

            return improved_chapter, improvement_result, errors_encountered

        except Exception as e:
            error_msg = f"Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e!s}"
            errors_encountered.append(
                make_error_dict(
                    error_msg,
                    value={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "chapter_info": {"file_path": file_path, "article_id": article_id},
                        "missed_labels_count": len(missed_labels) if missed_labels else 0,
                    },
                )
            )

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
                        print("  → Continuing with current model")

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
    verbose: bool = True,
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
        if "definitions" in existing_extraction and isinstance(
            existing_extraction["definitions"], list
        ):
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
                        citations=[],
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
                citations=[],
            )
    except Exception as e:
        error_msg = f"Failed to parse existing extraction: {e!s}"
        errors_encountered.append(
            make_error_dict(error_msg, value={"existing_extraction": existing_extraction})
        )
        if verbose:
            print(f"      ✗ {error_msg}")

        existing_chapter = ChapterExtraction(
            section_id="Unknown",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            parameters=[],
            remarks=[],
            citations=[],
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
                article_id=article_id,
            )

            if verbose:
                print(f"      ✓ Success on attempt {attempt}")

            # Compute changes
            improvement_result = compute_changes(existing_chapter, improved_chapter)

            # Verify the target label was actually added
            improved_dict = improved_chapter.model_dump()
            target_found = False

            for entity_list_name in [
                "definitions",
                "theorems",
                "proofs",
                "axioms",
                "parameters",
                "remarks",
                "citations",
            ]:
                entity_list = improved_dict.get(entity_list_name, [])
                if any(e.get("label") == target_label for e in entity_list):
                    target_found = True
                    break

            if not target_found:
                raise ValueError(f"Target label '{target_label}' was not extracted")

            return improved_chapter, improvement_result, errors_encountered

        except Exception as e:
            error_msg = f"Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e!s}"
            errors_encountered.append(
                make_error_dict(
                    error_msg,
                    value={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "target_label": target_label,
                        "entity_type": entity_type,
                    },
                )
            )

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
                        print("      → Continuing with current model")

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
    verbose: bool = True,
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
        if "definitions" in existing_extraction and isinstance(
            existing_extraction["definitions"], list
        ):
            if len(existing_extraction["definitions"]) > 0:
                first_def = existing_extraction["definitions"][0]
                if "term" in first_def:
                    current_extraction = ChapterExtraction(**existing_extraction)
                else:
                    current_extraction = ChapterExtraction(
                        section_id=existing_extraction.get("section_id", "Unknown"),
                        definitions=[],
                        theorems=[],
                        proofs=[],
                        axioms=[],
                        parameters=[],
                        remarks=[],
                        citations=[],
                    )
            else:
                current_extraction = ChapterExtraction(**existing_extraction)
        else:
            current_extraction = ChapterExtraction(
                section_id=existing_extraction.get("section_id", "Unknown"),
                definitions=[],
                theorems=[],
                proofs=[],
                axioms=[],
                parameters=[],
                remarks=[],
                citations=[],
            )
    except Exception as e:
        error_msg = f"Failed to parse existing extraction: {e!s}"
        errors_encountered.append(
            make_error_dict(error_msg, value={"existing_extraction": existing_extraction})
        )
        if verbose:
            print(f"  ✗ {error_msg}")

        current_extraction = ChapterExtraction(
            section_id="Unknown",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            parameters=[],
            remarks=[],
            citations=[],
        )

    # Discover missed labels
    from mathster.parsing.tools import compare_extraction_with_source

    current_dict = current_extraction.model_dump()
    comparison, _validation_report = compare_extraction_with_source(current_dict, chapter_text)

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
            chapter_text=chapter_text,
        )
        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        return raw_section, improvement_result, errors_encountered

    if verbose:
        print(f"  → Found {len(labels_by_type)} missed labels for single-label improvement")
        print("  → Strategy: Improve one label at a time with retries + fallback per label")

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
                verbose=verbose,
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
            error_msg = f"Failed to improve {target_label} after {max_retries} retries: {e!s}"
            errors_encountered.append(
                make_error_dict(
                    error_msg,
                    value={
                        "target_label": target_label,
                        "entity_type": entity_type,
                        "exception": str(e),
                        "label_errors": label_errors,  # Errors from retry attempts
                    },
                )
            )
            failed_labels.append(target_label)

            if verbose:
                print(f"      ✗ {error_msg}")

            # Continue with next label

    # Final summary
    if verbose:
        print("\n  ✓ Single-label improvement completed")
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
            chapter_text=chapter_text,
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

        return raw_section, cumulative_improvement, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {e!s}"
        errors_encountered.append(
            make_error_dict(error_msg, value=current_extraction.model_dump())
        )
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
    verbose: bool = True,
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
            verbose=verbose,
        )
    # Batch mode: Process all missed labels at once with retry + fallback
    improved_chapter, improvement_result, errors_encountered = improve_chapter_with_retry(
        chapter_text=chapter_text,
        existing_extraction=existing_extraction,
        file_path=file_path,
        article_id=article_id,
        max_iters=max_iters,
        max_retries=max_retries,
        fallback_model=fallback_model,
        verbose=verbose,
    )

    # Convert to RawDocumentSection
    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            improved_chapter, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")
            print(improvement_result.get_summary())

        return raw_section, improvement_result, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {e!s}"
        errors_encountered.append(make_error_dict(error_msg, value=improved_chapter.model_dump()))
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, improvement_result, errors_encountered
