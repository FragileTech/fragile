"""Extraction workflows using DSPy components."""

import re

from mathster.core.raw_data import RawDocumentSection
from mathster.parsing.conversion import (
    convert_dict_to_extraction_entity,
    convert_to_raw_document_section,
)
from mathster.parsing.dspy_components import MathematicalConceptExtractor, SingleLabelExtractor
from mathster.parsing.models import ChapterExtraction
from mathster.parsing.text_processing import analyze_labels_in_chapter
from mathster.parsing.validation import generate_detailed_error_report, make_error_dict


def extract_chapter_with_retry(
    chapter_text: str,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_iters: int = 10,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> tuple[ChapterExtraction, list[str]]:
    """Extract chapter with automatic retry on failure."""
    errors_encountered = []
    extractor = MathematicalConceptExtractor(max_iters=max_iters)
    switched_to_fallback = False

    for attempt in range(1, max_retries + 1):
        try:
            previous_error_report = ""
            if attempt > 1 and errors_encountered:
                last_error = errors_encountered[-1]
                previous_error_report = f"Previous attempt failed with: {last_error}"

            if verbose and attempt > 1:
                print(f"  → Retry attempt {attempt}/{max_retries}")

            extraction = extractor(
                chapter_with_lines=chapter_text,
                chapter_number=chapter_number,
                file_path=file_path,
                article_id=article_id,
                previous_error_report=previous_error_report,
            )

            if verbose and attempt > 1:
                print(f"  ✓ Retry successful on attempt {attempt}")

            return extraction, errors_encountered

        except Exception as e:
            extraction_context = {
                "chapter_number": chapter_number,
                "file_path": file_path,
                "article_id": article_id,
            }

            error_report = generate_detailed_error_report(
                error=e,
                attempt_number=attempt,
                max_retries=max_retries,
                extraction_context=extraction_context,
            )

            error_msg = f"Attempt {attempt} failed: {type(e).__name__}: {e!s}"
            errors_encountered.append(
                make_error_dict(
                    error_msg,
                    value={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "chapter_info": {
                            "chapter_number": chapter_number,
                            "file_path": file_path,
                            "article_id": article_id,
                        },
                    },
                )
            )

            if verbose:
                print(f"  ✗ Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
                if attempt < max_retries:
                    print(f"\n{error_report}\n")

            if attempt == 1 and max_retries > 1 and not switched_to_fallback:
                if verbose:
                    print(f"  → Switching to fallback model: {fallback_model}")

                from mathster.parsing.config import configure_dspy

                try:
                    configure_dspy(model=fallback_model)
                    switched_to_fallback = True
                    if verbose:
                        print(f"  ✓ Successfully switched to {fallback_model}")
                except Exception as switch_error:
                    if verbose:
                        print(f"  ⚠ Failed to switch model: {switch_error}")
                        print("  → Continuing with current model")

            if attempt == max_retries:
                if verbose:
                    print(f"  ✗ All {max_retries} attempts failed")
                    print(f"\n{error_report}\n")
                raise Exception(
                    f"Extraction failed after {max_retries} attempts. "
                    f"Last error: {type(e).__name__}: {e!s}"
                ) from e

    raise Exception(f"Extraction failed after {max_retries} attempts")


def extract_label_with_retry(
    chapter_text: str,
    target_label: str,
    entity_type: str,
    file_path: str,
    article_id: str,
    max_iters_per_label: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = False,
) -> tuple[dict, list[str]]:
    """Extract single label with automatic retry on failure."""
    errors_encountered = []
    extractor = SingleLabelExtractor(max_iters=max_iters_per_label)
    switched_to_fallback = False

    for attempt in range(1, max_retries + 1):
        try:
            previous_error_report = ""
            if attempt > 1 and errors_encountered:
                last_error = errors_encountered[-1]
                previous_error_report = f"Previous attempt failed with: {last_error}"

            if verbose and attempt > 1:
                print(f"      → Retry attempt {attempt}/{max_retries} for {target_label}")

            entity_dict = extractor(
                chapter_with_lines=chapter_text,
                target_label=target_label,
                file_path=file_path,
                article_id=article_id,
                previous_error_report=previous_error_report,
            )

            if "extraction_error" in entity_dict:
                raise Exception(entity_dict["extraction_error"])

            if verbose and attempt > 1:
                print(f"      ✓ Retry successful on attempt {attempt}")

            return entity_dict, errors_encountered

        except Exception as e:
            extraction_context = {
                "target_label": target_label,
                "entity_type": entity_type,
                "file_path": file_path,
                "article_id": article_id,
            }

            error_report = generate_detailed_error_report(
                error=e,
                attempt_number=attempt,
                max_retries=max_retries,
                extraction_context=extraction_context,
            )

            error_msg = f"Attempt {attempt} failed: {type(e).__name__}: {e!s}"
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
                print(f"      ✗ Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
                if attempt < max_retries:
                    print(f"\n{error_report}\n")

            if attempt == 1 and max_retries > 1 and not switched_to_fallback:
                if verbose:
                    print(f"      → Switching to fallback model: {fallback_model}")

                from mathster.parsing.config import configure_dspy

                try:
                    configure_dspy(model=fallback_model)
                    switched_to_fallback = True
                    if verbose:
                        print(f"      ✓ Successfully switched to {fallback_model}")
                except Exception as switch_error:
                    if verbose:
                        print(f"      ⚠ Failed to switch model: {switch_error}")
                        print("      → Continuing with current model")

            if attempt == max_retries:
                if verbose:
                    print(f"      ✗ All {max_retries} attempts failed for {target_label}")
                raise Exception(
                    f"Failed to extract {target_label} after {max_retries} attempts. "
                    f"Last error: {type(e).__name__}: {e!s}"
                ) from e

    raise Exception(f"Failed to extract {target_label} after {max_retries} attempts")


def extract_chapter(
    chapter_text: str,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_iters: int = 10,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> tuple[RawDocumentSection | None, list[str]]:
    """Extract mathematical concepts from a chapter using ReAct agent with retry logic."""
    errors_encountered = []

    try:
        extraction, retry_errors = extract_chapter_with_retry(
            chapter_text=chapter_text,
            chapter_number=chapter_number,
            file_path=file_path,
            article_id=article_id,
            max_iters=max_iters,
            max_retries=max_retries,
            fallback_model=fallback_model,
            verbose=verbose,
        )

        errors_encountered.extend(retry_errors)

        if verbose:
            if retry_errors:
                print(f"  ✓ Extraction completed after {len(retry_errors) + 1} attempts")
            else:
                print("  ✓ Extraction completed")

    except Exception as e:
        error_msg = f"Extraction failed after {max_retries} attempts: {e!s}"
        errors_encountered.append(
            make_error_dict(
                error_msg,
                value={
                    "chapter_number": chapter_number,
                    "file_path": file_path,
                    "article_id": article_id,
                    "exception": str(e),
                },
            )
        )
        if verbose:
            print(f"  ✗ {error_msg}")

        section_id = f"Chapter {chapter_number}"
        for line in chapter_text.split("\n")[:20]:
            content = re.sub(r"^\s*\d+:\s*", "", line)
            if content.startswith("## "):
                section_id = content.strip()
                break

        extraction = ChapterExtraction(
            section_id=section_id,
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            assumptions=[],
            parameters=[],
            remarks=[],
            citations=[],
        )

    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            extraction, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

            try:
                from mathster.parsing.text_processing import compare_extraction_with_source

                _, report = compare_extraction_with_source(raw_section, chapter_text)
                print("\n" + "=" * 70)
                print("EXTRACTION REPORT")
                print("=" * 70)
                print(report)
                print("=" * 70 + "\n")
            except Exception as e:
                print(f"  ⚠ Could not generate extraction report: {e}")

        return raw_section, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {e!s}"
        errors_encountered.append(make_error_dict(error_msg, value=extraction.model_dump()))
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, errors_encountered


def extract_chapter_by_labels(
    chapter_text: str,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_iters_per_label: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> tuple[RawDocumentSection | None, list[str]]:
    """Extract chapter by iterating over individual labels with retry logic."""
    errors_encountered = []

    labels_by_type, report = analyze_labels_in_chapter(chapter_text)
    total_labels = sum(len(labels) for labels in labels_by_type.values())

    if verbose:
        print(f"  → Found {total_labels} labels to extract")
        for entity_type, labels in labels_by_type.items():
            if labels:
                print(f"    • {entity_type}: {len(labels)}")

    if total_labels == 0:
        if verbose:
            print("  ⚠ No labels found in chapter")

        section_id = f"Chapter {chapter_number}"
        for line in chapter_text.split("\n")[:20]:
            content = re.sub(r"^\s*\d+:\s*", "", line)
            if content.startswith("## "):
                section_id = content.strip()
                break

        extraction = ChapterExtraction(
            section_id=section_id,
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            assumptions=[],
            parameters=[],
            remarks=[],
            citations=[],
        )
        raw_section, _ = convert_to_raw_document_section(
            extraction, file_path, article_id, chapter_text
        )
        return raw_section, []

    section_id = f"Chapter {chapter_number}"
    for line in chapter_text.split("\n")[:20]:
        content = re.sub(r"^\s*\d+:\s*", "", line)
        if content.startswith("## "):
            section_id = content.strip()
            break

    extraction = ChapterExtraction(
        section_id=section_id,
        definitions=[],
        theorems=[],
        proofs=[],
        axioms=[],
        assumptions=[],
        parameters=[],
        remarks=[],
        citations=[],
    )

    label_counter = 0
    successful_extractions = 0

    for entity_type, labels in labels_by_type.items():
        for label in labels:
            label_counter += 1
            if verbose:
                print(f"  [{label_counter}/{total_labels}] {label}...", end=" ")

            try:
                entity_dict, retry_errors = extract_label_with_retry(
                    chapter_text=chapter_text,
                    target_label=label,
                    entity_type=entity_type,
                    file_path=file_path,
                    article_id=article_id,
                    max_iters_per_label=max_iters_per_label,
                    max_retries=max_retries,
                    fallback_model=fallback_model,
                    verbose=False,
                )

                if retry_errors:
                    errors_encountered.extend(retry_errors)

                entity = convert_dict_to_extraction_entity(entity_dict, entity_type)
                entity_list = getattr(extraction, entity_type)
                entity_list.append(entity)

                successful_extractions += 1

                if verbose:
                    if retry_errors:
                        print(f"✓ (after {len(retry_errors) + 1} attempts)")
                    else:
                        print("✓")

            except Exception as e:
                error_msg = f"Failed to extract {label} after {max_retries} attempts: {e}"
                errors_encountered.append(
                    make_error_dict(
                        error_msg,
                        value={
                            "target_label": label,
                            "entity_type": entity_type,
                            "exception": str(e),
                        },
                    )
                )
                if verbose:
                    print("✗")
                    print(f"      Error: {error_msg}")

    if verbose:
        print(f"  ✓ Extracted {successful_extractions}/{total_labels} labels")

    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            extraction, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

            try:
                from mathster.parsing.text_processing import compare_extraction_with_source

                _, report = compare_extraction_with_source(raw_section, chapter_text)
                print("\n" + "=" * 70)
                print("EXTRACTION REPORT")
                print("=" * 70)
                print(report)
                print("=" * 70 + "\n")
            except Exception as e:
                print(f"  ⚠ Could not generate extraction report: {e}")

        return raw_section, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {e!s}"
        errors_encountered.append(make_error_dict(error_msg, value=extraction.model_dump()))
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, errors_encountered
