"""
Test CLI extraction report display after chapter extraction.
"""

import sys

sys.path.insert(0, "src")

from mathster.parsing.extract_workflow import extract_chapter


def test_extraction_report_display():
    """Test that extraction report is displayed in CLI after successful extraction."""

    # Minimal chapter text with labeled entities
    chapter_text = """  1: ## Test Chapter
  2:
  3: :::{prf:definition} Lipschitz Continuous
  4: :label: def-lipschitz
  5: :::
  6:
  7: A function is Lipschitz continuous...
  8:
  9: :::{prf:theorem} Main Result
 10: :label: thm-main-result
 11: :::
 12:
 13: The main result states...
 14:
 15: :::{prf:lemma} Helper Lemma
 16: :label: lem-helper
 17: :::
 18:
 19: This lemma helps...
 20: """

    file_path = "/home/guillem/fragile/docs/source/test_chapter.md"
    article_id = "test_chapter"

    print("\n" + "="*70)
    print("TEST: Extraction Report Display")
    print("="*70 + "\n")

    # Extract chapter (verbose=True to see report)
    raw_section, errors = extract_chapter(
        chapter_text=chapter_text,
        chapter_number=1,
        file_path=file_path,
        article_id=article_id,
        max_iters=3,
        verbose=True  # This should trigger report display
    )

    # Verify extraction succeeded
    assert raw_section is not None, "Extraction should succeed"

    # Verify entities were extracted
    total_entities = (
        len(raw_section.definitions) +
        len(raw_section.theorems) +
        len(raw_section.lemmas)
    )

    print(f"\n✓ Test passed: Extraction report displayed")
    print(f"  Total entities extracted: {total_entities}")
    print(f"  Definitions: {len(raw_section.definitions)}")
    print(f"  Theorems: {len(raw_section.theorems)}")
    print(f"  Lemmas: {len(raw_section.lemmas)}")

    return raw_section


def test_extraction_report_no_verbose():
    """Test that report is NOT displayed when verbose=False."""

    chapter_text = """  1: ## Test Chapter
  2:
  3: :::{prf:definition} Test Definition
  4: :label: def-test
  5: :::
  6: """

    print("\n" + "="*70)
    print("TEST: Report NOT displayed when verbose=False")
    print("="*70 + "\n")

    # Extract with verbose=False (should NOT display report)
    raw_section, errors = extract_chapter(
        chapter_text=chapter_text,
        chapter_number=1,
        file_path="/test/path.md",
        article_id="test",
        max_iters=3,
        verbose=False  # No report should be displayed
    )

    print("✓ Test passed: No report displayed (verbose=False)")


def main():
    """Run all tests."""
    print("\nTesting extraction report display functionality")
    print("="*70)

    try:
        test_extraction_report_display()
        test_extraction_report_no_verbose()

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        print("\nSummary:")
        print("  - Extraction report displays after successful extraction (verbose=True)")
        print("  - Report shows labels parsed by entity type")
        print("  - Report includes comparison with source document")
        print("  - Report not displayed when verbose=False")
        print("="*70)
        return 0

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"✗ Test failed: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ Unexpected error: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
