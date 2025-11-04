"""
Tests for label analysis functionality in DSPy extraction workflow.
"""

import sys


sys.path.insert(0, "src")

from mathster.parsing.tools import analyze_labels_in_chapter, classify_label


def test_classify_label_definitions():
    """Test classification of definition labels."""
    assert classify_label("def-lipschitz") == "definitions"
    assert classify_label("def-v-porous-on-balls") == "definitions"
    assert classify_label("def-phase-space-density") == "definitions"
    print("✓ Test 1 passed: Definition labels classified correctly")


def test_classify_label_theorems():
    """Test classification of theorem labels."""
    assert classify_label("thm-main-result") == "theorems"
    assert classify_label("thm-qsd-exchangeability") == "theorems"
    assert classify_label("thm-propagation-chaos-qsd") == "theorems"
    print("✓ Test 2 passed: Theorem labels classified correctly")


def test_classify_label_lemmas():
    """Test classification of lemma labels."""
    assert classify_label("lem-gradient-bound") == "lemmas"
    assert classify_label("lem-conditional-gaussian-qsd-euclidean") == "lemmas"
    print("✓ Test 3 passed: Lemma labels classified correctly")


def test_classify_label_propositions():
    """Test classification of proposition labels."""
    assert classify_label("prop-marginal-mixture") == "propositions"
    assert classify_label("prop-main") == "propositions"
    print("✓ Test 4 passed: Proposition labels classified correctly")


def test_classify_label_corollaries():
    """Test classification of corollary labels."""
    assert classify_label("cor-mean-field-lsi") == "corollaries"
    assert classify_label("cor-main") == "corollaries"
    print("✓ Test 5 passed: Corollary labels classified correctly")


def test_classify_label_axioms():
    """Test classification of axiom labels."""
    assert classify_label("axiom-bounded-diameter") == "axioms"
    assert classify_label("ax-bounded") == "axioms"
    assert classify_label("def-axiom-reward-regularity") == "axioms"
    print("✓ Test 6 passed: Axiom labels classified correctly")


def test_classify_label_assumptions():
    """Test classification of assumption labels."""
    assert classify_label("assumption-bounded-domain") == "assumptions"
    assert classify_label("assumption-lipschitz-regularity") == "assumptions"
    print("✓ Test 7 passed: Assumption labels classified correctly")


def test_classify_label_parameters():
    """Test classification of parameter labels."""
    assert classify_label("param-gamma") == "parameters"
    assert classify_label("param-beta-dynamics") == "parameters"
    print("✓ Test 8 passed: Parameter labels classified correctly")


def test_classify_label_remarks():
    """Test classification of remark labels."""
    assert classify_label("remark-mean-field-cloud") == "remarks"
    assert classify_label("remark-kinetic-necessity") == "remarks"
    print("✓ Test 9 passed: Remark labels classified correctly")


def test_classify_label_proofs():
    """Test classification of proof labels."""
    assert classify_label("proof-main-result") == "proofs"
    assert classify_label("proof-thm-convergence") == "proofs"
    print("✓ Test 10 passed: Proof labels classified correctly")


def test_classify_label_citations():
    """Test classification of citation labels."""
    assert classify_label("cite-han2016") == "citations"
    assert classify_label("cite-villani-2009") == "citations"
    print("✓ Test 11 passed: Citation labels classified correctly")


def test_classify_label_other():
    """Test classification of unknown label types."""
    assert classify_label("unknown-type") == "other"
    assert classify_label("foo-bar") == "other"
    print("✓ Test 12 passed: Unknown labels classified as 'other'")


def test_analyze_labels_empty_chapter():
    """Test analysis with no labels."""
    chapter_text = """
# Chapter 1

This is some text without any labels.

## Section 1.1

More text here.
"""

    labels_dict, report = analyze_labels_in_chapter(chapter_text)

    assert len(labels_dict) == 0
    assert "No explicit :label: directives found" in report
    assert "Generate appropriate labels" in report
    print("✓ Test 13 passed: Empty chapter handled correctly")


def test_analyze_labels_mixed_entities():
    """Test analysis with multiple entity types."""
    chapter_text = """
:::{prf:definition} Lipschitz Continuous
:label: def-lipschitz
:::

:::{prf:theorem} Main Result
:label: thm-main-result
:::

:::{prf:lemma} Gradient Bound
:label: lem-gradient-bound
:::

:::{prf:proposition} Marginal Mixture
:label: prop-marginal
:::

:::{prf:remark} Important Note
:label: remark-note
:::
"""

    labels_dict, report = analyze_labels_in_chapter(chapter_text)

    # Check dictionary structure
    assert "definitions" in labels_dict
    assert "theorems" in labels_dict
    assert "lemmas" in labels_dict
    assert "propositions" in labels_dict
    assert "remarks" in labels_dict

    assert labels_dict["definitions"] == ["def-lipschitz"]
    assert labels_dict["theorems"] == ["thm-main-result"]
    assert labels_dict["lemmas"] == ["lem-gradient-bound"]
    assert labels_dict["propositions"] == ["prop-marginal"]
    assert labels_dict["remarks"] == ["remark-note"]

    # Check report content
    assert "Definitions (1):" in report
    assert "  - def-lipschitz" in report
    assert "Theorems (1):" in report
    assert "  - thm-main-result" in report
    assert "TOTAL: 5 labeled entities" in report
    assert "EXTRACTION INSTRUCTIONS:" in report
    assert "Extract ALL entities listed above using their EXACT labels" in report

    print("✓ Test 14 passed: Mixed entity types analyzed correctly")


def test_analyze_labels_multiple_same_type():
    """Test analysis with multiple labels of the same type."""
    chapter_text = """
:::{prf:definition} First Definition
:label: def-first
:::

:::{prf:definition} Second Definition
:label: def-second
:::

:::{prf:definition} Third Definition
:label: def-third
:::
"""

    labels_dict, report = analyze_labels_in_chapter(chapter_text)

    assert "definitions" in labels_dict
    assert len(labels_dict["definitions"]) == 3
    assert set(labels_dict["definitions"]) == {"def-first", "def-second", "def-third"}

    assert "Definitions (3):" in report
    assert "  - def-first" in report
    assert "  - def-second" in report
    assert "  - def-third" in report
    assert "TOTAL: 3 labeled entities" in report

    print("✓ Test 15 passed: Multiple labels of same type handled correctly")


def test_analyze_labels_with_line_numbers():
    """Test analysis with numbered chapter text."""
    chapter_text = """  1: :::{prf:theorem} Main Result
  2: :label: thm-main
  3: :::
  4:
  5: :::{prf:definition} Important Term
  6: :label: def-term
  7: :::
"""

    labels_dict, report = analyze_labels_in_chapter(chapter_text)

    # Should work the same with line numbers
    assert "theorems" in labels_dict
    assert "definitions" in labels_dict
    assert labels_dict["theorems"] == ["thm-main"]
    assert labels_dict["definitions"] == ["def-term"]
    assert "TOTAL: 2 labeled entities" in report

    print("✓ Test 16 passed: Analysis works with line-numbered text")


def test_analyze_labels_sorting():
    """Test that labels are sorted alphabetically within types."""
    chapter_text = """
:label: def-zebra
:label: def-apple
:label: def-middle
"""

    _labels_dict, report = analyze_labels_in_chapter(chapter_text)

    # Check labels are sorted in report
    lines = report.split("\n")
    def_section_start = next(i for i, line in enumerate(lines) if "Definitions" in line)
    label_lines = []
    for i in range(def_section_start + 1, len(lines)):
        if lines[i].startswith("  - "):
            label_lines.append(lines[i])
        elif lines[i] == "":
            break

    # Labels should be sorted: apple, middle, zebra
    assert label_lines[0] == "  - def-apple"
    assert label_lines[1] == "  - def-middle"
    assert label_lines[2] == "  - def-zebra"

    print("✓ Test 17 passed: Labels sorted alphabetically in report")


def test_analyze_labels_assumptions():
    """Test analysis with new assumption entity type."""
    chapter_text = """
:::{assumption} Bounded Domain
:label: assumption-bounded-domain
:::

:::{assumption} Lipschitz Regularity
:label: assumption-lipschitz-regularity
:::
"""

    labels_dict, report = analyze_labels_in_chapter(chapter_text)

    assert "assumptions" in labels_dict
    assert len(labels_dict["assumptions"]) == 2
    assert "assumption-bounded-domain" in labels_dict["assumptions"]
    assert "assumption-lipschitz-regularity" in labels_dict["assumptions"]

    assert "Assumptions (2):" in report
    assert "  - assumption-bounded-domain" in report
    assert "  - assumption-lipschitz-regularity" in report

    print("✓ Test 18 passed: Assumption labels analyzed correctly")


def main():
    """Run all tests."""
    print("Testing label analysis functionality")
    print("=" * 60)

    try:
        test_classify_label_definitions()
        test_classify_label_theorems()
        test_classify_label_lemmas()
        test_classify_label_propositions()
        test_classify_label_corollaries()
        test_classify_label_axioms()
        test_classify_label_assumptions()
        test_classify_label_parameters()
        test_classify_label_remarks()
        test_classify_label_proofs()
        test_classify_label_citations()
        test_classify_label_other()
        test_analyze_labels_empty_chapter()
        test_analyze_labels_mixed_entities()
        test_analyze_labels_multiple_same_type()
        test_analyze_labels_with_line_numbers()
        test_analyze_labels_sorting()
        test_analyze_labels_assumptions()

        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Summary:")
        print("  - classify_label() works for all 12 entity types")
        print("  - analyze_labels_in_chapter() extracts labels correctly")
        print("  - Report format is LLM-friendly and comprehensive")
        print("  - Handles edge cases (no labels, line numbers, sorting)")
        print("  - New assumption entity type fully supported")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
