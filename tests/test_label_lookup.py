"""
Tests for label lookup functionality in DSPy extraction workflow.
"""

import sys


sys.path.insert(0, "src")

from mathster.parsing.extract_workflow import lookup_label_from_context


def test_lookup_label_finds_directive():
    """Test that lookup finds :label: directive in context."""
    context = """
  100: :::{prf:theorem} Convergence Result
  101: :label: thm-convergence
  102:
  103: The algorithm converges.
  104: :::
  105:
  106: By Theorem 1.4, we have convergence.
"""
    label = lookup_label_from_context("Theorem 1.4", context, "theorem")
    assert label == "thm-convergence", f"Expected 'thm-convergence', got '{label}'"
    print("✓ Test 1 passed: Found :label: directive near reference")


def test_lookup_label_generates_from_text():
    """Test that lookup generates label when no directive found."""
    context = "Some text without labels mentioning Theorem 1.4"
    label = lookup_label_from_context("Theorem 1.4", context, "theorem")
    assert label == "thm-1-4", f"Expected 'thm-1-4', got '{label}'"
    print("✓ Test 2 passed: Generated label from text when no directive found")


def test_lookup_label_definition():
    """Test definition label generation."""
    context = "Some text without labels"
    label = lookup_label_from_context("Lipschitz continuous", context, "definition")
    assert (
        label == "def-lipschitz-continuous"
    ), f"Expected 'def-lipschitz-continuous', got '{label}'"
    print("✓ Test 3 passed: Generated definition label")


def test_lookup_label_lemma():
    """Test lemma label generation."""
    context = "Some text mentioning Lemma 2.3"
    label = lookup_label_from_context("Lemma 2.3", context, "theorem")
    assert label == "lem-2-3", f"Expected 'lem-2-3', got '{label}'"
    print("✓ Test 4 passed: Generated lemma label")


def test_lookup_label_with_line_numbers():
    """Test that lookup handles line numbers correctly."""
    context = """
  150: :::{prf:definition} Lipschitz Continuous
  151: :label: def-lipschitz
  152:
  153: A function is Lipschitz continuous if...
  154: :::
  155:
  156: We use Lipschitz continuous functions.
"""
    label = lookup_label_from_context("Lipschitz continuous", context, "definition")
    # Should find def-lipschitz in the context
    assert label in {
        "def-lipschitz",
        "def-lipschitz-continuous",
    }, f"Expected definition label, got '{label}'"
    print(f"✓ Test 5 passed: Found definition label: {label}")


def test_lookup_label_proposition():
    """Test proposition label generation."""
    context = "Some text mentioning Proposition 3.1"
    label = lookup_label_from_context("Proposition 3.1", context, "theorem")
    assert label == "prop-3-1", f"Expected 'prop-3-1', got '{label}'"
    print("✓ Test 6 passed: Generated proposition label")


def test_lookup_label_corollary():
    """Test corollary label generation."""
    context = "Some text mentioning Corollary 4.2"
    label = lookup_label_from_context("Corollary 4.2", context, "theorem")
    assert label == "cor-4-2", f"Expected 'cor-4-2', got '{label}'"
    print("✓ Test 7 passed: Generated corollary label")


def main():
    """Run all tests."""
    print("Testing label lookup functionality")
    print("=" * 60)

    try:
        test_lookup_label_finds_directive()
        test_lookup_label_generates_from_text()
        test_lookup_label_definition()
        test_lookup_label_lemma()
        test_lookup_label_with_line_numbers()
        test_lookup_label_proposition()
        test_lookup_label_corollary()

        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
