#!/usr/bin/env python3
"""
Test script for convert_unicode_math.py
Verifies that Unicode to LaTeX conversions are correct.
"""

from tools.convert_unicode_math import convert_unicode_to_latex, UNICODE_TO_LATEX


def test_individual_conversions():
    """Test each Unicode symbol converts correctly."""
    print("Testing individual symbol conversions...")
    failures = []

    for unicode_char, expected_latex in UNICODE_TO_LATEX.items():
        result = convert_unicode_to_latex(unicode_char)
        if result != expected_latex:
            failures.append(f"  {repr(unicode_char)} → {repr(result)} (expected {repr(expected_latex)})")

    if failures:
        print("❌ FAILED:")
        for f in failures:
            print(f)
        return False
    else:
        print(f"✓ All {len(UNICODE_TO_LATEX)} symbols convert correctly")
        return True


def test_compound_expressions():
    """Test compound mathematical expressions."""
    print("\nTesting compound expressions...")
    test_cases = [
        # (input, expected_output, description)
        ("(d')^β", "(d')^\\beta", "exponent with Greek letter"),
        ("ε_d > 0", "\\varepsilon_d > 0", "epsilon with subscript"),
        ("||v_i||²", "||v_i||^{2}", "norm with superscript 2"),
        ("c_{v\_reg}", "c_{v\_reg}", "subscript (no Unicode)"),
        ("α ∈ (0, 1)", "\\alpha \\in (0, 1)", "Greek with set membership"),
        ("x² + y³", "x^{2} + y^{3}", "multiple superscripts"),
        ("a ≈ b ≤ c ≥ d", "a \\approx b \\leq c \\geq d", "comparison operators"),
        ("γ × κ ÷ σ", "\\gamma \\times \\kappa \\div \\sigma", "arithmetic with Greek"),
        ("∑_{i=1}^n", "\\sum_{i=1}^n", "sum notation"),
        ("Var(x) > R²_var", "Var(x) > R^{2}_var", "variance with superscript"),
        ("α_restitution ∈ [0, 1]", "\\alpha_restitution \\in [0, 1]", "alpha with subscript"),
        ("Ψ_clone", "\\Psi_clone", "uppercase Psi"),
        ("μ₁ ∪ μ₂", "\\mu₁ \\cup \\mu₂", "union (note: subscript not converted)"),
        ("D_diam(ε)/2", "D_diam(\\varepsilon)/2", "epsilon in function"),
    ]

    failures = []
    for input_str, expected, description in test_cases:
        result = convert_unicode_to_latex(input_str)
        if result != expected:
            failures.append(f"  {description}:")
            failures.append(f"    Input:    {repr(input_str)}")
            failures.append(f"    Got:      {repr(result)}")
            failures.append(f"    Expected: {repr(expected)}")

    if failures:
        print("❌ FAILED:")
        for f in failures:
            print(f)
        return False
    else:
        print(f"✓ All {len(test_cases)} compound expressions convert correctly")
        return True


def test_preserves_non_unicode():
    """Test that non-Unicode content is preserved."""
    print("\nTesting preservation of non-Unicode content...")
    test_cases = [
        "This is plain text",
        "def-d-state-space",
        "eq-virtual-reward",
        "c_{v\_reg}",
        "V_Var,x",
        "`code block`",
        "$x^{2}$",
        "\\alpha \\beta \\gamma",  # Already LaTeX
    ]

    failures = []
    for test_str in test_cases:
        result = convert_unicode_to_latex(test_str)
        if result != test_str:
            failures.append(f"  Changed: {repr(test_str)} → {repr(result)}")

    if failures:
        print("❌ FAILED - these should not have changed:")
        for f in failures:
            print(f)
        return False
    else:
        print(f"✓ All {len(test_cases)} non-Unicode strings preserved")
        return True


def test_in_context():
    """Test conversions in realistic markdown context."""
    print("\nTesting in markdown context...")
    test_cases = [
        (
            "The fitness potential becomes dominated by the diversity term (`(d')^β`).",
            "The fitness potential becomes dominated by the diversity term (`(d')^\\beta`).",
            "inline code with Greek"
        ),
        (
            "where `ε_d > 0` (The Interaction Range for Diversity).",
            "where `\\varepsilon_d > 0` (The Interaction Range for Diversity).",
            "inline code with epsilon"
        ),
        (
            "The `-c_{v\_reg} ||v_i||²` term gives this walker",
            "The `-c_{v\_reg} ||v_i||^{2}` term gives this walker",
            "inline code with superscript"
        ),
        (
            "If **`α_restitution = 1`**, the collision is **perfectly elastic**.",
            "If **`\\alpha_restitution = 1`**, the collision is **perfectly elastic**.",
            "bold with inline code and Greek"
        ),
    ]

    failures = []
    for input_str, expected, description in test_cases:
        result = convert_unicode_to_latex(input_str)
        if result != expected:
            failures.append(f"  {description}:")
            failures.append(f"    Got:      {repr(result)}")
            failures.append(f"    Expected: {repr(expected)}")

    if failures:
        print("❌ FAILED:")
        for f in failures:
            print(f)
        return False
    else:
        print(f"✓ All {len(test_cases)} contextual tests pass")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing convert_unicode_math.py")
    print("=" * 60)

    results = [
        test_individual_conversions(),
        test_compound_expressions(),
        test_preserves_non_unicode(),
        test_in_context(),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
