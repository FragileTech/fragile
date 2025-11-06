"""
Tests for parameter_extraction.text_processing module.

Tests parameter declaration finding and symbol variant generation.
"""

import pytest

from mathster.parameter_extraction.text_processing import (
    collect_parameters_from_extraction,
    find_parameter_declarations,
)
from mathster.parameter_extraction.text_processing.analysis import _get_symbol_variants


class TestSymbolVariants:
    """Test symbol variant generation."""

    def test_greek_letter_variants(self):
        """Test Greek letter variant generation."""
        variants = _get_symbol_variants("tau")

        assert "tau" in variants
        assert "τ" in variants
        assert "\\tau" in variants

    def test_subscripted_greek_variants(self):
        """Test subscripted Greek letter variants."""
        variants = _get_symbol_variants("gamma_fric")

        assert "gamma_fric" in variants
        assert "\\gamma_{\\mathrm{fric}}" in variants
        assert "γ_fric" in variants

    def test_non_greek_subscripts(self):
        """Test non-Greek subscripted variables."""
        variants = _get_symbol_variants("x_i")

        assert "x_i" in variants
        assert "x_{i}" in variants


class TestParameterDeclarationFinding:
    """Test parameter declaration finding."""

    def test_find_let_be_pattern(self):
        """Test 'Let X be' pattern matching."""
        text = "001: Let τ be the time step size.\n002: We use τ = 0.01."

        declarations = find_parameter_declarations(text, ["tau"])

        assert "tau" in declarations
        assert declarations["tau"]["pattern"] == "let_be"
        assert declarations["tau"]["line_start"] == 1

    def test_find_definition_operator(self):
        """Test ':=' pattern matching."""
        text = "100: where $\\sigma_v := \\sqrt{\\Theta/m}$"

        declarations = find_parameter_declarations(text, ["sigma_v"])

        assert "sigma_v" in declarations
        assert declarations["sigma_v"]["pattern"] == "definition"
        # Line start is based on enumeration (1-indexed from split)
        assert declarations["sigma_v"]["line_start"] == 1

    def test_find_first_latex_mention(self):
        """Test first LaTeX mention fallback."""
        text = "50: The parameter $\\alpha$ controls the rate.\n51: More text."

        declarations = find_parameter_declarations(text, ["alpha"])

        assert "alpha" in declarations
        # Line number is 1 because it's the first line in the text array
        assert declarations["alpha"]["line_start"] >= 1


class TestParameterCollection:
    """Test parameter collection from extractions."""

    def test_collect_from_definitions(self):
        """Test collecting parameters mentioned in definitions."""
        extraction = {
            "definitions": [
                {"parameters_mentioned": ["alpha", "beta"]},
                {"parameters_mentioned": ["tau"]},
            ],
            "theorems": [],
        }

        params = collect_parameters_from_extraction(extraction)

        assert "alpha" in params
        assert "beta" in params
        assert "tau" in params
        assert len(params) == 3

    def test_collect_from_multiple_entity_types(self):
        """Test collecting from definitions and theorems."""
        extraction = {
            "definitions": [{"parameters_mentioned": ["alpha"]}],
            "theorems": [{"parameters_mentioned": ["beta", "alpha"]}],
        }

        params = collect_parameters_from_extraction(extraction)

        assert "alpha" in params
        assert "beta" in params
        assert len(params) == 2  # Deduplicated
