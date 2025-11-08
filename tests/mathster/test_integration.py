"""
Integration tests for complete mathster pipeline.

Tests that all modules work together correctly.
"""

import pytest


class TestModuleImports:
    """Test that all refactored modules import correctly."""

    def test_dspy_integration_imports(self):
        """Test dspy_integration module imports."""
        from mathster.dspy_integration import configure_dspy, make_error_dict
        from mathster.dspy_integration.text_utils import sanitize_label, split_markdown_by_chapters

        assert callable(configure_dspy)
        assert callable(make_error_dict)
        assert callable(sanitize_label)
        assert callable(split_markdown_by_chapters)

    def test_parameter_extraction_imports(self):
        """Test parameter_extraction module imports."""
        from mathster.parameter_extraction import extract_parameters, refine_parameters
        from mathster.parameter_extraction.text_processing import find_parameter_declarations
        from mathster.parameter_extraction.validation import validate_parameter
        from mathster.parameter_extraction.conversion import convert_parameter

        assert callable(extract_parameters)
        assert callable(refine_parameters)
        assert callable(find_parameter_declarations)
        assert callable(validate_parameter)
        assert callable(convert_parameter)

    def test_parsing_imports(self):
        """Test parsing module imports."""
        from mathster.parsing import extract_chapter, improve_chapter
        from mathster.directives import extract_directive_hints

        assert callable(extract_chapter)
        assert callable(improve_chapter)
        assert callable(extract_directive_hints)


class TestParameterExtractionPipeline:
    """Test complete parameter extraction pipeline."""

    def test_symbol_variant_generation(self):
        """Test that symbol variants are generated correctly."""
        from mathster.parameter_extraction.text_processing.analysis import _get_symbol_variants

        variants = _get_symbol_variants("tau")
        assert "tau" in variants
        assert "τ" in variants
        assert "\\tau" in variants

    def test_parameter_collection(self):
        """Test collecting parameters from extraction."""
        from mathster.parameter_extraction.text_processing import collect_parameters_from_extraction

        extraction = {"definitions": [{"parameters_mentioned": ["alpha", "beta"]}]}
        params = collect_parameters_from_extraction(extraction)

        assert "alpha" in params
        assert "beta" in params


class TestDirectivePreprocessing:
    """Test directive preprocessing integration."""

    def test_extract_hints_from_text(self):
        """Test extracting directive hints."""
        from mathster.directives import extract_directive_hints

        text = """:::{prf:theorem} Test
:label: thm-test

Statement.
:::"""

        hints = extract_directive_hints(text, chapter_number=0)

        assert len(hints) == 1
        assert hints[0]["directive_type"] == "theorem"
        assert hints[0]["label"] == "thm-test"
        assert hints[0]["title"] == "Test"


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_import_chain(self):
        """Test complete import chain across all modules."""
        # This test verifies no circular imports or missing dependencies

        from mathster.dspy_integration import configure_dspy
        from mathster.parsing import extract_chapter
        from mathster.parameter_extraction import extract_parameters

        # If we get here without ImportError, the refactoring is successful
        assert True
