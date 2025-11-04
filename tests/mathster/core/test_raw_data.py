"""Comprehensive tests for raw_data module."""

from pathlib import Path

# Direct imports to avoid package-level import issues
# Similar pattern to test_article_system.py
# First, ensure article_system is importable
import sys
import warnings as warnings_module

from pydantic import ValidationError
import pytest


article_system_path = Path(__file__).parents[3] / "src" / "mathster" / "core"
if str(article_system_path) not in sys.path:
    sys.path.insert(0, str(article_system_path))

# Import modules directly without going through mathster package
import article_system
import raw_data


# Import what we need
RawDefinition = raw_data.RawDefinition
normalize_term_to_label = raw_data.normalize_term_to_label
SourceLocation = article_system.SourceLocation
TextLocation = article_system.TextLocation

# Rebuild models to resolve forward references
RawDefinition.model_rebuild()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_source_location():
    """Sample SourceLocation for RawDefinition instances."""
    return SourceLocation(
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        line_range=TextLocation.from_single_range(100, 120),
        label="def-test-term",
        article_id="03_cloning",
    )


# =============================================================================
# TESTS FOR normalize_term_to_label()
# =============================================================================


class TestNormalizeTermToLabel:
    """Tests for the normalize_term_to_label() utility function."""

    def test_simple_term(self):
        """Test normalization of simple two-word term."""
        result = normalize_term_to_label("Lipschitz continuous")
        assert result == "def-lipschitz-continuous"

    def test_term_with_hyphens(self):
        """Test that existing hyphens are preserved."""
        result = normalize_term_to_label("v-porous on balls")
        assert result == "def-v-porous-on-balls"

    def test_term_with_multiple_words(self):
        """Test normalization of multi-word term."""
        result = normalize_term_to_label("Smooth Piecewise Function")
        assert result == "def-smooth-piecewise-function"

    def test_custom_prefix(self):
        """Test normalization with custom prefix."""
        result = normalize_term_to_label("Main Result", prefix="thm")
        assert result == "thm-main-result"

    def test_term_with_special_characters(self):
        """Test that special characters are removed."""
        result = normalize_term_to_label("gamma (friction)", prefix="param")
        assert result == "param-gamma-friction"

    def test_term_with_underscores(self):
        """Test that underscores are converted to hyphens."""
        result = normalize_term_to_label("test_parameter_value")
        assert result == "def-test-parameter-value"

    def test_term_with_multiple_spaces(self):
        """Test that multiple spaces are collapsed."""
        result = normalize_term_to_label("multiple    spaces    here")
        assert result == "def-multiple-spaces-here"

    def test_term_with_trailing_spaces(self):
        """Test that trailing/leading spaces are handled."""
        result = normalize_term_to_label("  spaced term  ")
        assert result == "def-spaced-term"

    def test_empty_term_raises_error(self):
        """Test that empty term raises ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize empty term"):
            normalize_term_to_label("")

    def test_whitespace_only_term_raises_error(self):
        """Test that whitespace-only term raises ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize empty term"):
            normalize_term_to_label("   ")

    def test_term_with_only_special_chars_raises_error(self):
        """Test that term with only special characters raises ValueError."""
        with pytest.raises(ValueError, match="Term normalization produced empty label"):
            normalize_term_to_label("!@#$%^&*()")

    def test_uppercase_conversion(self):
        """Test that uppercase is converted to lowercase."""
        result = normalize_term_to_label("UPPERCASE TERM")
        assert result == "def-uppercase-term"

    def test_mixed_case_conversion(self):
        """Test that mixed case is converted to lowercase."""
        result = normalize_term_to_label("MiXeD CaSe TeRm")
        assert result == "def-mixed-case-term"

    def test_term_with_numbers(self):
        """Test term containing numbers."""
        result = normalize_term_to_label("L2 norm convergence")
        assert result == "def-l2-norm-convergence"

    def test_consecutive_hyphens_collapsed(self):
        """Test that consecutive hyphens are collapsed to single hyphen."""
        result = normalize_term_to_label("term---with---hyphens")
        assert result == "def-term-with-hyphens"


# =============================================================================
# TESTS FOR RawDefinition AUTO-POPULATION
# =============================================================================


class TestRawDefinitionAutoPopulation:
    """Tests for RawDefinition label auto-population and validation."""

    def test_auto_populate_label_from_source(self, sample_source_location):
        """Test that label is auto-populated from source.label when not provided."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Test Term",
            # label not provided - should be auto-populated from source.label
        )

        # source.label is "def-test-term"
        assert definition.label == "def-test-term"

    def test_auto_populate_uses_source_label(self):
        """Test that auto-population uses source.label regardless of term."""
        source = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(100, 120),
            label="def-lipschitz-continuous",
            article_id="03_cloning",
        )

        definition = RawDefinition(
            source=source,
            full_text=TextLocation.from_single_range(100, 120),
            term="Some Other Term",  # Term doesn't matter - uses source.label
        )

        assert definition.label == "def-lipschitz-continuous"

    def test_mismatch_warning_emitted(self, sample_source_location):
        """Test that warning is emitted when user label doesn't match source.label."""
        with pytest.warns(
            UserWarning,
            match=r"label='def-wrong-label'.*source\.label='def-test-term'",
        ):
            definition = RawDefinition(
                source=sample_source_location,
                full_text=TextLocation.from_single_range(100, 120),
                term="Test Term",
                label="def-wrong-label",  # Doesn't match source.label
            )

        # User value should be preserved despite mismatch
        assert definition.label == "def-wrong-label"

    def test_user_label_preserved_on_mismatch(self, sample_source_location):
        """Test that user-provided label is kept even when it doesn't match computed."""
        with pytest.warns(UserWarning):
            definition = RawDefinition(
                source=sample_source_location,
                full_text=TextLocation.from_single_range(100, 120),
                term="Some Term",
                label="def-custom-label",
            )

        assert definition.label == "def-custom-label"

    def test_no_warning_when_labels_match(self, sample_source_location):
        """Test that no warning is emitted when user label matches source.label."""
        # source.label is "def-test-term"
        source_label = sample_source_location.label

        # Create with matching label - should not warn
        with warnings_module.catch_warnings(record=True) as w:
            warnings_module.simplefilter("always")
            definition = RawDefinition(
                source=sample_source_location,
                full_text=TextLocation.from_single_range(100, 120),
                term="Test Term",
                label=source_label,  # Matches source.label
            )

            # Filter to UserWarnings only
            user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]

        assert len(user_warnings) == 0
        assert definition.label == source_label

    def test_invalid_source_label_raises_error(self):
        """Test that source.label not matching def-* pattern raises error."""
        # Create source with invalid label (doesn't start with "def-")
        source = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(100, 120),
            label="thm-wrong-prefix",  # Wrong prefix!
            article_id="03_cloning",
        )

        with pytest.raises(ValueError, match="does not match expected pattern for definitions"):
            RawDefinition(
                source=source,
                full_text=TextLocation.from_single_range(100, 120),
                term="Test Term",
            )

    def test_parameters_mentioned_optional(self, sample_source_location):
        """Test that parameters_mentioned field is optional."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Test Term",
            # parameters_mentioned not provided
        )

        assert definition.parameters_mentioned == []

    def test_parameters_mentioned_populated(self, sample_source_location):
        """Test that parameters_mentioned can be provided."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Test Term",
            parameters_mentioned=["v", "alpha", "h"],
        )

        assert definition.parameters_mentioned == ["v", "alpha", "h"]

    def test_frozen_model(self, sample_source_location):
        """Test that RawDefinition is frozen (immutable)."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Test Term",
        )

        with pytest.raises(ValidationError, match="Instance is frozen"):
            definition.label = "def-new-label"

    def test_various_source_labels(self):
        """Test that different source labels are correctly used."""
        test_cases = [
            "def-lipschitz-continuous",
            "def-v-porous-on-lines",
            "def-smooth-rescale-function",
            "def-l2-norm",
            "def-quasi-stationary-distribution",
        ]

        for source_label in test_cases:
            source = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label=source_label,
                article_id="03_cloning",
            )

            definition = RawDefinition(
                source=source,
                full_text=TextLocation.from_single_range(100, 120),
                term="Any Term",
            )

            assert (
                definition.label == source_label
            ), f"Failed for source_label '{source_label}': got '{definition.label}'"

    def test_warning_includes_source_path(self, sample_source_location):
        """Test that warning message includes source file path for context."""
        with pytest.warns(
            UserWarning,
            match=r"Source:.*03_cloning\.md",
        ):
            RawDefinition(
                source=sample_source_location,
                full_text=TextLocation.from_single_range(100, 120),
                term="Test Term",
                label="def-wrong-label",
            )

    def test_auto_population_preserves_other_fields(self, sample_source_location):
        """Test that auto-population doesn't affect other fields."""
        full_text_loc = TextLocation.from_single_range(100, 120)
        definition = RawDefinition(
            source=sample_source_location,
            full_text=full_text_loc,
            term="Test Term",
            parameters_mentioned=["x", "y", "z"],
        )

        assert definition.full_text == full_text_loc
        assert definition.term == "Test Term"
        assert definition.parameters_mentioned == ["x", "y", "z"]
        assert definition.source == sample_source_location
        assert definition.label == "def-test-term"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestRawDefinitionIntegration:
    """Integration tests for RawDefinition with real-world scenarios."""

    def test_typical_usage_without_label(self, sample_source_location):
        """Test typical usage where user only provides term and content."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Lipschitz continuous",
            parameters_mentioned=["f", "X", "Y"],
        )

        # Label should come from source.label (which is "def-test-term")
        assert definition.label == sample_source_location.label
        assert definition.term == "Lipschitz continuous"
        assert definition.parameters_mentioned == ["f", "X", "Y"]

    def test_backward_compatibility_with_explicit_label(self, sample_source_location):
        """Test backward compatibility when users provide explicit labels."""
        # This should work but warns if label doesn't match source.label
        with pytest.warns(UserWarning, match="source\\.label"):
            definition = RawDefinition(
                source=sample_source_location,
                full_text=TextLocation.from_single_range(100, 120),
                term="Modern Term",
                label="def-legacy-label",  # Doesn't match source.label
            )

        # User-provided label should be preserved
        assert definition.label == "def-legacy-label"

    def test_model_dump_includes_label(self, sample_source_location):
        """Test that model_dump() includes the auto-populated label."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Test Term",
        )

        data = definition.model_dump()
        assert "label" in data
        # Label should match source.label
        assert data["label"] == sample_source_location.label

    def test_model_dump_json_serializable(self, sample_source_location):
        """Test that model can be serialized to JSON."""
        definition = RawDefinition(
            source=sample_source_location,
            full_text=TextLocation.from_single_range(100, 120),
            term="Test Term",
        )

        json_str = definition.model_dump_json()
        # Should include the source.label
        assert sample_source_location.label in json_str
        assert "Test Term" in json_str
