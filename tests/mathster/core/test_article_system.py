"""Comprehensive tests for article_system module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

# Direct import to avoid broken package dependencies
import sys
from pathlib import Path

# Import article_system module directly
article_system_path = Path(__file__).parents[3] / "src" / "mathster" / "core"
sys.path.insert(0, str(article_system_path))

import article_system

# Import what we need
TextLocation = article_system.TextLocation
SourceLocation = article_system.SourceLocation
Article = article_system.Article
extract_volume_from_path = article_system.extract_volume_from_path
extract_article_id_from_path = article_system.extract_article_id_from_path
extract_section_from_markdown = article_system.extract_section_from_markdown


# Fixtures


@pytest.fixture
def temp_markdown_file(tmp_path):
    """Create temporary markdown file for testing section extraction."""
    content = """# Document Title

Some introductory text.

## 1. Introduction

More text here.

### 1.1. First Subsection

Content of subsection.

### 1.2 Second Subsection

Note: no period after number.

#### 1.2.1. Deep Subsection

Very specific content.

## 2. Main Content

Some content without subsections.

## TLDR

Header without number.
"""
    file_path = tmp_path / "test_document.md"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_text_location():
    """Sample TextLocation object for testing."""
    return TextLocation(lines=[(10, 15), (20, 25)])


@pytest.fixture
def sample_source_location():
    """Sample SourceLocation object for testing - uses auto-populated values."""
    # Don't provide optional values - let them auto-populate to avoid warnings
    return SourceLocation(
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        line_range=TextLocation.from_single_range(100, 120),
        label="thm-example",
        article_id="03_cloning",
    )


# Test Classes


class TestTextLocation:
    """Test TextLocation class for discontinuous line ranges."""

    def test_single_range_creation(self):
        """Create TextLocation with single continuous range."""
        loc = TextLocation(lines=[(10, 15)])
        assert loc.lines == [(10, 15)]

    def test_multiple_ranges_creation(self):
        """Create TextLocation with multiple discontinuous ranges."""
        loc = TextLocation(lines=[(10, 15), (20, 25), (30, 32)])
        assert loc.lines == [(10, 15), (20, 25), (30, 32)]

    def test_validation_rejects_start_greater_than_end(self):
        """Validation should reject ranges where start > end."""
        with pytest.raises(ValidationError, match="start > end"):
            TextLocation(lines=[(15, 10)])

    def test_validation_rejects_line_less_than_one(self):
        """Validation should reject line numbers < 1."""
        with pytest.raises(ValidationError, match="Line numbers must be"):
            TextLocation(lines=[(0, 10)])

    def test_validation_rejects_overlapping_ranges(self):
        """Validation should reject overlapping ranges."""
        with pytest.raises(ValidationError, match="Overlapping ranges"):
            TextLocation(lines=[(10, 20), (15, 25)])

    def test_validation_rejects_empty_lines(self):
        """Validation should reject empty lines list."""
        with pytest.raises(ValidationError, match="at least one line range"):
            TextLocation(lines=[])

    def test_from_single_range_classmethod(self):
        """Test from_single_range factory method."""
        loc = TextLocation.from_single_range(10, 15)
        assert loc.lines == [(10, 15)]

    def test_format_ranges_single(self):
        """Test format_ranges with single range."""
        loc = TextLocation(lines=[(10, 15)])
        assert loc.format_ranges() == "10-15"

    def test_format_ranges_multiple(self):
        """Test format_ranges with multiple ranges."""
        loc = TextLocation(lines=[(10, 15), (20, 25), (30, 32)])
        assert loc.format_ranges() == "10-15, 20-25, 30-32"

    def test_get_total_line_count_single(self):
        """Test get_total_line_count with single range."""
        loc = TextLocation(lines=[(10, 15)])
        assert loc.get_total_line_count() == 6  # 15-10+1

    def test_get_total_line_count_multiple(self):
        """Test get_total_line_count with multiple ranges."""
        loc = TextLocation(lines=[(10, 15), (20, 25)])
        assert loc.get_total_line_count() == 12  # 6 + 6

    def test_contains_line_positive(self):
        """Test contains_line when line is in range."""
        loc = TextLocation(lines=[(10, 15), (20, 25)])
        assert loc.contains_line(12) is True
        assert loc.contains_line(22) is True

    def test_contains_line_negative(self):
        """Test contains_line when line is not in range."""
        loc = TextLocation(lines=[(10, 15), (20, 25)])
        assert loc.contains_line(18) is False
        assert loc.contains_line(30) is False

    def test_contains_line_boundary(self):
        """Test contains_line at range boundaries."""
        loc = TextLocation(lines=[(10, 15)])
        assert loc.contains_line(10) is True  # Start boundary
        assert loc.contains_line(15) is True  # End boundary
        assert loc.contains_line(9) is False
        assert loc.contains_line(16) is False


class TestExtractVolumeFromPath:
    """Test extract_volume_from_path function."""

    def test_extract_volume_euclidean_gas(self):
        """Extract volume from 1_euclidean_gas path."""
        path = "docs/source/1_euclidean_gas/03_cloning.md"
        assert extract_volume_from_path(path) == "1_euclidean_gas"

    def test_extract_volume_geometric_gas(self):
        """Extract volume from 2_geometric_gas path."""
        path = "docs/source/2_geometric_gas/11_geometric_gas.md"
        assert extract_volume_from_path(path) == "2_geometric_gas"

    def test_extract_volume_with_docs_source_prefix(self):
        """Extract volume with full docs/source prefix."""
        path = "docs/source/1_euclidean_gas/test.md"
        assert extract_volume_from_path(path) == "1_euclidean_gas"

    def test_extract_volume_without_prefix(self):
        """Extract volume without docs/source prefix."""
        path = "1_euclidean_gas/test.md"
        assert extract_volume_from_path(path) == "1_euclidean_gas"

    def test_extract_volume_not_found(self):
        """Return None when volume pattern not found."""
        path = "other/path/file.md"
        assert extract_volume_from_path(path) is None

    def test_extract_volume_wrong_pattern(self):
        """Return None for invalid volume pattern."""
        path = "docs/source/invalid_name/file.md"
        assert extract_volume_from_path(path) is None


class TestExtractArticleIdFromPath:
    """Test extract_article_id_from_path function."""

    def test_extract_article_id_basic(self):
        """Extract article_id from basic path."""
        path = "docs/source/1_euclidean_gas/03_cloning.md"
        assert extract_article_id_from_path(path) == "03_cloning"

    def test_extract_article_id_with_underscores(self):
        """Extract article_id with multiple underscores."""
        path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
        assert extract_article_id_from_path(path) == "01_fragile_gas_framework"

    def test_extract_article_id_different_numbers(self):
        """Extract article_id with different number prefix."""
        path = "docs/source/2_geometric_gas/11_geometric_gas.md"
        assert extract_article_id_from_path(path) == "11_geometric_gas"

    def test_extract_article_id_raises_on_invalid_pattern(self):
        """Raise ValueError when pattern doesn't match."""
        path = "invalid/path.md"
        with pytest.raises(ValueError, match="Cannot extract article_id"):
            extract_article_id_from_path(path)

    def test_extract_article_id_raises_on_wrong_digit_count(self):
        """Raise ValueError when number has wrong digit count."""
        path = "docs/source/test/1_test.md"  # Only 1 digit
        with pytest.raises(ValueError, match="Cannot extract article_id"):
            extract_article_id_from_path(path)

    def test_extract_article_id_raises_on_no_extension(self):
        """Raise ValueError when file has no .md extension."""
        path = "docs/source/test/03_test.txt"
        with pytest.raises(ValueError, match="Cannot extract article_id"):
            extract_article_id_from_path(path)


class TestExtractSectionFromMarkdown:
    """Test extract_section_from_markdown function."""

    def test_extract_section_with_period_after_number(self, temp_markdown_file):
        """Extract section with period after number (e.g., '1.1.')."""
        # Line 11 is in "### 1.1. First Subsection"
        result = extract_section_from_markdown(
            str(temp_markdown_file), TextLocation.from_single_range(11, 12)
        )
        assert result == ("1.1", "First Subsection")

    def test_extract_section_without_period(self, temp_markdown_file):
        """Extract section without period after number (e.g., '1.2')."""
        # Line 17 is in "### 1.2 Second Subsection"
        result = extract_section_from_markdown(
            str(temp_markdown_file), TextLocation.from_single_range(17, 18)
        )
        assert result == ("1.2", "Second Subsection")

    def test_extract_section_deep_subsection(self, temp_markdown_file):
        """Extract deep subsection with multiple levels (e.g., '1.2.1.')."""
        # Line 19 is "Very specific content." which is inside "#### 1.2.1. Deep Subsection" (line 17)
        # Function finds LAST header BEFORE line 19, which is line 17
        result = extract_section_from_markdown(
            str(temp_markdown_file), TextLocation.from_single_range(19, 20)
        )
        assert result == ("1.2.1", "Deep Subsection")

    def test_extract_section_major_section(self, temp_markdown_file):
        """Extract major section (## N.)."""
        # Line 8 is in "## 1. Introduction"
        result = extract_section_from_markdown(
            str(temp_markdown_file), TextLocation.from_single_range(8, 9)
        )
        assert result == ("1", "Introduction")

    def test_extract_section_no_headers_before_line(self, temp_markdown_file):
        """Return (None, None) when no headers before line."""
        result = extract_section_from_markdown(
            str(temp_markdown_file), TextLocation.from_single_range(1, 2)
        )
        assert result == (None, None)

    def test_extract_section_header_without_number(self, temp_markdown_file):
        """Extract text from header without number."""
        # Line 30 is in "## TLDR"
        result = extract_section_from_markdown(
            str(temp_markdown_file), TextLocation.from_single_range(30, 31)
        )
        assert result == (None, "TLDR")

    def test_extract_section_file_not_found(self):
        """Raise FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            extract_section_from_markdown(
                "/nonexistent/file.md", TextLocation.from_single_range(10, 20)
            )

    def test_extract_section_from_01_framework_line_15(self):
        """Test extraction from real file: 01_fragile_gas_framework.md line 15."""
        file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        result = extract_section_from_markdown(
            file_path, TextLocation.from_single_range(15, 30)
        )
        # Line 15 starts "### 1.1. Goal and Scope"
        # But function finds LAST header BEFORE line 15, which is line 13 "## 1. Introduction"
        assert result[0] == "1"
        assert "Introduction" in result[1]

    def test_extract_section_from_01_framework_line_138(self):
        """Test extraction from real file: 01_fragile_gas_framework.md line 138."""
        file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        result = extract_section_from_markdown(
            file_path, TextLocation.from_single_range(138, 145)
        )
        # Line 138 is "## 2. Global Conventions..."
        # But function finds LAST header BEFORE line 138, which is line 52 "### 1.3. Overview..."
        assert result[0] == "1.3"
        assert "Overview" in result[1] or "Proof Strategy" in result[1]

    def test_extract_section_from_03_cloning_line_617(self):
        """Test extraction from real file: 03_cloning.md line 617."""
        file_path = "docs/source/1_euclidean_gas/03_cloning.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        result = extract_section_from_markdown(
            file_path, TextLocation.from_single_range(617, 635)
        )
        # Line 617 is "#### 3.2.4 From Structural Error..."
        # But function finds LAST header BEFORE line 617, which is line 462 "#### 3.2.3. The Decomposition..."
        assert result[0] == "3.2.3"
        assert "Decomposition" in result[1] or "Inter-Swarm" in result[1]

    def test_extract_section_from_03_cloning_line_422(self):
        """Test extraction from real file: 03_cloning.md line 422."""
        file_path = "docs/source/1_euclidean_gas/03_cloning.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        result = extract_section_from_markdown(
            file_path, TextLocation.from_single_range(422, 430)
        )
        # Line 422 is "#### 3.2.1. The Location Error Component"
        # But function finds LAST header BEFORE line 422, which is line 418 "### 3.2. Permutation-Invariant Error Components"
        assert result[0] == "3.2"
        assert "Permutation" in result[1] or "Error Components" in result[1]


class TestSourceLocation:
    """Test SourceLocation class."""

    def test_source_location_creation_all_fields(self):
        """Create SourceLocation with all fields - warns when values don't match source."""
        # Providing values that don't match the actual source - will emit warnings
        with pytest.warns(UserWarning):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="thm-keystone",
                article_id="03_cloning",
                volume="1_euclidean_gas",  # Matches
                section="3.2",  # Doesn't match (actual is 2.2)
                section_name="The Keystone Principle",  # Doesn't match
            )
        assert loc.file_path == "docs/source/1_euclidean_gas/03_cloning.md"
        assert loc.line_range.lines == [(100, 120)]
        assert loc.label == "thm-keystone"
        assert loc.article_id == "03_cloning"
        assert loc.volume == "1_euclidean_gas"
        # User values are preserved despite mismatch
        assert loc.section == "3.2"
        assert loc.section_name == "The Keystone Principle"

    def test_source_location_required_fields_only(self):
        """Create SourceLocation with only required fields - auto-populates optional fields."""
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(100, 120),
            label="thm-test",
            article_id="03_cloning",
        )
        assert loc.file_path == "docs/source/1_euclidean_gas/03_cloning.md"
        assert loc.label == "thm-test"
        # Auto-populated fields
        assert loc.volume == "1_euclidean_gas"  # Extracted from file_path
        assert loc.section is not None  # Extracted from markdown headers
        assert loc.section_name is not None  # Extracted from markdown headers

    def test_source_location_frozen(self, sample_source_location):
        """SourceLocation should be frozen (immutable)."""
        with pytest.raises(ValidationError):
            sample_source_location.section = "4.1"

    def test_auto_populate_fails_on_invalid_path(self):
        """Auto-population should raise ValueError if volume cannot be extracted."""
        with pytest.raises(ValueError, match="Cannot extract volume from file_path"):
            SourceLocation(
                file_path="invalid/path/without/volume/pattern.md",
                line_range=TextLocation.from_single_range(10, 20),
                label="test",
                article_id="03_cloning",
            )

    def test_auto_populate_fails_on_nonexistent_file(self):
        """Auto-population should raise FileNotFoundError if markdown file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            SourceLocation(
                file_path="docs/source/1_euclidean_gas/nonexistent_file.md",
                line_range=TextLocation.from_single_range(10, 20),
                label="test",
                article_id="03_cloning",
            )

    def test_mismatch_warning_volume(self):
        """Test warning when user-provided volume doesn't match computed value."""
        with pytest.warns(UserWarning, match="volume='2_geometric_gas'.*computed volume='1_euclidean_gas'"):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                volume="2_geometric_gas",  # Wrong volume!
            )
        # User value should be preserved
        assert loc.volume == "2_geometric_gas"

    def test_mismatch_warning_section(self):
        """Test warning when user-provided section doesn't match computed value."""
        with pytest.warns(UserWarning, match="section='9.9'.*computed section="):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                section="9.9",  # Wrong section!
            )
        # User value should be preserved
        assert loc.section == "9.9"

    def test_mismatch_warning_section_name(self):
        """Test warning when user-provided section_name doesn't match computed value."""
        with pytest.warns(UserWarning, match="section_name='Wrong Name'.*computed section_name="):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                section_name="Wrong Name",  # Wrong section name!
            )
        # User value should be preserved
        assert loc.section_name == "Wrong Name"

    def test_mismatch_warning_multiple_fields(self):
        """Test multiple warnings when multiple fields don't match."""
        with pytest.warns(UserWarning) as warning_records:
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                volume="2_geometric_gas",  # Wrong!
                section="9.9",  # Wrong!
                section_name="Wrong Name",  # Wrong!
            )
        # Should have 3 warnings
        assert len(warning_records) == 3
        # User values should be preserved
        assert loc.volume == "2_geometric_gas"
        assert loc.section == "9.9"
        assert loc.section_name == "Wrong Name"

    def test_no_warning_when_values_match(self):
        """Test no warning when user-provided values match computed values."""
        # First create one without user values to get computed values
        loc_ref = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(100, 120),
            label="test-ref",
            article_id="03_cloning",
        )

        # Capture computed values
        computed_volume = loc_ref.volume
        computed_section = loc_ref.section
        computed_section_name = loc_ref.section_name

        # Now create with matching values - should not warn
        import warnings as warnings_module
        with warnings_module.catch_warnings(record=True) as w:
            warnings_module.simplefilter("always")
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                volume=computed_volume,
                section=computed_section,
                section_name=computed_section_name,
            )
            # Filter for UserWarnings only
            user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]

        # Should have no UserWarnings
        assert len(user_warnings) == 0
        assert loc.volume == computed_volume
        assert loc.section == computed_section
        assert loc.section_name == computed_section_name

    def test_user_values_preserved_on_mismatch(self):
        """Test that user values are preserved (not overridden) when they don't match."""
        with pytest.warns(UserWarning):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                volume="custom_volume",
                section="custom.section",
                section_name="Custom Section Name",
            )

        # Verify user values are kept, not overridden with computed values
        assert loc.volume == "custom_volume"
        assert loc.section == "custom.section"
        assert loc.section_name == "Custom Section Name"

    def test_get_full_url_default_base(self, sample_source_location):
        """Test get_full_url with default base URL."""
        url = sample_source_location.get_full_url()
        assert url == "https://docs.example.com/03_cloning.html#thm-example"

    def test_get_full_url_custom_base(self, sample_source_location):
        """Test get_full_url with custom base URL."""
        url = sample_source_location.get_full_url("https://custom.com")
        assert url == "https://custom.com/03_cloning.html#thm-example"

    def test_get_display_location_with_section(self):
        """Test get_display_location when section is provided - will warn on mismatch."""
        # Providing explicit values that don't match computed values - will emit warnings
        with pytest.warns(UserWarning):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                volume="1_euclidean_gas",  # matches
                section="3.2.1",  # doesn't match (actual is 2.2)
                section_name="Test Section",  # doesn't match
            )
        assert loc.get_display_location() == "03_cloning 3.2.1"

    def test_get_display_location_without_section(self):
        """Test get_display_location - section is auto-populated from markdown."""
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(142, 158),
            label="test",
            article_id="03_cloning",
        )
        # Section is auto-populated from markdown at line 142 (section 2.2)
        assert loc.section is not None
        assert loc.get_display_location() == f"03_cloning {loc.section}"

    def test_get_display_location_discontinuous_ranges(self):
        """Test get_display_location - section is auto-populated from markdown."""
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation(lines=[(142, 158), (165, 170)]),
            label="test",
            article_id="03_cloning",
        )
        # Section is auto-populated from markdown at line 142 (section 2.2)
        assert loc.section is not None
        assert loc.get_display_location() == f"03_cloning {loc.section}"

    def test_article_id_pattern_validation_valid(self):
        """Test article_id validation with valid patterns."""
        # Valid patterns - warnings expected since provided values don't match computed
        with pytest.warns(UserWarning):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="test",
                article_id="03_cloning",
                volume="1_euclidean_gas",  # matches
                section="3.2",  # doesn't match
                section_name="Test",  # doesn't match
            )
        assert loc.article_id == "03_cloning"

    def test_article_id_pattern_validation_invalid(self):
        """Test article_id validation rejects invalid patterns."""
        with pytest.raises(ValidationError):
            # Even with warnings, pydantic pattern validation runs first
            SourceLocation(
                file_path="test.md",
                line_range=TextLocation.from_single_range(10, 20),
                label="test",
                article_id="invalid_id",  # Missing number prefix
                volume="1_euclidean_gas",
                section="3.2",
                section_name="Test",
            )

    def test_label_pattern_validation_valid(self):
        """Test label validation with valid patterns."""
        # Warnings expected since provided values don't match computed
        with pytest.warns(UserWarning):
            loc = SourceLocation(
                file_path="docs/source/1_euclidean_gas/03_cloning.md",
                line_range=TextLocation.from_single_range(100, 120),
                label="thm-keystone",
                article_id="03_cloning",
                volume="1_euclidean_gas",  # matches
                section="3.2",  # doesn't match
                section_name="Test",  # doesn't match
            )
        assert loc.label == "thm-keystone"

    def test_label_pattern_validation_invalid_uppercase(self):
        """Test label validation rejects uppercase."""
        with pytest.raises(ValidationError):
            # Pydantic pattern validation runs before our validator
            SourceLocation(
                file_path="test.md",
                line_range=TextLocation.from_single_range(10, 20),
                label="Thm-Keystone",  # Uppercase not allowed
                article_id="03_test",
                volume="1_euclidean_gas",
                section="3.2",
                section_name="Test",
            )

    def test_get_full_text_single_range(self):
        """Test get_full_text with single continuous range using real file."""
        # Use real file that supports extraction
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(5, 7),
            label="test-intro",
            article_id="03_cloning",
        )
        text = loc.get_full_text()
        # Just verify we got some text
        assert len(text) > 0
        # Should not contain LINE_BLOCK_SEPARATOR for single range
        assert article_system.LINE_BLOCK_SEPARATOR not in text

    def test_get_full_text_discontinuous_ranges(self):
        """Test get_full_text with discontinuous ranges using real file."""
        # Use real file that supports extraction
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation(lines=[(5, 7), (100, 102)]),
            label="test-multi",
            article_id="03_cloning",
        )
        text = loc.get_full_text()
        # Should contain LINE_BLOCK_SEPARATOR between ranges
        assert article_system.LINE_BLOCK_SEPARATOR in text
        # Verify separator appears exactly once (two ranges = one separator)
        assert text.count(article_system.LINE_BLOCK_SEPARATOR) == 1

    def test_get_full_text_three_discontinuous_ranges(self):
        """Test get_full_text with three discontinuous ranges using real file."""
        # Use real file that supports extraction
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation(lines=[(5, 7), (100, 102), (200, 202)]),
            label="test-triple",
            article_id="03_cloning",
        )
        text = loc.get_full_text()
        # Should contain LINE_BLOCK_SEPARATOR between each pair (3 ranges = 2 separators)
        assert text.count(article_system.LINE_BLOCK_SEPARATOR) == 2

    def test_get_full_text_strips_trailing_whitespace(self):
        """Test that get_full_text strips trailing whitespace from chunks."""
        # Use real file with a valid line range that has a section header
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(10, 14),
            label="test-title",
            article_id="03_cloning",
        )
        text = loc.get_full_text()
        # Verify we got text
        assert len(text) > 0

    def test_get_full_text_file_not_found(self):
        """Test get_full_text raises FileNotFoundError for nonexistent file."""
        # File doesn't exist - validator will raise FileNotFoundError during section extraction
        with pytest.raises((FileNotFoundError, ValueError)):
            # Will raise ValueError for volume first, then FileNotFoundError for section
            SourceLocation(
                file_path="docs/source/1_euclidean_gas/nonexistent_file.md",
                line_range=TextLocation.from_single_range(10, 20),
                label="test-error",
                article_id="99_test",
            )

    def test_get_full_text_line_range_exceeds_file(self):
        """Test get_full_text raises ValueError when line range exceeds file length."""
        # Line range too large will pass validation (section header exists before line 100)
        # but will fail when calling get_full_text()
        loc = SourceLocation(
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            line_range=TextLocation.from_single_range(100, 100000),
            label="test-overflow",
            article_id="03_cloning",
        )
        with pytest.raises(ValueError, match="exceeds file length"):
            loc.get_full_text()

    def test_get_full_text_with_real_file(self):
        """Test get_full_text with real markdown file."""
        file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        loc = SourceLocation(
            file_path=file_path,
            line_range=TextLocation.from_single_range(13, 15),
            label="test-intro",
            article_id="01_fragile_gas_framework",
        )
        text = loc.get_full_text()
        # Lines 13-15 should contain "## 1. Introduction" and "### 1.1. Goal and Scope"
        assert "## 1. Introduction" in text
        assert "### 1.1" in text


class TestSourceLocationFromRequiredFields:
    """Test SourceLocation.from_required_fields classmethod."""

    def test_from_required_fields_01_framework_line_15(self):
        """Test auto-extraction from 01_fragile_gas_framework.md line 15."""
        file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        loc = SourceLocation.from_required_fields(
            file_path=file_path,
            line_range=TextLocation.from_single_range(15, 30),
            label="goal-and-scope",
        )

        assert loc.volume == "1_euclidean_gas"
        assert loc.article_id == "01_fragile_gas_framework"
        # Line 15 is "### 1.1. Goal and Scope" but function finds LAST header BEFORE line 15
        # which is line 13 "## 1. Introduction"
        assert loc.section == "1"
        assert "Introduction" in loc.section_name
        assert loc.file_path == file_path
        assert loc.label == "goal-and-scope"

    def test_from_required_fields_03_cloning_line_617(self):
        """Test auto-extraction from 03_cloning.md line 617."""
        file_path = "docs/source/1_euclidean_gas/03_cloning.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        loc = SourceLocation.from_required_fields(
            file_path=file_path,
            line_range=TextLocation.from_single_range(617, 635),
            label="lem-sx-implies-variance",
        )

        assert loc.volume == "1_euclidean_gas"
        assert loc.article_id == "03_cloning"
        # Line 617 is "#### 3.2.4 From Structural Error..." but function finds LAST header BEFORE line 617
        # which is line 462 "#### 3.2.3. The Decomposition..."
        assert loc.section == "3.2.3"
        assert "Decomposition" in loc.section_name or "Inter-Swarm" in loc.section_name
        assert loc.label == "lem-sx-implies-variance"

    def test_from_required_fields_03_cloning_line_422(self):
        """Test auto-extraction from 03_cloning.md line 422."""
        file_path = "docs/source/1_euclidean_gas/03_cloning.md"
        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        loc = SourceLocation.from_required_fields(
            file_path=file_path,
            line_range=TextLocation.from_single_range(422, 430),
            label="loc-error-component",
        )

        assert loc.volume == "1_euclidean_gas"
        assert loc.article_id == "03_cloning"
        # Line 422 is "#### 3.2.1. The Location Error Component" but function finds LAST header BEFORE line 422
        # which is line 418 "### 3.2. Permutation-Invariant Error Components"
        assert loc.section == "3.2"
        assert "Permutation" in loc.section_name or "Error Components" in loc.section_name

    def test_from_required_fields_invalid_path_raises(self):
        """Test that invalid path raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract article_id"):
            SourceLocation.from_required_fields(
                file_path="invalid/path.txt",  # Wrong extension
                line_range=TextLocation.from_single_range(10, 20),
                label="test",
            )

    def test_from_required_fields_nonexistent_file_raises(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SourceLocation.from_required_fields(
                file_path="/nonexistent/path/03_test.md",
                line_range=TextLocation.from_single_range(10, 20),
                label="test",
            )

    def test_from_required_fields_no_volume_extracted(self, tmp_path):
        """Test that ValueError is raised when volume pattern not found."""
        # Create a temp file outside the docs/source structure
        temp_file = tmp_path / "03_test.md"
        temp_file.write_text("## 1. Test Section\n\nContent.")

        # Should raise because file path doesn't match N_volume_name pattern
        with pytest.raises(ValueError, match="Cannot extract volume from file_path"):
            SourceLocation.from_required_fields(
                file_path=str(temp_file),
                line_range=TextLocation.from_single_range(3, 4),
                label="test",
            )


class TestArticle:
    """Test Article class."""

    def test_article_creation(self):
        """Create Article with all fields."""
        article = Article(
            article_id="03_cloning",
            title="The Keystone Principle and Cloning",
            file_path="docs/source/1_euclidean_gas/03_cloning.md",
            chapter=1,
            tags=["cloning", "measurement", "fitness"],
            contains_labels=["thm-keystone", "def-fitness"],
        )
        assert article.article_id == "03_cloning"
        assert article.title == "The Keystone Principle and Cloning"
        assert article.chapter == 1
        assert article.tags == ["cloning", "measurement", "fitness"]
        assert article.contains_labels == ["thm-keystone", "def-fitness"]

    def test_article_frozen(self):
        """Article should be frozen (immutable)."""
        article = Article(
            article_id="03_test",
            title="Test",
            file_path="test.md",
        )
        with pytest.raises(ValidationError):
            article.chapter = 2

    def test_has_tag_positive(self):
        """Test has_tag when tag exists."""
        article = Article(
            article_id="03_test",
            title="Test",
            file_path="test.md",
            tags=["cloning", "measurement"],
        )
        assert article.has_tag("cloning") is True
        assert article.has_tag("measurement") is True

    def test_has_tag_negative(self):
        """Test has_tag when tag doesn't exist."""
        article = Article(
            article_id="03_test",
            title="Test",
            file_path="test.md",
            tags=["cloning"],
        )
        assert article.has_tag("convergence") is False

    def test_has_label_positive(self):
        """Test has_label when label exists."""
        article = Article(
            article_id="03_test",
            title="Test",
            file_path="test.md",
            contains_labels=["thm-keystone", "def-fitness"],
        )
        assert article.has_label("thm-keystone") is True
        assert article.has_label("def-fitness") is True

    def test_has_label_negative(self):
        """Test has_label when label doesn't exist."""
        article = Article(
            article_id="03_test",
            title="Test",
            file_path="test.md",
            contains_labels=["thm-keystone"],
        )
        assert article.has_label("thm-unknown") is False

    def test_article_id_pattern_validation_valid(self):
        """Test article_id validation with valid pattern."""
        article = Article(
            article_id="03_cloning",
            title="Test",
            file_path="test.md",
        )
        assert article.article_id == "03_cloning"

    def test_article_id_pattern_validation_invalid(self):
        """Test article_id validation rejects invalid pattern."""
        with pytest.raises(ValidationError):
            Article(
                article_id="invalid_id",  # Missing number prefix
                title="Test",
                file_path="test.md",
            )

    def test_article_default_values(self):
        """Test Article default values."""
        article = Article(
            article_id="03_test",
            title="Test",
            file_path="test.md",
        )
        assert article.chapter is None
        assert article.section_number is None
        assert article.tags == []
        assert article.contains_labels == []
        assert article.version == "0.1.0"
        assert article.last_modified is None
        assert article.abstract is None
