"""
Direct test of RawDefinition mode='before' validator.
Bypasses package imports to avoid unrelated import errors.
"""
import sys
import warnings
from pathlib import Path

# Add src directories to path to import modules directly
sys.path.insert(0, str(Path(__file__).parent / "src" / "mathster" / "core"))

# Import modules directly without package init
import article_system
import raw_data

# Get the classes we need
RawDefinition = raw_data.RawDefinition
SourceLocation = article_system.SourceLocation
TextLocation = article_system.TextLocation

# Rebuild model to resolve forward references
RawDefinition.model_rebuild()

print("Testing RawDefinition mode='before' validator implementation")
print("=" * 70)

# Test 1: Auto-populate from source.label (no user label provided)
print("\n✓ Test 1: Auto-populate from source.label")
source = SourceLocation(
    file_path="docs/test.md",
    line_range=TextLocation.from_single_range(100, 120),
    label="def-test-term",
    article_id="test_doc",
)

definition = RawDefinition(
    source=source,
    full_text=TextLocation.from_single_range(100, 120),
    term="Test Term",
)

assert definition.label == "def-test-term", f"Expected def-test-term, got {definition.label}"
print(f"  Label auto-populated: {definition.label}")

# Test 2: User label matches source.label (no warning)
print("\n✓ Test 2: User label matches source.label (no warning)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    definition2 = RawDefinition(
        source=source,
        full_text=TextLocation.from_single_range(100, 120),
        term="Test Term",
        label="def-test-term",  # Matches source.label
    )
    user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]

assert len(user_warnings) == 0, f"Expected no warnings, got {len(user_warnings)}"
assert definition2.label == "def-test-term"
print(f"  No warning emitted, label: {definition2.label}")

# Test 3: User label doesn't match source.label (emits warning, keeps user value)
print("\n✓ Test 3: User label mismatch (warning + preserve user value)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    definition3 = RawDefinition(
        source=source,
        full_text=TextLocation.from_single_range(100, 120),
        term="Test Term",
        label="def-wrong-label",  # Doesn't match source.label
    )
    user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]

assert len(user_warnings) > 0, "Expected warning for mismatch"
assert "mismatch" in str(user_warnings[0].message).lower()
assert definition3.label == "def-wrong-label", "Expected user value to be kept"
print(f"  Warning: {user_warnings[0].message}")
print(f"  User value preserved: {definition3.label}")

# Test 4: Invalid source.label raises error
print("\n✓ Test 4: Invalid source.label (should raise ValueError)")
invalid_source = SourceLocation(
    file_path="docs/test.md",
    line_range=TextLocation.from_single_range(100, 120),
    label="thm-wrong-prefix",  # Wrong prefix!
    article_id="test_doc",
)

try:
    definition4 = RawDefinition(
        source=invalid_source,
        full_text=TextLocation.from_single_range(100, 120),
        term="Test Term",
    )
    print("  ✗ FAILED - Should have raised ValueError!")
    sys.exit(1)
except ValueError as e:
    assert "does not match expected pattern" in str(e)
    print(f"  ValueError raised correctly: {str(e)[:60]}...")

# Test 5: Verify it's actually using mode='before' (dict handling)
print("\n✓ Test 5: Verify mode='before' handles dict input")
# In mode='before', the validator receives raw dict data
raw_dict = {
    "source": {
        "file_path": "docs/test.md",
        "line_range": {"ranges": [(100, 120)]},
        "label": "def-dict-test",
        "article_id": "test_doc",
    },
    "full_text": {"ranges": [(100, 120)]},
    "term": "Dict Test",
}

definition5 = RawDefinition(**raw_dict)
assert definition5.label == "def-dict-test"
print(f"  Dict input handled correctly, label: {definition5.label}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("mode='before' validator implementation is working correctly.")
print("\nKey features verified:")
print("  • Auto-populates label from source.label when not provided")
print("  • Warns when user label doesn't match source.label")
print("  • Preserves user value on mismatch (permissive policy)")
print("  • Validates source.label starts with 'def-' (strict validation)")
print("  • Handles both dict and SourceLocation object inputs")
