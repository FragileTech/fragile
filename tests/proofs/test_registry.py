"""
Comprehensive tests for fragile.proofs.registry.

Tests MathematicalRegistry, reference system, and storage.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from fragile.proofs import (
    CombinedTagQuery,
    MathematicalObject,
    MathematicalRegistry,
    ObjectType,
    Property,
    Relationship,
    RelationType,
    TagQuery,
    TheoremBox,
    TheoremOutputType,
    create_simple_object,
    create_simple_theorem,
    load_registry_from_directory,
    save_registry_to_directory,
)


class TestTagQuery:
    """Test TagQuery functionality."""

    def test_any_mode(self):
        """Test ANY mode (OR logic)."""
        query = TagQuery(tags=["euclidean-gas", "discrete"], mode="any")

        # Object with euclidean-gas should match
        assert query.matches({"euclidean-gas", "particle"})

        # Object with discrete should match
        assert query.matches({"discrete", "system"})

        # Object with neither should not match
        assert not query.matches({"continuous", "pde"})

    def test_all_mode(self):
        """Test ALL mode (AND logic)."""
        query = TagQuery(tags=["euclidean-gas", "discrete"], mode="all")

        # Object with both tags should match
        assert query.matches({"euclidean-gas", "discrete", "particle"})

        # Object with only one tag should not match
        assert not query.matches({"euclidean-gas", "continuous"})
        assert not query.matches({"discrete", "pde"})

    def test_none_mode(self):
        """Test NONE mode (NOT logic)."""
        query = TagQuery(tags=["deprecated"], mode="none")

        # Object without deprecated tag should match
        assert query.matches({"euclidean-gas", "discrete"})

        # Object with deprecated tag should not match
        assert not query.matches({"euclidean-gas", "deprecated"})


class TestCombinedTagQuery:
    """Test CombinedTagQuery functionality."""

    def test_must_have(self):
        """Test must_have (AND) condition."""
        query = CombinedTagQuery(must_have=["euclidean-gas", "discrete"])

        assert query.matches({"euclidean-gas", "discrete", "particle"})
        assert not query.matches({"euclidean-gas", "continuous"})

    def test_any_of(self):
        """Test any_of (OR) condition."""
        query = CombinedTagQuery(any_of=["discrete", "continuous"])

        assert query.matches({"discrete", "particle"})
        assert query.matches({"continuous", "pde"})
        assert not query.matches({"euclidean-gas", "particle"})

    def test_must_not_have(self):
        """Test must_not_have (NOT) condition."""
        query = CombinedTagQuery(must_not_have=["deprecated"])

        assert query.matches({"euclidean-gas", "discrete"})
        assert not query.matches({"euclidean-gas", "deprecated"})

    def test_combined_conditions(self):
        """Test all conditions together."""
        query = CombinedTagQuery(
            must_have=["euclidean-gas"],
            any_of=["discrete", "continuous"],
            must_not_have=["deprecated"],
        )

        # Has euclidean-gas, has discrete, no deprecated -> match
        assert query.matches({"euclidean-gas", "discrete", "particle"})

        # Has euclidean-gas, has continuous, no deprecated -> match
        assert query.matches({"euclidean-gas", "continuous", "pde"})

        # Missing euclidean-gas -> no match
        assert not query.matches({"discrete", "particle"})

        # Has deprecated -> no match
        assert not query.matches({"euclidean-gas", "discrete", "deprecated"})

        # Missing both discrete and continuous -> no match
        assert not query.matches({"euclidean-gas", "particle"})


class TestMathematicalRegistry:
    """Test MathematicalRegistry functionality."""

    def test_registry_creation(self):
        """Test empty registry creation."""
        registry = MathematicalRegistry()
        assert len(registry._objects) == 0
        assert len(registry._relationships) == 0
        assert len(registry._theorems) == 0

    def test_add_object(self):
        """Test adding object to registry."""
        registry = MathematicalRegistry()
        obj = create_simple_object("obj-test", "Test Object", "X", ObjectType.SET)

        registry.add(obj)
        assert len(registry._objects) == 1
        assert "obj-test" in registry._objects

    def test_add_duplicate_object(self):
        """Test adding duplicate object raises error."""
        registry = MathematicalRegistry()
        obj = create_simple_object("obj-test", "Test Object", "X", ObjectType.SET)

        registry.add(obj)

        # Adding same object again should raise error
        with pytest.raises(ValueError, match="Duplicate ID"):
            registry.add(obj)

    def test_add_theorem(self):
        """Test adding theorem to registry."""
        registry = MathematicalRegistry()
        theorem = create_simple_theorem("thm-test", "Test Theorem", TheoremOutputType.PROPERTY)

        registry.add(theorem)
        assert len(registry._theorems) == 1
        assert "thm-test" in registry._theorems

    def test_add_relationship(self):
        """Test adding relationship to registry."""
        registry = MathematicalRegistry()

        # Add objects first
        obj_a = create_simple_object("obj-a", "Object A", "X", ObjectType.SET)
        obj_b = create_simple_object("obj-b", "Object B", "X", ObjectType.SET)
        registry.add(obj_a)
        registry.add(obj_b)

        # Add relationship
        rel = Relationship(
            label="rel-a-b-equivalence",
            relationship_type=RelationType.EQUIVALENCE,
            source_object="obj-a",
            target_object="obj-b",
            bidirectional=True,
            established_by="thm-test",
            expression="A ≡ B",
        )
        registry.add(rel)

        assert len(registry._relationships) == 1
        assert "rel-a-b-equivalence" in registry._relationships

    def test_query_by_tag_simple(self):
        """Test simple tag query."""
        registry = MathematicalRegistry()

        obj1 = MathematicalObject(
            label="obj-1",
            name="Object 1",
            mathematical_expression="X",
            object_type=ObjectType.SET,
            tags=["euclidean-gas", "discrete"],
        )
        obj2 = MathematicalObject(
            label="obj-2",
            name="Object 2",
            mathematical_expression="Y",
            object_type=ObjectType.SET,
            tags=["euclidean-gas", "continuous"],
        )

        registry.add(obj1)
        registry.add(obj2)

        # Query for discrete
        query = TagQuery(tags=["discrete"], mode="all")
        result = registry.query_by_tag(query)

        assert result.total_count == 1
        assert result.matches[0].label == "obj-1"

    def test_query_by_tags_combined(self):
        """Test combined tag query."""
        registry = MathematicalRegistry()

        obj1 = MathematicalObject(
            label="obj-1",
            name="Object 1",
            mathematical_expression="X",
            object_type=ObjectType.SET,
            tags=["euclidean-gas", "discrete"],
        )
        obj2 = MathematicalObject(
            label="obj-2",
            name="Object 2",
            mathematical_expression="Y",
            object_type=ObjectType.SET,
            tags=["euclidean-gas", "continuous"],
        )

        registry.add(obj1)
        registry.add(obj2)

        # Query for euclidean-gas with either discrete or continuous
        query = CombinedTagQuery(
            must_have=["euclidean-gas"],
            any_of=["discrete", "continuous"],
        )
        result = registry.query_by_tags(query)

        assert result.total_count == 2  # Both objects match

    def test_validate_referential_integrity(self):
        """Test referential integrity validation."""
        registry = MathematicalRegistry()

        # Add object
        obj = create_simple_object("obj-a", "Object A", "X", ObjectType.SET)
        registry.add(obj)

        # Add relationship referencing non-existent object
        rel = Relationship(
            label="rel-a-b-equivalence",
            relationship_type=RelationType.EQUIVALENCE,
            source_object="obj-a",
            target_object="obj-nonexistent",  # Doesn't exist!
            bidirectional=True,
            established_by="thm-test",
            expression="A ≡ B",
        )
        registry.add(rel)

        # Validate should find missing reference
        missing = registry.validate_referential_integrity()
        assert len(missing) > 0
        # Check that the error message mentions the missing object
        assert any("obj-nonexistent" in msg for msg in missing)


class TestStorage:
    """Test registry storage and loading."""

    def test_save_and_load_registry(self):
        """Test saving registry to directory and loading it back."""
        registry = MathematicalRegistry()

        # Add some objects
        obj1 = create_simple_object("obj-1", "Object 1", "X", ObjectType.SET)
        obj2 = create_simple_object("obj-2", "Object 2", "X", ObjectType.FUNCTION)
        registry.add(obj1)
        registry.add(obj2)

        # Add theorem
        theorem = create_simple_theorem("thm-1", "Theorem 1", TheoremOutputType.PROPERTY)
        registry.add(theorem)

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_registry"
            save_registry_to_directory(registry, save_path)

            # Check files were created
            assert (save_path / "objects").exists()
            assert (save_path / "theorems").exists()
            assert (save_path / "index.json").exists()

            # Load registry back
            loaded_registry = load_registry_from_directory(MathematicalRegistry, save_path)

            # Verify contents
            assert len(loaded_registry._objects) == 2
            assert len(loaded_registry._theorems) == 1
            assert "obj-1" in loaded_registry._objects
            assert "obj-2" in loaded_registry._objects
            assert "thm-1" in loaded_registry._theorems

    def test_save_and_load_with_relationships(self):
        """Test saving and loading registry with relationships."""
        registry = MathematicalRegistry()

        # Add objects
        obj_a = create_simple_object("obj-a", "Object A", "X", ObjectType.SET)
        obj_b = create_simple_object("obj-b", "Object B", "X", ObjectType.SET)
        registry.add(obj_a)
        registry.add(obj_b)

        # Add relationship
        rel = Relationship(
            label="rel-a-b-equivalence",
            relationship_type=RelationType.EQUIVALENCE,
            source_object="obj-a",
            target_object="obj-b",
            bidirectional=True,
            established_by="thm-test",
            expression="A ≡ B",
        )
        registry.add(rel)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_registry"
            save_registry_to_directory(registry, save_path)
            loaded_registry = load_registry_from_directory(MathematicalRegistry, save_path)

            # Verify relationship was preserved
            assert len(loaded_registry._relationships) == 1
            assert "rel-a-b-equivalence" in loaded_registry._relationships
