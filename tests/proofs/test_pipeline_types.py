"""
Comprehensive tests for fragile.proofs.core.pipeline_types.

Tests all core types, enums, helper functions, and validation rules.
"""

from pydantic import ValidationError
import pytest

from fragile.proofs.core.pipeline_types import (
    Attribute,
    AttributeEvent,
    AttributeEventType,
    AttributeRefinement,
    Axiom,
    create_simple_object,
    create_simple_theorem,
    Err,
    MathematicalObject,
    ObjectType,
    Ok,
    Parameter,
    ParameterType,
    RefinementType,
    Relationship,
    RelationshipAttribute,
    RelationType,
    TheoremBox,
    TheoremOutputType,
)


class TestEnums:
    """Test all enum types."""

    def test_theorem_output_type(self):
        """Test TheoremOutputType enum."""
        assert TheoremOutputType.PROPERTY == "Property"
        assert TheoremOutputType.RELATION == "Relation"
        assert TheoremOutputType.EXISTENCE == "Existence"
        assert len(list(TheoremOutputType)) >= 13  # All 13 types

    def test_object_type(self):
        """Test ObjectType enum."""
        assert ObjectType.SET == "set"
        assert ObjectType.FUNCTION == "function"
        assert ObjectType.MEASURE == "measure"

    def test_relation_type(self):
        """Test RelationType enum."""
        assert RelationType.EQUIVALENCE == "equivalence"
        assert RelationType.EMBEDDING == "embedding"
        assert len(list(RelationType)) >= 8

    def test_parameter_type(self):
        """Test ParameterType enum."""
        assert ParameterType.REAL == "real"
        assert ParameterType.INTEGER == "integer"


class TestResultTypes:
    """Test Ok and Err result types."""

    def test_ok_creation(self):
        """Test Ok type creation."""
        ok = Ok(value=42)
        assert ok.value == 42

    def test_ok_with_object(self):
        """Test Ok with complex object."""
        obj = {"key": "value"}
        ok = Ok(value=obj)
        assert ok.value == obj

    def test_err_creation(self):
        """Test Err type creation."""
        err = Err(error="Something went wrong")
        assert err.error == "Something went wrong"


class TestAttributeRefinement:
    """Test AttributeRefinement type."""

    def test_creation(self):
        """Test AttributeRefinement creation."""
        refinement = AttributeRefinement(
            original_attribute="attr-continuous",
            refined_attribute="attr-smooth",
            refinement_theorem="thm-regularity",
            refinement_type=RefinementType.STRENGTHENING,
        )
        assert refinement.refinement_type == RefinementType.STRENGTHENING
        assert refinement.original_attribute == "attr-continuous"

    def test_all_refinement_types(self):
        """Test all RefinementType values."""
        types = [
            RefinementType.STRENGTHENING,
            RefinementType.GENERALIZATION,
            RefinementType.QUANTIFICATION,
        ]
        for rtype in types:
            ref = AttributeRefinement(
                original_attribute="attr-test1",
                refined_attribute="attr-test2",
                refinement_theorem="thm-test",
                refinement_type=rtype,
            )
            assert ref.refinement_type == rtype


class TestAttribute:
    """Test Attribute type."""

    def test_minimal_property(self):
        """Test minimal property creation."""
        prop = Attribute(
            label="attr-test",
            object_label="obj-test",
            expression="Test property",
            established_by="thm-test",
        )
        assert prop.label == "attr-test"
        assert prop.object_label == "obj-test"

    def test_property_with_expression(self):
        """Test property with mathematical expression."""
        prop = Attribute(
            label="attr-lipschitz",
            object_label="obj-function",
            expression=r"|\nabla f(x)| \leq L",
            established_by="thm-regularity",
        )
        assert r"\nabla" in prop.expression

    def test_attribute_label_validation(self):
        """Test property label must start with prop-."""
        with pytest.raises(ValidationError, match="attr-"):
            Attribute(
                label="invalid-label",  # Should be prop-invalid-label
                object_label="obj-test",
                expression="Test",
                established_by="thm-test",
            )

    def test_property_with_refinement(self):
        """Test property with refinement history."""
        refinement = AttributeRefinement(
            original_attribute="attr-continuous",
            refined_attribute="attr-smooth",
            refinement_theorem="thm-regularity",
            refinement_type=RefinementType.STRENGTHENING,
        )
        prop = Attribute(
            label="attr-test",
            object_label="obj-test",
            expression="Test",
            established_by="thm-test",
            refinements=[refinement],
        )
        assert len(prop.refinements) > 0
        assert prop.refinements[0].refinement_type == RefinementType.STRENGTHENING


class TestAttributeEvent:
    """Test AttributeEvent type."""

    def test_property_added(self):
        """Test property addition event."""
        event = AttributeEvent(
            timestamp=0,
            attribute_label="attr-new",
            added_by_theorem="thm-creator",
            event_type=AttributeEventType.ADDED,
        )
        assert event.event_type == AttributeEventType.ADDED
        assert event.added_by_theorem == "thm-creator"

    def test_property_refined(self):
        """Test property refinement event."""
        event = AttributeEvent(
            timestamp=1,
            attribute_label="attr-existing",
            added_by_theorem="thm-refiner",
            event_type=AttributeEventType.REFINED,
        )
        assert event.event_type == AttributeEventType.REFINED


class TestMathematicalObject:
    """Test MathematicalObject type."""

    def test_minimal_object(self):
        """Test minimal mathematical object creation."""
        obj = MathematicalObject(
            label="obj-euclidean-space",
            name="Euclidean Space",
            mathematical_expression=r"\mathbb{R}^d",
            object_type=ObjectType.SET,
        )
        assert obj.label == "obj-euclidean-space"
        assert obj.object_type == ObjectType.SET

    def test_object_with_properties(self):
        """Test object with multiple properties."""
        obj = MathematicalObject(
            label="obj-function",
            name="Smooth Function",
            mathematical_expression="f: X -> Y",
            object_type=ObjectType.FUNCTION,
            current_attributes=[
                Attribute(
                    label="attr-continuous",
                    object_label="obj-function",
                    expression="f is continuous",
                    established_by="thm-regularity",
                ),
                Attribute(
                    label="attr-bounded",
                    object_label="obj-function",
                    expression="f is bounded",
                    established_by="thm-regularity",
                ),
            ],
        )
        assert len(obj.current_attributes) == 2

    def test_object_label_validation(self):
        """Test object label must start with obj-."""
        with pytest.raises(ValidationError, match="obj-"):
            MathematicalObject(
                label="invalid",
                name="Test",
                mathematical_expression="X",
                object_type=ObjectType.SET,
            )

    def test_object_with_tags(self):
        """Test object with tags."""
        obj = MathematicalObject(
            label="obj-test",
            name="Test Object",
            mathematical_expression="X",
            object_type=ObjectType.SET,
            tags=["euclidean-gas", "discrete", "particle-system"],
        )
        assert "euclidean-gas" in obj.tags
        assert len(obj.tags) == 3


class TestRelationship:
    """Test Relationship type."""

    def test_minimal_relationship(self):
        """Test minimal relationship creation."""
        rel = Relationship(
            label="rel-a-b-equivalence",
            relationship_type=RelationType.EQUIVALENCE,
            source_object="obj-a",
            target_object="obj-b",
            bidirectional=True,
            established_by="thm-equivalence",
            expression="A ≡ B",
        )
        assert rel.relationship_type == RelationType.EQUIVALENCE
        assert rel.bidirectional is True

    def test_embedding_relationship(self):
        """Test embedding relationship (unidirectional)."""
        rel = Relationship(
            label="rel-discrete-continuous-embedding",
            relationship_type=RelationType.EMBEDDING,
            source_object="obj-discrete",
            target_object="obj-continuous",
            bidirectional=False,
            established_by="thm-embedding",
            expression=r"\phi: X_{\text{discrete}} \hookrightarrow X_{\text{continuous}}",
        )
        assert rel.bidirectional is False
        assert r"\hookrightarrow" in rel.expression

    def test_relationship_with_properties(self):
        """Test relationship with attributes."""
        rel_attr = RelationshipAttribute(
            label="error-bound",
            expression="Error bounded by O(N^{-1/d})",
        )
        rel = Relationship(
            label="rel-a-b-approximation",
            relationship_type=RelationType.APPROXIMATION,
            source_object="obj-a",
            target_object="obj-b",
            bidirectional=False,
            established_by="thm-test",
            expression="A ≈ B",
            attributes=[rel_attr],
        )
        assert len(rel.attributes) == 1
        assert "N^{-1/d}" in rel.attributes[0].expression


class TestAxiom:
    """Test Axiom type."""

    def test_axiom_creation(self):
        """Test axiom creation."""
        axiom = Axiom(
            label="axiom-bounded-displacement",
            statement="Displacement operator is bounded",
            mathematical_expression=r"\|\Psi_{\text{disp}}\| \leq 1",
            foundational_framework="Euclidean Gas",
        )
        assert axiom.label.startswith("axiom-")
        assert axiom.foundational_framework == "Euclidean Gas"

    def test_axiom_label_validation(self):
        """Test axiom label must start with axiom-."""
        with pytest.raises(ValidationError, match="axiom-"):
            Axiom(
                label="invalid",
                statement="Test",
                mathematical_expression="X",
                foundational_framework="Test",
            )


class TestParameter:
    """Test Parameter type."""

    def test_parameter_creation(self):
        """Test parameter creation."""
        param = Parameter(
            label="param-gamma",
            name="gamma",
            parameter_type=ParameterType.REAL,
            symbol=r"\gamma",
        )
        assert param.parameter_type == ParameterType.REAL
        assert param.symbol == r"\gamma"

    def test_parameter_with_constraints(self):
        """Test parameter with constraints."""
        param = Parameter(
            label="param-n",
            name="N",
            parameter_type=ParameterType.NATURAL,
            symbol="N",
            constraints="N > 0",
        )
        assert param.parameter_type == ParameterType.NATURAL
        assert param.constraints == "N > 0"


class TestTheoremBox:
    """Test TheoremBox type."""

    def test_minimal_theorem(self):
        """Test minimal theorem creation."""
        theorem = TheoremBox(
            label="thm-test",
            name="Test Theorem",
            output_type=TheoremOutputType.PROPERTY,
            input_objects=["obj-a"],
            properties_required={"obj-a": ["attr-1"]},
        )
        assert theorem.label == "thm-test"
        assert theorem.output_type == TheoremOutputType.PROPERTY

    def test_theorem_label_validation(self):
        """Test theorem label must start with thm-/lem-/prop-."""
        with pytest.raises(ValidationError, match="pattern"):
            TheoremBox(
                label="invalid",
                name="Test",
                output_type=TheoremOutputType.PROPERTY,
            )

    def test_theorem_with_relationships(self):
        """Test theorem establishing relationships."""
        rel = Relationship(
            label="rel-a-b-equivalence",
            relationship_type=RelationType.EQUIVALENCE,
            source_object="obj-a",
            target_object="obj-b",
            bidirectional=True,
            established_by="thm-equivalence",
            expression="A ≡ B",
        )
        theorem = TheoremBox(
            label="thm-equivalence",
            name="Systems Equivalence",
            output_type=TheoremOutputType.EQUIVALENCE,
            input_objects=["obj-a", "obj-b"],
            relations_established=[rel],
        )
        assert len(theorem.relations_established) == 1


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_simple_object(self):
        """Test create_simple_object helper."""
        obj = create_simple_object(
            label="obj-test",
            name="Test Object",
            expr="X",
            obj_type=ObjectType.SET,
        )
        assert obj.label == "obj-test"
        assert obj.name == "Test Object"
        assert obj.object_type == ObjectType.SET

    def test_create_simple_theorem(self):
        """Test create_simple_theorem helper."""
        theorem = create_simple_theorem(
            label="thm-test",
            name="Test Theorem",
            output_type=TheoremOutputType.PROPERTY,
        )
        assert theorem.label == "thm-test"
        assert theorem.name == "Test Theorem"
        assert theorem.output_type == TheoremOutputType.PROPERTY


class TestImmutability:
    """Test that all types are frozen (immutable)."""

    def test_property_immutable(self):
        """Test Property is frozen."""
        prop = Attribute(
            label="attr-test",
            object_label="obj-test",
            expression="Test",
            established_by="thm-test",
        )
        with pytest.raises(ValidationError):
            prop.label = "attr-modified"

    def test_object_immutable(self):
        """Test MathematicalObject is frozen."""
        obj = create_simple_object("obj-test", "Test", "X", ObjectType.SET)
        with pytest.raises(ValidationError):
            obj.label = "obj-modified"

    def test_theorem_immutable(self):
        """Test TheoremBox is frozen."""
        theorem = create_simple_theorem("thm-test", "Test", TheoremOutputType.PROPERTY)
        with pytest.raises(ValidationError):
            theorem.label = "thm-modified"
