"""
Test suite for type conversion utilities.

Tests the conversion from enriched_types (document extraction) to
math_types (framework computation).
"""

import pytest

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.enriched_types import (
    EquationBox,
    ParameterBox,
    ParameterScope,
    RemarkBox,
    RemarkType,
)
from fragile.proofs.core.math_types import (
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
    TheoremBox,
    TheoremOutputType,
)
from fragile.proofs.core.type_conversions import (
    ConversionReport,
    convert_parameter_boxes,
    extract_chapter_document,
    link_all_enriched_types,
    link_equation_to_theorems,
    link_remark_to_entities,
    merge_constraints,
    parameter_box_to_parameter,
)


# =============================================================================
# HELPER UTILITIES TESTS
# =============================================================================


def test_extract_chapter_document_standard_path():
    """Test chapter/document extraction from standard path."""
    source = SourceLocation(
        document_id="04_convergence",
        file_path="docs/source/1_euclidean_gas/04_convergence/theorem.md",
        line_range=(10, 20),
    )
    chapter, document = extract_chapter_document(source)

    assert chapter == "1_euclidean_gas"
    assert document == "04_convergence"


def test_extract_chapter_document_with_md_extension():
    """Test extraction when document has .md extension."""
    source = SourceLocation(
        document_id="11_geometric_gas",
        file_path="docs/source/2_geometric_gas/11_geometric_gas.md",
        line_range=(1, 10),
    )
    chapter, document = extract_chapter_document(source)

    assert chapter == "2_geometric_gas"
    assert document == "11_geometric_gas"  # .md should be removed


def test_extract_chapter_document_standalone_file():
    """Test extraction from standalone file (no chapter/document)."""
    source = SourceLocation(
        document_id="99_standalone", file_path="standalone.md", line_range=(1, 10)
    )
    chapter, document = extract_chapter_document(source)

    assert chapter is None
    assert document is None


def test_extract_chapter_document_none_source():
    """Test extraction when source is None."""
    chapter, document = extract_chapter_document(None)

    assert chapter is None
    assert document is None


def test_merge_constraints_multiple():
    """Test merging multiple constraints."""
    result = merge_constraints(["γ > 0", "γ < 1", "γ ≠ 0.5"])
    assert result == "γ > 0, γ < 1, γ ≠ 0.5"


def test_merge_constraints_single():
    """Test merging single constraint."""
    result = merge_constraints(["N >= 2"])
    assert result == "N >= 2"


def test_merge_constraints_empty():
    """Test merging empty list."""
    result = merge_constraints([])
    assert result is None


# =============================================================================
# PARAMETER CONVERSION TESTS
# =============================================================================


def test_parameter_box_to_parameter_full_data():
    """Test conversion with all fields populated."""
    source = SourceLocation(
        document_id="04_convergence",
        file_path="docs/source/1_euclidean_gas/04_convergence/params.md",
        line_range=(42, 45),
    )

    param_box = ParameterBox(
        label="param-gamma",
        symbol="γ",
        latex="\\gamma",
        domain=ParameterType.REAL,
        meaning="friction coefficient controlling damping",
        scope=ParameterScope.GLOBAL,
        constraints=["γ > 0", "γ < 1"],
        default_value="0.5",
        source=source,
    )

    param = parameter_box_to_parameter(param_box)

    assert param.label == "param-gamma"
    assert param.name == "friction coefficient controlling damping"
    assert param.symbol == "γ"
    assert param.parameter_type == ParameterType.REAL
    assert param.constraints == "γ > 0, γ < 1"
    assert param.default_value == "0.5"
    assert param.chapter == "1_euclidean_gas"
    assert param.document == "04_convergence"


def test_parameter_box_to_parameter_minimal_data():
    """Test conversion with minimal required fields."""
    param_box = ParameterBox(
        label="param-n",
        symbol="N",
        latex="N",
        domain=ParameterType.NATURAL,
        meaning="number of walkers",
        scope=ParameterScope.LOCAL,
    )

    param = parameter_box_to_parameter(param_box)

    assert param.label == "param-n"
    assert param.name == "number of walkers"
    assert param.symbol == "N"
    assert param.parameter_type == ParameterType.NATURAL
    assert param.constraints is None  # No constraints
    assert param.default_value is None
    assert param.chapter is None  # No source
    assert param.document is None


def test_parameter_box_to_parameter_no_source():
    """Test conversion when source location is missing."""
    param_box = ParameterBox(
        label="param-epsilon",
        symbol="ε",
        latex="\\varepsilon",
        domain=ParameterType.REAL,
        meaning="regularization parameter",
        scope=ParameterScope.GLOBAL,
        constraints=["ε > 0"],
    )

    param = parameter_box_to_parameter(param_box)

    assert param.chapter is None
    assert param.document is None
    assert param.constraints == "ε > 0"


# =============================================================================
# BATCH CONVERSION TESTS
# =============================================================================


def test_convert_parameter_boxes_all_success():
    """Test batch conversion with all successes."""
    param_boxes = {
        "param-gamma": ParameterBox(
            label="param-gamma",
            symbol="γ",
            latex="\\gamma",
            domain=ParameterType.REAL,
            meaning="friction coefficient",
            scope=ParameterScope.GLOBAL,
        ),
        "param-n": ParameterBox(
            label="param-n",
            symbol="N",
            latex="N",
            domain=ParameterType.NATURAL,
            meaning="number of walkers",
            scope=ParameterScope.GLOBAL,
        ),
    }

    params, report = convert_parameter_boxes(param_boxes)

    assert report.total_items == 2
    assert report.successful == 2
    assert report.failed == 0
    assert len(params) == 2
    assert "param-gamma" in params
    assert "param-n" in params


def test_convert_parameter_boxes_with_warnings():
    """Test batch conversion generates warnings for missing source."""
    param_boxes = {
        "param-gamma": ParameterBox(
            label="param-gamma",
            symbol="γ",
            latex="\\gamma",
            domain=ParameterType.REAL,
            meaning="friction coefficient",
            scope=ParameterScope.GLOBAL,
            source=None,  # Missing source triggers warning
        ),
    }

    _params, report = convert_parameter_boxes(param_boxes)

    assert report.successful == 1
    assert len(report.warnings) > 0
    assert any("No source location" in w for w in report.warnings)


def test_conversion_report_summary():
    """Test conversion report summary formatting."""
    report = ConversionReport(
        total_items=10,
        successful=8,
        failed=2,
        warnings=["Warning 1", "Warning 2"],
        errors=["Error 1", "Error 2"],
    )

    summary = report.summary()

    assert "8/10 successful" in summary
    assert "80.0%" in summary
    assert "Failed: 2" in summary
    assert "Warnings: 2" in summary
    assert "Errors: 2" in summary


# =============================================================================
# SEMANTIC LINKING TESTS
# =============================================================================


def test_link_equation_to_theorems():
    """Test linking equation to theorems."""
    eq_box = EquationBox(
        label="eq-langevin",
        equation_number="(2.1)",
        latex_content="dx_t = v_t dt, \\quad dv_t = -\\gamma v_t dt + \\sqrt{2\\gamma} dW_t",
        appears_in_theorems=["thm-convergence", "thm-ergodicity"],
    )

    theorems = {
        "thm-convergence": TheoremBox(
            label="thm-convergence",
            name="Convergence to QSD",
            output_type=TheoremOutputType.CONVERGENCE,
            uses_definitions=[],  # Initially empty
        ),
        "thm-ergodicity": TheoremBox(
            label="thm-ergodicity",
            name="Ergodicity",
            output_type=TheoremOutputType.PROPERTY,
            uses_definitions=["def-qsd"],  # Already has some definitions
        ),
    }

    updated = link_equation_to_theorems(eq_box, theorems)

    # Check equation was added to both theorems
    assert "eq-langevin" in updated["thm-convergence"].uses_definitions
    assert "eq-langevin" in updated["thm-ergodicity"].uses_definitions
    # Check existing definition preserved
    assert "def-qsd" in updated["thm-ergodicity"].uses_definitions


def test_link_equation_no_duplicate():
    """Test linking equation doesn't create duplicates."""
    eq_box = EquationBox(
        label="eq-langevin",
        latex_content="...",
        appears_in_theorems=["thm-convergence"],
    )

    theorems = {
        "thm-convergence": TheoremBox(
            label="thm-convergence",
            name="Convergence",
            output_type=TheoremOutputType.CONVERGENCE,
            uses_definitions=["eq-langevin"],  # Already has this equation
        ),
    }

    updated = link_equation_to_theorems(eq_box, theorems)

    # Should not duplicate
    assert updated["thm-convergence"].uses_definitions.count("eq-langevin") == 1


def test_link_remark_to_theorems():
    """Test linking remark to theorems (remarks not linked since TheoremBox has no tags)."""
    remark_box = RemarkBox(
        label="remark-kinetic-necessity",
        remark_type=RemarkType.REMARK,
        content="The condition v > 0 is essential because...",
        relates_to=["thm-convergence"],
        provides_intuition_for=["thm-convergence"],
    )

    theorems = {
        "thm-convergence": TheoremBox(
            label="thm-convergence",
            name="Convergence",
            output_type=TheoremOutputType.CONVERGENCE,
        ),
    }

    updated_thms, _ = link_remark_to_entities(remark_box, theorems, {})

    # Theorems returned unchanged (TheoremBox doesn't have tags field)
    assert updated_thms["thm-convergence"] == theorems["thm-convergence"]


def test_link_remark_to_objects():
    """Test linking remark to mathematical objects."""
    remark_box = RemarkBox(
        label="remark-particle-interpretation",
        remark_type=RemarkType.NOTE,
        content="Think of walkers as particles...",
        relates_to=["obj-walker"],
    )

    objects = {
        "obj-walker": MathematicalObject(
            label="obj-walker",
            name="Walker",
            mathematical_expression="w = (x, v, s)",
            object_type=ObjectType.STRUCTURE,
            tags=[],
        ),
    }

    _, updated_objs = link_remark_to_entities(remark_box, {}, objects)

    # Remark label should be added as tag
    assert "remark-particle-interpretation" in updated_objs["obj-walker"].tags


# =============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# =============================================================================


def test_link_all_enriched_types_comprehensive():
    """Test full pipeline: convert parameters + link equations + link remarks."""
    # Setup enriched types
    param_boxes = {
        "param-gamma": ParameterBox(
            label="param-gamma",
            symbol="γ",
            latex="\\gamma",
            domain=ParameterType.REAL,
            meaning="friction coefficient",
            scope=ParameterScope.GLOBAL,
            constraints=["γ > 0"],
        ),
    }

    equation_boxes = {
        "eq-langevin": EquationBox(
            label="eq-langevin",
            latex_content="dx_t = v_t dt",
            appears_in_theorems=["thm-convergence"],
        ),
    }

    remark_boxes = {
        "remark-kinetic": RemarkBox(
            label="remark-kinetic",
            remark_type=RemarkType.REMARK,
            content="The kinetic term is essential...",
            relates_to=["obj-walker"],  # Link to object, not theorem
        ),
    }

    theorems = {
        "thm-convergence": TheoremBox(
            label="thm-convergence",
            name="Convergence",
            output_type=TheoremOutputType.CONVERGENCE,
            uses_definitions=[],
        ),
    }

    objects = {
        "obj-walker": MathematicalObject(
            label="obj-walker",
            name="Walker",
            mathematical_expression="w = (x, v, s)",
            object_type=ObjectType.STRUCTURE,
            tags=[],
        ),
    }

    # Run comprehensive linking
    params, updated_thms, updated_objs, report = link_all_enriched_types(
        param_boxes, equation_boxes, remark_boxes, theorems, objects
    )

    # Check parameters converted
    assert "param-gamma" in params
    assert params["param-gamma"].name == "friction coefficient"
    assert report.successful == 1

    # Check equation linked
    assert "eq-langevin" in updated_thms["thm-convergence"].uses_definitions

    # Check remark linked to object (not theorem, since TheoremBox has no tags)
    assert "remark-kinetic" in updated_objs["obj-walker"].tags


def test_link_all_enriched_types_empty_inputs():
    """Test comprehensive linking with empty inputs."""
    params, thms, objs, report = link_all_enriched_types(
        param_boxes={},
        equation_boxes={},
        remark_boxes={},
        theorems={},
        objects={},
    )

    assert len(params) == 0
    assert len(thms) == 0
    assert len(objs) == 0
    assert report.total_items == 0
    assert report.successful == 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


def test_parameter_conversion_preserves_all_data():
    """Test that no data is lost during conversion."""
    source = SourceLocation(
        document_id="04_convergence",
        file_path="docs/source/1_euclidean_gas/04_convergence/params.md",
        line_range=(1, 10),
    )

    param_box = ParameterBox(
        label="param-test",
        symbol="τ",
        latex="\\tau",
        domain=ParameterType.REAL,
        meaning="time step",
        scope=ParameterScope.GLOBAL,
        constraints=["τ > 0", "τ < 0.1"],
        default_value="0.01",
        full_definition_text="Throughout, τ > 0 denotes the time step...",
        appears_in=["thm-1", "thm-2"],
        source=source,
    )

    param = parameter_box_to_parameter(param_box)

    # Check all convertible data preserved
    assert param.label == param_box.label
    assert param.symbol == param_box.symbol
    assert param.parameter_type == param_box.domain
    assert param.default_value == param_box.default_value
    assert "τ > 0" in param.constraints
    assert "τ < 0.1" in param.constraints


def test_link_equation_to_nonexistent_theorem():
    """Test linking equation when theorem doesn't exist."""
    eq_box = EquationBox(
        label="eq-test",
        latex_content="...",
        appears_in_theorems=["thm-nonexistent"],
    )

    theorems = {
        "thm-exists": TheoremBox(
            label="thm-exists",
            name="Existing",
            output_type=TheoremOutputType.PROPERTY,
        ),
    }

    # Should not raise error, just skip nonexistent theorem
    updated = link_equation_to_theorems(eq_box, theorems)

    # Original theorem unchanged
    assert "eq-test" not in updated["thm-exists"].uses_definitions


def test_conversion_report_with_many_errors():
    """Test report summary truncates long error lists."""
    report = ConversionReport(
        total_items=100,
        successful=90,
        failed=10,
        warnings=[f"Warning {i}" for i in range(20)],
        errors=[f"Error {i}" for i in range(15)],
    )

    summary = report.summary()

    # Should show first 3 warnings/errors plus count
    assert "... and 17 more" in summary  # 20 - 3 = 17 more warnings
    assert "... and 12 more" in summary  # 15 - 3 = 12 more errors


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================


def test_parameter_conversion_label_preservation():
    """Test that label format is always preserved."""
    param_box = ParameterBox(
        label="param-test-123",
        symbol="x",
        latex="x",
        domain=ParameterType.REAL,
        meaning="test parameter",
        scope=ParameterScope.LOCAL,
    )

    param = parameter_box_to_parameter(param_box)

    # Label must be preserved exactly
    assert param.label == "param-test-123"
    # Must start with param-
    assert param.label.startswith("param-")


def test_batch_conversion_preserves_order():
    """Test that batch conversion preserves dictionary keys."""
    param_boxes = {
        f"param-{i}": ParameterBox(
            label=f"param-{i}",
            symbol=f"x{i}",
            latex=f"x_{i}",
            domain=ParameterType.REAL,
            meaning=f"parameter {i}",
            scope=ParameterScope.LOCAL,
        )
        for i in range(10)
    }

    params, _ = convert_parameter_boxes(param_boxes)

    # All keys should be preserved
    assert set(params.keys()) == set(param_boxes.keys())
