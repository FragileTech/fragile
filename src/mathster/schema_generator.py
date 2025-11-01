"""
Schema Generator for LLM-Guided Mathematical Object Creation.

This module generates a comprehensive JSON schema file containing all Pydantic
model schemas, workflow examples, dependency graphs, and cross-references to
documentation. Optimized for LLM consumption in the agentic math pipeline.

Usage:
    # Generate schema programmatically
    from fragile.mathster.schema_generator import generate_complete_schema
    schema = generate_complete_schema()

    # Generate via CLI
    python -m fragile.mathster.schema_generator
    python -m fragile.mathster.schema_generator --output custom_schema.json
    python -m fragile.mathster.schema_generator --include-examples

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, get_args, get_origin

from pydantic import BaseModel


# =============================================================================
# MODEL DISCOVERY
# =============================================================================


def discover_all_models() -> dict[str, list[type[BaseModel]]]:
    """
    Discover all Pydantic models in fragile.mathster module.

    Returns:
        Dictionary mapping module names to lists of model classes
    """
    models_by_module = {
        "core": [],
        "sympy": [],
        "registry": [],
        "relationships": [],
    }

    # Import all submodules
    submodules = {
        "core": ["pipeline_types", "proof_system", "proof_integration"],
        "sympy": [
            "expressions",
            "dual_representation",
            "validation",
            "proof_integration",
            "object_extensions",
        ],
        "registry": ["reference_system", "registry", "storage"],
        "relationships": ["graphs"],
    }

    for category, module_names in submodules.items():
        for module_name in module_names:
            try:
                module = importlib.import_module(f"fragile.mathster.{category}.{module_name}")
                # Find all BaseModel subclasses in module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseModel)
                        and obj is not BaseModel
                        and obj.__module__.startswith("fragile.mathster")
                    ):
                        models_by_module[category].append(obj)
            except ImportError as e:
                print(f"Warning: Could not import {category}.{module_name}: {e}", file=sys.stderr)

    return models_by_module


def get_model_name(model: type[BaseModel]) -> str:
    """Get the full qualified name of a model."""
    return model.__name__


def get_model_module(model: type[BaseModel]) -> str:
    """Get the module path of a model."""
    return model.__module__


# =============================================================================
# DEPENDENCY ANALYSIS
# =============================================================================


def extract_field_dependencies(model: type[BaseModel]) -> set[str]:
    """
    Extract dependencies from model fields.

    Analyzes field type annotations to find references to other BaseModel classes.
    """
    dependencies = set()

    for field_info in model.model_fields.values():
        annotation = field_info.annotation
        _extract_dependencies_from_annotation(annotation, dependencies)

    return dependencies


def _extract_dependencies_from_annotation(annotation: Any, dependencies: set[str]) -> None:
    """Recursively extract dependencies from type annotation."""
    # Handle None
    if annotation is None or annotation is type(None):
        return

    # Get origin for generic types (List, Dict, Optional, etc.)
    origin = get_origin(annotation)

    if origin is not None:
        # Generic type - recurse into args
        args = get_args(annotation)
        for arg in args:
            _extract_dependencies_from_annotation(arg, dependencies)
    else:
        # Check if it's a BaseModel subclass
        try:
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                dependencies.add(annotation.__name__)
        except TypeError:
            # Not a class
            pass


def build_dependency_graph(models: list[type[BaseModel]]) -> dict[str, list[str]]:
    """
    Build dependency graph showing which models reference which.

    Returns:
        Dictionary mapping model names to list of model names they depend on
    """
    graph = {}

    for model in models:
        model_name = get_model_name(model)
        dependencies = extract_field_dependencies(model)
        graph[model_name] = sorted(dependencies)

    return graph


def build_inverse_dependency_graph(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Build inverse dependency graph (what uses this model).

    Returns:
        Dictionary mapping model names to list of models that use them
    """
    inverse = {name: [] for name in graph.keys()}

    for model_name, dependencies in graph.items():
        for dep in dependencies:
            if dep in inverse:
                inverse[dep].append(model_name)

    # Sort for consistency
    for model_name in inverse:
        inverse[model_name] = sorted(inverse[model_name])

    return inverse


def topological_sort_models(graph: dict[str, list[str]]) -> list[str]:
    """
    Topologically sort models by dependencies (primitives first).

    Returns:
        List of model names in dependency order
    """
    # Kahn's algorithm for topological sort
    in_degree = dict.fromkeys(graph.keys(), 0)

    for dependencies in graph.values():
        for dep in dependencies:
            if dep in in_degree:
                in_degree[dep] += 1

    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        queue.sort()  # For deterministic ordering
        node = queue.pop(0)
        result.append(node)

        # Reduce in-degree for nodes that depend on this one
        for name, dependencies in graph.items():
            if node in dependencies:
                in_degree[name] -= 1
                if in_degree[name] == 0 and name not in result:
                    queue.append(name)

    return result


# =============================================================================
# EXAMPLE GENERATION
# =============================================================================


def generate_example_instance(model: type[BaseModel]) -> dict[str, Any] | None:
    """
    Generate a valid example instance of a Pydantic model.

    Uses model defaults, factories, and smart fallbacks.
    """
    try:
        # Try to create instance with defaults
        instance = model.model_validate({})
        return instance.model_dump(mode="json")
    except Exception:
        # If that fails, try to provide minimal valid values
        example_data = {}
        for field_name, field_info in model.model_fields.items():
            if field_info.is_required():
                # Provide minimal valid value based on type
                example_data[field_name] = _get_minimal_value(field_info.annotation)

        try:
            instance = model.model_validate(example_data)
            return instance.model_dump(mode="json")
        except Exception as e:
            print(
                f"Warning: Could not generate example for {model.__name__}: {e}", file=sys.stderr
            )
            return None


def _get_minimal_value(annotation: Any) -> Any:
    """Get minimal valid value for a type annotation."""
    # Handle None
    if annotation is None or annotation is type(None):
        return None

    origin = get_origin(annotation)

    # Handle Optional[T] -> None
    if origin is type(None):
        return None

    # Handle List[T] -> []
    if origin is list:
        return []

    # Handle Dict[K, V] -> {}
    if origin is dict:
        return {}

    # Handle Set[T] -> set()
    if origin is set:
        return set()

    # Primitive types
    if annotation is str:
        return "example"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False

    # BaseModel subclass
    try:
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            # Return None - will be handled by Optional
            return None
    except TypeError:
        pass

    return None


# =============================================================================
# METADATA EXTRACTION
# =============================================================================


def extract_model_metadata(model: type[BaseModel]) -> dict[str, Any]:
    """
    Extract comprehensive metadata from a Pydantic model.

    Returns:
        Dictionary with docstring, fields, schema, module info
    """
    return {
        "name": get_model_name(model),
        "module": get_model_module(model),
        "docstring": inspect.getdoc(model) or "No documentation available",
        "json_schema": model.model_json_schema(),
        "fields": _extract_field_metadata(model),
    }


def _extract_field_metadata(model: type[BaseModel]) -> dict[str, Any]:
    """Extract metadata for all fields in a model."""
    field_metadata = {}

    for field_name, field_info in model.model_fields.items():
        field_metadata[field_name] = {
            "type": str(field_info.annotation),
            "required": field_info.is_required(),
            "description": field_info.description or "No description",
            "default": str(field_info.default) if field_info.default is not None else None,
        }

    return field_metadata


# =============================================================================
# DOCUMENTATION LINKING
# =============================================================================


def link_to_glossary(model: type[BaseModel], glossary_path: Path) -> list[str]:
    """
    Find cross-references to this model in docs/glossary.md.

    Searches for model name or related labels.
    """
    if not glossary_path.exists():
        return []

    model_name = get_model_name(model).lower()
    references = []

    try:
        with open(glossary_path, encoding="utf-8") as f:
            content = f.read()

        # Search for mentions of the model name
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if model_name in line.lower():
                # Try to extract label from nearby lines
                for j in range(max(0, i - 3), min(len(lines), i + 3)):
                    if "label:" in lines[j].lower() or "**label**:" in lines[j].lower():
                        # Extract label
                        label_line = lines[j]
                        if "`" in label_line:
                            start = label_line.find("`") + 1
                            end = label_line.find("`", start)
                            if start > 0 and end > start:
                                references.append(label_line[start:end])
                        break

    except Exception as e:
        print(f"Warning: Could not read glossary: {e}", file=sys.stderr)

    return list(set(references))  # Remove duplicates


# =============================================================================
# SCHEMA GENERATION
# =============================================================================


def generate_complete_schema(
    output_path: Path | None = None,
    include_examples: bool = True,
) -> dict[str, Any]:
    """
    Generate complete LLM-optimized schema JSON.

    Args:
        output_path: Optional path to write JSON file
        include_examples: Whether to include example instances

    Returns:
        Complete schema dictionary
    """
    print("🔍 Discovering Pydantic models...")
    models_by_module = discover_all_models()

    # Flatten all models
    all_models = []
    for models in models_by_module.values():
        all_models.extend(models)

    # Remove duplicates
    seen = set()
    unique_models = []
    for model in all_models:
        name = get_model_name(model)
        if name not in seen:
            seen.add(name)
            unique_models.append(model)

    print(f"✓ Found {len(unique_models)} unique models")

    print("📊 Building dependency graph...")
    dep_graph = build_dependency_graph(unique_models)
    inverse_graph = build_inverse_dependency_graph(dep_graph)
    sorted_names = topological_sort_models(dep_graph)
    print("✓ Dependency graph built")

    print("📝 Extracting metadata...")
    schemas_by_dependency = {}
    glossary_path = Path(__file__).parent.parent.parent.parent / "docs" / "glossary.md"

    # Group by dependency level
    dependency_levels = {}
    for name in sorted_names:
        # Find the model
        model = next((m for m in unique_models if get_model_name(m) == name), None)
        if model is None:
            continue

        # Determine level based on number of dependencies
        num_deps = len(dep_graph.get(name, []))
        if num_deps == 0:
            level = 0  # Primitives
        elif num_deps <= 2:
            level = 1  # Basic
        elif num_deps <= 5:
            level = 2  # Intermediate
        else:
            level = 3  # Complex

        if level not in dependency_levels:
            dependency_levels[level] = []

        metadata = extract_model_metadata(model)
        metadata["dependencies"] = dep_graph.get(name, [])
        metadata["used_by"] = inverse_graph.get(name, [])
        metadata["glossary_refs"] = link_to_glossary(model, glossary_path)

        if include_examples:
            metadata["example_instance"] = generate_example_instance(model)

        dependency_levels[level].append(metadata)

    # Convert to named levels
    level_names = {
        0: "level_0_primitives",
        1: "level_1_basic",
        2: "level_2_intermediate",
        3: "level_3_complex",
    }

    for level, name in level_names.items():
        if level in dependency_levels:
            schemas_by_dependency[name] = dependency_levels[level]

    print(f"✓ Metadata extracted for {len(unique_models)} models")

    # Build complete schema
    schema = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0.0",
            "purpose": "LLM reference for mathematical object creation via agentic pipeline",
            "total_schemas": len(unique_models),
            "source_modules": list(models_by_module.keys()),
        },
        "workflow_guide": {
            "overview": "Fragile proof pipeline: Create mathematical objects → Define theorems → Write mathster → Validate → Submit for dual review (Gemini 2.5 Pro + Codex) → Register",
            "dual_review_protocol": "CLAUDE.md § Collaborative Review Workflow (Step 2: Dual Independent Review via MCP)",
            "steps": [
                "1. Create MathematicalObjects with properties using core.pipeline_types",
                "2. Define TheoremBox specifying input objects, required properties, and established outputs",
                "3. Write ProofBox using compositional ProofSteps (direct_derivation, sub_proof, lemma_application)",
                "4. Validate proof against theorem using validate_proof_for_theorem()",
                "5. Submit to Gemini 2.5 Pro via mcp__gemini-cli__ask-gemini (model: gemini-2.5-pro)",
                "6. Submit to Codex via mcp__codex__codex for independent review",
                "7. Compare reviews: consensus issues (high confidence), discrepancies (verify manually), unique issues (medium confidence)",
                "8. Implement fixes after critical evaluation and cross-checking against docs/glossary.md",
                "9. Register in MathematicalRegistry for persistence and querying",
                "10. Build RelationshipGraph for analysis and visualization",
            ],
            "best_practices": [
                "Use property-level granularity: ProofInputs specify exact properties needed, not entire objects",
                "Always validate dataflow before submission: proof.validate_dataflow() must return []",
                "Cross-reference to docs/glossary.md (683 entries) for notation consistency",
                "Use dual review (Gemini 2.5 Pro + Codex) with IDENTICAL prompts for rigor checking",
                "Maintain referential integrity: all object/theorem/proof IDs must exist in registry",
                "Follow ID conventions: obj-*, thm-*, proof-*, rel-*-*-* (kebab-case)",
                "When reviewers contradict, manually verify against framework docs (potential hallucination)",
                "Never accept claims that cannot be verified in docs/glossary.md or source documents",
            ],
        },
        "schemas_by_dependency": schemas_by_dependency,
        "dependency_graph": dep_graph,
        "inverse_dependency_graph": inverse_graph,
        "common_workflows": {
            "create_simple_theorem": {
                "description": "Create a basic theorem with one input object, one output property",
                "code_example": "See examples/complete_integration_example.py lines 100-150",
                "schema_refs": ["TheoremBox", "MathematicalObject", "Property"],
            },
            "create_compositional_proof": {
                "description": "Multi-step proof with sub-mathster and property-level dataflow",
                "code_example": "See examples/proof_system_example.py - complete hierarchical proof structure",
                "schema_refs": [
                    "ProofBox",
                    "ProofStep",
                    "DirectDerivation",
                    "SubProofReference",
                    "ProofInput",
                    "ProofOutput",
                ],
            },
            "validate_and_submit": {
                "description": "Validate proof against theorem, submit for dual review, implement fixes",
                "workflow": [
                    "1. result = validate_proof_for_theorem(proof, theorem)",
                    "2. If result.is_valid: proceed, else fix issues first",
                    "3. mcp__gemini-cli__ask-gemini(prompt='Review proof...', model='gemini-2.5-pro')",
                    "4. mcp__codex__codex(prompt='<identical prompt>', ...)",
                    "5. Compare reviews: consensus (high conf), discrepancies (verify), unique (medium conf)",
                    "6. Cross-check suggestions against docs/glossary.md before implementing",
                    "7. If disagreeing with reviewers, document reasoning and inform user",
                ],
            },
        },
        "validation_rules": {
            "id_conventions": {
                "objects": "obj-{kebab-case} (e.g., obj-euclidean-gas-discrete)",
                "theorems": "thm-{kebab-case} (e.g., thm-mean-field-limit)",
                "mathster": "proof-{theorem-label} (e.g., proof-thm-mean-field-limit)",
                "relationships": "rel-{source}-{target}-{type} (e.g., rel-discrete-continuous-equivalence)",
                "properties": "prop-{kebab-case} (e.g., prop-lipschitz-potential)",
            },
            "referential_integrity": "All referenced IDs (in relationships, theorems, mathster) must exist in MathematicalRegistry. Use registry.validate_referential_integrity() to check.",
            "property_granularity": "ProofInputs/ProofOutputs use PropertyReference to specify exact properties needed from objects, enabling fine-grained dataflow validation.",
            "dataflow_validation": "proof.validate_dataflow() checks that all step inputs are satisfied by previous step outputs or proof-level inputs. Must return empty list.",
        },
        "documentation_index": {
            "glossary": "docs/glossary.md (683 mathematical entries across Euclidean Gas and Geometric Gas chapters)",
            "claude_guide": "CLAUDE.md § Mathematical Proofing and Documentation (includes dual review protocol)",
            "examples": "examples/README.md (9 comprehensive examples with 400+ lines of documentation)",
            "lean_guide": "docs/LEAN_EMULATION_GUIDE.md (Lean-compatible design patterns)",
            "framework_docs": [
                "docs/source/1_euclidean_gas/ - Euclidean Gas framework (12 documents)",
                "docs/source/2_geometric_gas/ - Geometric Gas framework (advanced topics)",
            ],
        },
    }

    # Write to file if path provided
    if output_path is not None:
        print(f"💾 Writing schema to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        print("✓ Schema written successfully")

    return schema


# =============================================================================
# FOCUSED SCHEMA GENERATORS
# =============================================================================


def generate_proof_schema(
    output_path: Path | None = None,
    include_examples: bool = True,
) -> dict[str, Any]:
    """
    Generate focused schema for rigorous proof writing.

    This schema contains ONLY the models needed for writing rigorous,
    publishable mathematical mathster with full validation.

    Includes:
    - Core proof types (10 models): ProofBox, ProofStep, etc.
    - Mathematical objects (8 models): MathematicalObject, Property, etc.
    - SymPy validation system (9 models): DualExpr, SymPyValidator, etc.
    - Integration utilities (3 models): ProofValidationResult, etc.
    - Supporting types (2 models): Ok, Err

    Total: ~32 models (vs 76 in full schema)

    Args:
        output_path: Optional path to write JSON file
        include_examples: Whether to include example instances

    Returns:
        Focused schema dictionary optimized for rigorous proof writing
    """
    print("🔍 Generating rigorous proof schema...")

    # Get complete schema first
    complete_schema = generate_complete_schema(output_path=None, include_examples=include_examples)

    # Define whitelist of models for rigorous mathster
    proof_models_whitelist = {
        # Core proof types (10)
        "ProofBox",
        "ProofStep",
        "ProofInput",
        "ProofOutput",
        "PropertyReference",
        "AssumptionReference",
        "DirectDerivation",
        "SubProofReference",
        "LemmaApplication",
        "ProofStepType",
        "ProofStepStatus",
        # Mathematical objects (8)
        "MathematicalObject",
        "Property",
        "PropertyEvent",
        "PropertyRefinement",
        "TheoremBox",
        "Relationship",
        "RelationshipProperty",
        "Axiom",
        # SymPy validation system (9) - CRITICAL for rigorous mathster
        "DualExpr",
        "DualStatement",
        "SymPyContext",
        "SymPyValidator",
        "ValidationResult",
        "ValidationIssue",
        "Transformation",
        "SymbolDeclaration",
        "PluginRegistry",
        # Integration utilities (3)
        "ProofValidationResult",
        "ProofTheoremMismatch",
        "ProofExpansionRequest",
        # Supporting types and enums
        "Ok",
        "Err",
        "ObjectType",
        "TheoremOutputType",
        "RelationType",
        "PropertyEventType",
        "RefinementType",
        "ValidationStatus",
        "TransformationType",
    }

    # Filter schemas
    filtered_schemas = {}
    for level_name, models in complete_schema["schemas_by_dependency"].items():
        filtered_models = [m for m in models if m["name"] in proof_models_whitelist]
        if filtered_models:
            filtered_schemas[level_name] = filtered_models

    # Build focused schema
    schema = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0.0",
            "purpose": "LLM reference for writing rigorous mathematical mathster",
            "task_focus": "rigorous_proofs",
            "quality_standard": "Top-tier journal (Annals of Mathematics, JAMS)",
            "total_schemas": sum(len(models) for models in filtered_schemas.values()),
            "excluded_from_full": complete_schema["metadata"]["total_schemas"]
            - sum(len(models) for models in filtered_schemas.values()),
        },
        "task_guide": {
            "objective": "Create rigorous, publishable mathematical mathster with full validation",
            "when_to_use": "When writing complete, peer-review-ready mathster with full mathematical derivations",
            "quality_standard": "Every claim must be proven. Every step must be justified. SymPy validation where possible.",
            "review_process": "MANDATORY dual review by Gemini 2.5 Pro + Codex before acceptance",
            "key_difference_from_sketches": "Includes full SymPy validation system for mathematical correctness checking",
        },
        "workflow": {
            "steps": [
                "1. Understand theorem: Read TheoremBox to see what properties are required and what we establish",
                "2. Plan proof strategy: Outline high-level approach (3-5 key steps)",
                "3. Create ProofBox: Define property-level ProofInputs (exact properties needed, not whole objects)",
                "4. Write each ProofStep sequentially:",
                "   - DirectDerivation: MUST include full mathematical content in `mathematical_content` field",
                "   - Use DualExpr for LaTeX + SymPy dual representation where possible",
                "   - Apply Transformations step-by-step to show mathematical reasoning",
                "   - Reference specific properties using PropertyReference (not object IDs)",
                "5. Validate dataflow: proof.validate_dataflow() must return empty list",
                "6. Validate against theorem: validate_proof_for_theorem(proof, theorem)",
                "7. SymPy validation: Use SymPyValidator for mathematical correctness where applicable",
                "8. Dual review (MANDATORY):",
                "   - Submit to Gemini 2.5 Pro: mcp__gemini-cli__ask-gemini(model='gemini-2.5-pro')",
                "   - Submit to Codex: mcp__codex__codex (use IDENTICAL prompt)",
                "   - Compare: consensus issues (high confidence), discrepancies (verify manually), unique (medium confidence)",
                "9. Cross-check all suggestions against docs/glossary.md before implementing",
                "10. Mark steps as ProofStepStatus.VERIFIED after validation passes",
            ],
            "best_practices": [
                "ALWAYS include full mathematical derivation in DirectDerivation steps",
                "Use DualExpr (LaTeX + SymPy) for expressions that can be validated",
                "Never mark a step as VERIFIED without validation (SymPy or dual review)",
                "Property-level granularity: specify exact properties, not entire objects",
                "Dataflow first: ensure proof.validate_dataflow() returns [] before dual review",
                "Dual review with IDENTICAL prompts: prevents inconsistent feedback",
                "When reviewers contradict, manually verify against docs/glossary.md (potential hallucination)",
                "Cross-reference notation to docs/glossary.md (683 entries) for consistency",
                "Use SubProofReference for complex steps (hierarchical mathster)",
                "All claims must be proven - no SKETCHED steps in final proof",
            ],
            "common_patterns": [
                "Induction mathster: Base case + inductive step as separate ProofSteps",
                "Contradiction mathster: Assume negation → derive contradiction → conclude original",
                "Constructive mathster: Build object step-by-step, verify properties",
                "Hierarchical mathster: Main proof with SubProofReference for technical lemmas",
                "SymPy-validated derivations: Use Transformation system for step-by-step validation",
            ],
        },
        "schemas": filtered_schemas,
        "examples": {
            "minimal_direct_proof": {
                "description": "Single-step proof with DirectDerivation",
                "use_case": "Simple theorems requiring one direct argument",
                "example_structure": {
                    "proof_id": "proof-thm-simple",
                    "proves": "thm-simple",
                    "steps": [
                        {
                            "step_type": "DirectDerivation",
                            "status": "EXPANDED",
                            "mathematical_content": "Full LaTeX derivation here",
                        }
                    ],
                },
                "code_reference": "See examples/complete_integration_example.py",
            },
            "hierarchical_proof": {
                "description": "Multi-level proof with sub-mathster",
                "use_case": "Complex theorems requiring modular proof structure",
                "example_structure": {
                    "main_proof": "proof-thm-main",
                    "steps": [
                        {"step_type": "SubProofReference", "sub_proof_id": "proof-lem-technical"},
                        {"step_type": "DirectDerivation", "mathematical_content": "..."},
                    ],
                    "sub_proofs": {"proof-lem-technical": "..."},
                },
                "code_reference": "See examples/proof_system_example.py lines 50-200",
            },
            "sympy_validated_proof": {
                "description": "Proof with SymPy mathematical validation",
                "use_case": "Algebraic/analytical theorems with symbolic computation",
                "workflow": [
                    "1. Define DualExpr for each mathematical expression",
                    "2. Use Transformation to show step-by-step derivation",
                    "3. SymPyValidator.validate() to check correctness",
                    "4. Include ValidationResult in proof documentation",
                ],
                "code_reference": "See examples/sympy_integration_example.py",
            },
        },
        "validation_checklist": [
            "[ ] All ProofInputs specify exact properties (PropertyReference), not object IDs",
            "[ ] All ProofOutputs establish new properties with PropertyReference",
            "[ ] proof.validate_dataflow() returns empty list (no missing inputs)",
            "[ ] validate_proof_for_theorem() returns is_valid=True",
            "[ ] All DirectDerivation steps have non-empty mathematical_content",
            "[ ] SymPy validation used where applicable (algebraic/analytical steps)",
            "[ ] Dual review completed: Gemini 2.5 Pro + Codex with IDENTICAL prompts",
            "[ ] All reviewer suggestions cross-checked against docs/glossary.md",
            "[ ] No ProofStepStatus.SKETCHED in final proof (all EXPANDED or VERIFIED)",
            "[ ] All notation consistent with docs/glossary.md (683 entries)",
            "[ ] Proof claims exactly match theorem requirements (no more, no less)",
            "[ ] All IDs follow conventions: proof-{thm-label}, prop-{name}, obj-{name}",
        ],
        "common_pitfalls": [
            "❌ Marking step as VERIFIED without actual validation",
            "❌ Using object_id instead of PropertyReference in ProofInput",
            "❌ Empty mathematical_content in DirectDerivation (must include full derivation)",
            "❌ Skipping SymPy validation for algebraic steps (use DualExpr + Transformation)",
            "❌ Accepting reviewer suggestions without cross-checking docs/glossary.md",
            "❌ Using different prompts for Gemini vs Codex (must be IDENTICAL)",
            "❌ Leaving ProofStepStatus.SKETCHED in supposedly rigorous proof",
            "❌ Claiming more than theorem requires (scope creep)",
            "❌ Forgetting to validate dataflow before dual review (waste of reviewer time)",
            "❌ Inventing notation not in docs/glossary.md (breaks consistency)",
        ],
        "documentation_refs": {
            "examples": {
                "proof_system_example.py": "Hierarchical proof structure with sub-mathster (lines 37-400)",
                "complete_integration_example.py": "Theorem → Proof → Validation workflow (lines 100-300)",
                "sympy_integration_example.py": "SymPy validation and DualExpr usage (full file)",
            },
            "guides": {
                "CLAUDE.md": "§ Mathematical Proofing and Documentation (dual review protocol)",
                "examples/README.md": "Complete proof system documentation (400+ lines)",
                "LEAN_EMULATION_GUIDE.md": "Lean-compatible design patterns",
            },
            "glossary": "docs/glossary.md (683 mathematical entries - USE THIS for notation)",
            "framework_docs": [
                "docs/source/1_euclidean_gas/ - Euclidean Gas framework (12 documents)",
                "docs/source/2_geometric_gas/ - Geometric Gas framework",
            ],
        },
    }

    # Write to file if path provided
    if output_path is not None:
        print(f"💾 Writing rigorous proof schema to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        print("✓ Rigorous proof schema written successfully")
        print(f"  Models included: {schema['metadata']['total_schemas']}")
        print(f"  Models excluded: {schema['metadata']['excluded_from_full']}")

    return schema


def generate_sketch_schema(
    output_path: Path | None = None,
    include_examples: bool = True,
) -> dict[str, Any]:
    """
    Generate focused schema for proof sketch writing.

    This schema contains ONLY the models needed for writing proof sketches
    that outline strategy and structure without full mathematical derivation.

    Includes:
    - Core proof types (10 models): ProofBox, ProofStep, etc.
    - Mathematical objects (7 models): MathematicalObject, Property, etc.
    - Integration utilities (2 models): ProofValidationResult, ProofExpansionRequest
    - Supporting types (4 models): Ok, Err, enums

    Total: ~23 models (vs 76 in full schema)

    EXCLUDES: SymPy validation system (not needed for sketches)

    Args:
        output_path: Optional path to write JSON file
        include_examples: Whether to include example instances

    Returns:
        Focused schema dictionary optimized for proof sketch writing
    """
    print("🔍 Generating proof sketch schema...")

    # Get complete schema first
    complete_schema = generate_complete_schema(output_path=None, include_examples=include_examples)

    # Define whitelist of models for proof sketches
    sketch_models_whitelist = {
        # Core proof types (10) - SAME as rigorous
        "ProofBox",
        "ProofStep",
        "ProofInput",
        "ProofOutput",
        "PropertyReference",
        "AssumptionReference",
        "DirectDerivation",
        "SubProofReference",
        "LemmaApplication",
        "ProofStepType",
        "ProofStepStatus",
        # Mathematical objects (7) - lighter than rigorous
        "MathematicalObject",
        "Property",
        "PropertyEvent",
        "TheoremBox",
        "Relationship",
        "Axiom",
        "Parameter",
        # EXCLUDED: PropertyRefinement, RelationshipProperty (less detail needed)
        # Integration utilities (2) - focused on expansion workflow
        "ProofValidationResult",  # Basic structural validation
        "ProofExpansionRequest",  # KEY for sketch → rigorous workflow!
        # EXCLUDED: ProofTheoremMismatch (less relevant for sketches)
        # Supporting types and enums (4)
        "Ok",
        "Err",
        "ProofStepType",
        "ProofStepStatus",
        "ObjectType",
        "TheoremOutputType",
        # NO SymPy validation enums!
    }

    # Filter schemas
    filtered_schemas = {}
    for level_name, models in complete_schema["schemas_by_dependency"].items():
        filtered_models = [m for m in models if m["name"] in sketch_models_whitelist]
        if filtered_models:
            filtered_schemas[level_name] = filtered_models

    # Build focused schema
    schema = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0.0",
            "purpose": "LLM reference for writing proof sketches",
            "task_focus": "proof_sketches",
            "quality_standard": "Correct structure, clear strategy, valid dataflow (no full derivations required)",
            "total_schemas": sum(len(models) for models in filtered_schemas.values()),
            "excluded_from_full": complete_schema["metadata"]["total_schemas"]
            - sum(len(models) for models in filtered_schemas.values()),
        },
        "task_guide": {
            "objective": "Create proof sketches outlining strategy and structure without full mathematical derivation",
            "when_to_use": "When planning proof approach, identifying key steps, or deferring complex sub-mathster for later expansion",
            "quality_standard": "Clear strategy, correct dataflow, all key steps identified (even if not fully proven)",
            "review_process": "Structural validation, strategy review (dual review optional for sketches)",
            "key_difference_from_rigorous": "NO SymPy validation. Focus on STRUCTURE not DERIVATION. ProofStepStatus.SKETCHED encouraged.",
        },
        "workflow": {
            "steps": [
                "1. Understand theorem: Read TheoremBox to see what we need to prove",
                "2. Draft high-level strategy: Outline 3-5 key steps (approach, not full proof)",
                "3. Create ProofBox: Define property-level ProofInputs (same structure as rigorous)",
                "4. Write each ProofStep (SKETCHED mode):",
                "   - DirectDerivation: Brief description of approach (NOT full mathematical content)",
                "   - SubProofReference: Identify complex steps to defer for later expansion",
                "   - LemmaApplication: Identify key results needed (even if not proven yet)",
                "   - Set status=ProofStepStatus.SKETCHED for all initial steps",
                "5. Validate dataflow: proof.validate_dataflow() must return empty list (even for sketches!)",
                "6. Use ProofEngine.get_expansion_requests() to identify what needs expansion later",
                "7. Review sketch for structural correctness:",
                "   - Does the dataflow make sense?",
                "   - Are all key steps identified?",
                "   - Is the proof strategy clear?",
                "8. Iterate: expand sketches → rigorous mathster using llm_proof.json",
            ],
            "best_practices": [
                "Use ProofStepStatus.SKETCHED liberally - this is a SKETCH, not final proof",
                "DirectDerivation: describe approach, not full derivation (e.g., 'Apply Gronwall inequality')",
                "SubProofReference: defer complex sub-arguments for later expansion",
                "LemmaApplication: identify key results needed (helps planning)",
                "Dataflow still matters: proof.validate_dataflow() must return []",
                "Property-level granularity: still use PropertyReference (good practice)",
                "Don't claim you've proven something if you've only sketched it",
                "Use ProofExpansionRequest to track what needs work",
                "ProofEngine workflow: register proof → get expansion requests → expand systematically",
                "Good sketches become rigorous mathster by expanding SKETCHED steps one by one",
            ],
            "common_patterns": [
                "Proof strategy outline: 3-5 SKETCHED steps showing overall approach",
                "Deferred complexity: Main proof with SubProofReference for technical details",
                "Lemma identification: Mark which results you need (LemmaApplication)",
                "Incremental expansion: Expand one SKETCHED step at a time to EXPANDED",
                "Sketch → Rigorous pipeline: Use ProofEngine to manage expansion process",
            ],
        },
        "schemas": filtered_schemas,
        "examples": {
            "simple_sketch": {
                "description": "3-step sketch with SKETCHED status",
                "use_case": "Initial proof planning before full development",
                "example_structure": {
                    "proof_id": "proof-thm-sketch",
                    "proves": "thm-sketch",
                    "steps": [
                        {
                            "step_id": "step-1",
                            "description": "Establish well-posedness via contraction mapping",
                            "step_type": "DirectDerivation",
                            "status": "SKETCHED",  # ← Key for sketches!
                        },
                        {
                            "step_id": "step-2",
                            "description": "Construct coupling between systems",
                            "step_type": "SubProofReference",
                            "status": "SKETCHED",
                        },
                    ],
                },
                "code_reference": "See examples/proof_system_example.py lines 150-200",
            },
            "hierarchical_sketch": {
                "description": "Main sketch with deferred sub-mathster",
                "use_case": "Complex mathster where sub-arguments need separate treatment",
                "workflow": [
                    "1. Sketch main proof with 3-5 high-level steps",
                    "2. Identify complex steps → mark as SubProofReference",
                    "3. Create placeholder sub-mathster (also SKETCHED)",
                    "4. Use ProofEngine to track expansion requests",
                    "5. Expand sub-mathster first, then main proof",
                ],
                "code_reference": "See examples/proof_system_example.py full workflow",
            },
            "sketch_to_rigorous_progression": {
                "description": "How to expand a sketch into rigorous proof",
                "workflow": [
                    "1. Start with SKETCHED proof (use llm_sketch.json)",
                    "2. Validate dataflow: proof.validate_dataflow() == []",
                    "3. Register with ProofEngine: engine.register_proof(proof)",
                    "4. Get expansion requests: engine.get_expansion_requests(proof_id)",
                    "5. For each request: expand step from SKETCHED → EXPANDED",
                    "6. Switch to llm_proof.json for full derivations with SymPy",
                    "7. Mark expanded steps as VERIFIED after dual review",
                ],
                "tools": "ProofEngine, ProofExpansionRequest (see schemas above)",
            },
        },
        "validation_checklist": [
            "[ ] All ProofInputs use PropertyReference (even in sketches - good practice)",
            "[ ] proof.validate_dataflow() returns empty list (structural correctness)",
            "[ ] All key proof steps identified (even if SKETCHED)",
            "[ ] Proof strategy is clear and makes sense",
            "[ ] Complex steps deferred to SubProofReference (good decomposition)",
            "[ ] ProofStepStatus.SKETCHED used appropriately (honest about incompleteness)",
            "[ ] ProofExpansionRequest workflow understood (how to expand later)",
            "[ ] NOT claiming more than actually proven (SKETCHED ≠ VERIFIED)",
        ],
        "common_pitfalls": [
            "❌ Claiming sketch is rigorous proof (SKETCHED ≠ VERIFIED)",
            "❌ Skipping dataflow validation (even sketches need valid structure)",
            "❌ Not using SubProofReference for complex steps (try to do everything in one step)",
            "❌ Forgetting to use ProofEngine for tracking expansion requests",
            "❌ Using PropertyReference inconsistently (use it even in sketches)",
            "❌ Vague step descriptions (be specific about approach, even if not full derivation)",
            "❌ Not identifying key lemmas needed (LemmaApplication helps planning)",
            "❌ Treating sketch as final proof (it's a PLAN, not the PROOF)",
        ],
        "documentation_refs": {
            "examples": {
                "proof_system_example.py": "Sketch workflow with ProofEngine (lines 150-250, 500-550)",
                "complete_integration_example.py": "Basic proof structure",
            },
            "guides": {
                "examples/README.md": "Proof system documentation (see 'Compositional Proofs' section)",
                "CLAUDE.md": "§ Mathematical Proofing (focus on sketch → rigorous workflow)",
            },
            "expansion_workflow": "Use ProofEngine + ProofExpansionRequest to systematically expand sketches",
            "rigorous_transition": "Once sketch validated, use llm_proof.json to expand with full derivations",
        },
    }

    # Write to file if path provided
    if output_path is not None:
        print(f"💾 Writing proof sketch schema to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        print("✓ Proof sketch schema written successfully")
        print(f"  Models included: {schema['metadata']['total_schemas']}")
        print(f"  Models excluded: {schema['metadata']['excluded_from_full']}")
        print("  Key difference: NO SymPy validation system (sketches focus on structure)")

    return schema


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main() -> None:
    """CLI entry point for schema generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LLM-optimized schemas for Fragile proof system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all three schemas (full, proof, sketch)
  python -m fragile.mathster.schema_generator --all

  # Generate rigorous proof schema only
  python -m fragile.mathster.schema_generator --proof

  # Generate proof sketch schema only
  python -m fragile.mathster.schema_generator --sketch

  # Generate full schema (default)
  python -m fragile.mathster.schema_generator

Output files:
  --all:    llm_schemas.json + llm_proof.json + llm_sketch.json
  --proof:  llm_proof.json
  --sketch: llm_sketch.json
  default:  llm_schemas.json
        """,
    )

    # Schema type selection (mutually exclusive)
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--all",
        action="store_true",
        help="Generate all three schemas (full, proof, sketch)",
    )
    schema_group.add_argument(
        "--proof",
        action="store_true",
        help="Generate rigorous proof schema only (llm_proof.json)",
    )
    schema_group.add_argument(
        "--sketch",
        action="store_true",
        help="Generate proof sketch schema only (llm_sketch.json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Custom output path (only for single schema generation)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Exclude example instances (smaller file)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FRAGILE PROOF SYSTEM - LLM SCHEMA GENERATOR")
    print("=" * 80)
    print()

    # Determine output directory
    base_dir = Path(__file__).parent

    if args.all:
        # Generate all three schemas
        print("📦 Generating ALL schemas (full, proof, sketch)...")
        print()

        # Full schema
        full_schema = generate_complete_schema(
            output_path=base_dir / "llm_schemas.json",
            include_examples=not args.no_examples,
        )
        print()

        # Proof schema
        proof_schema = generate_proof_schema(
            output_path=base_dir / "llm_proof.json",
            include_examples=not args.no_examples,
        )
        print()

        # Sketch schema
        sketch_schema = generate_sketch_schema(
            output_path=base_dir / "llm_sketch.json",
            include_examples=not args.no_examples,
        )

        print()
        print("=" * 80)
        print("ALL SCHEMAS GENERATED")
        print("=" * 80)
        print(
            f"📄 llm_schemas.json:  {full_schema['metadata']['total_schemas']} models, "
            f"{(base_dir / 'llm_schemas.json').stat().st_size / 1024:.1f} KB"
        )
        print(
            f"📄 llm_proof.json:    {proof_schema['metadata']['total_schemas']} models, "
            f"{(base_dir / 'llm_proof.json').stat().st_size / 1024:.1f} KB (rigorous mathster)"
        )
        print(
            f"📄 llm_sketch.json:   {sketch_schema['metadata']['total_schemas']} models, "
            f"{(base_dir / 'llm_sketch.json').stat().st_size / 1024:.1f} KB (proof sketches)"
        )

    elif args.proof:
        # Generate proof schema only
        output_path = args.output or base_dir / "llm_proof.json"
        schema = generate_proof_schema(
            output_path=output_path,
            include_examples=not args.no_examples,
        )

        print()
        print("=" * 80)
        print("RIGOROUS PROOF SCHEMA COMPLETE")
        print("=" * 80)
        print(f"Total schemas: {schema['metadata']['total_schemas']}")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        print()
        print("✓ Ready for rigorous proof writing with SymPy validation")

    elif args.sketch:
        # Generate sketch schema only
        output_path = args.output or base_dir / "llm_sketch.json"
        schema = generate_sketch_schema(
            output_path=output_path,
            include_examples=not args.no_examples,
        )

        print()
        print("=" * 80)
        print("PROOF SKETCH SCHEMA COMPLETE")
        print("=" * 80)
        print(f"Total schemas: {schema['metadata']['total_schemas']}")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        print()
        print("✓ Ready for proof sketch writing (structure-focused)")

    else:
        # Generate full schema (default)
        output_path = args.output or base_dir / "llm_schemas.json"
        schema = generate_complete_schema(
            output_path=output_path,
            include_examples=not args.no_examples,
        )

        print()
        print("=" * 80)
        print("FULL SCHEMA GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total schemas: {schema['metadata']['total_schemas']}")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        print()
        print("✓ Ready for LLM consumption in agentic math pipeline")


if __name__ == "__main__":
    main()
