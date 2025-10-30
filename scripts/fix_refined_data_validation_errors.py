#!/usr/bin/env python3
"""
Fix refined_data validation errors found by enriched schema validation.

Fixes:
1. Axioms: Add mathematical_expression field (extract from statement)
2. Remarks: Rename full_text → content
3. Theorems/Corollaries: Convert lemma_dag_edges from {source, target} objects to [from, to] pairs
4. Proofs: Report incomplete stubs for manual review
"""

import json
from pathlib import Path
import re
from typing import Any


def extract_mathematical_expression(statement: str) -> str:
    """
    Extract mathematical expressions from axiom statement.

    Looks for LaTeX math blocks ($$...$$) or inline math ($...$).
    Falls back to a truncated version of the statement.
    """
    # Try to find display math blocks
    display_math = re.findall(r"\$\$(.*?)\$\$", statement, re.DOTALL)
    if display_math:
        # Join all display math blocks
        return " ".join(expr.strip() for expr in display_math)

    # Try to find inline math
    inline_math = re.findall(r"\$(.*?)\$", statement)
    if inline_math:
        return ", ".join(expr.strip() for expr in inline_math[:3])  # First 3

    # Fallback: extract key mathematical symbols/terms
    # Look for common patterns like >= 0, x ∈ X, etc.
    math_patterns = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*\s*[∈≥≤=<>]\s*[^.]+", statement)
    if math_patterns:
        return "; ".join(p.strip() for p in math_patterns[:2])

    # Last resort: use first sentence
    first_sentence = statement.split(".")[0].strip()
    return first_sentence[:200] if len(first_sentence) <= 200 else first_sentence[:197] + "..."


def fix_axiom(file_path: Path, data: dict[str, Any]) -> bool:
    """Fix axiom by adding mathematical_expression field."""
    if data.get("mathematical_expression"):
        return False  # Already has it

    # Extract from statement
    statement = data.get("statement", "") or data.get("full_text", "")
    if not statement:
        print(f"  ⚠️  {file_path.name}: No statement found, skipping")
        return False

    math_expr = extract_mathematical_expression(statement)
    data["mathematical_expression"] = math_expr

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ {file_path.name}: Added mathematical_expression")
    return True


def fix_remark(file_path: Path, data: dict[str, Any]) -> bool:
    """Fix remark by renaming full_text → content."""
    if data.get("content"):
        return False  # Already has it

    full_text = data.get("full_text", "")
    if not full_text:
        print(f"  ⚠️  {file_path.name}: No full_text found, skipping")
        return False

    # Rename full_text → content
    data["content"] = full_text
    if "full_text" in data:
        del data["full_text"]

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ {file_path.name}: Renamed full_text → content")
    return True


def fix_lemma_dag_edges(file_path: Path, data: dict[str, Any]) -> bool:
    """Convert lemma_dag_edges from [{source, target}] to [[from, to]] format."""
    edges = data.get("lemma_dag_edges", [])
    if not edges:
        return False

    # Check if already in correct format (list of lists)
    if edges and isinstance(edges[0], list):
        return False  # Already correct

    # Check if in wrong format (list of objects)
    if not isinstance(edges[0], dict):
        print(f"  ⚠️  {file_path.name}: lemma_dag_edges has unexpected format: {type(edges[0])}")
        return False

    # Convert from [{source, target}] to [[from, to]]
    converted_edges = []
    for edge in edges:
        if isinstance(edge, dict) and "source" in edge and "target" in edge:
            converted_edges.append([edge["source"], edge["target"]])
        else:
            print(f"  ⚠️  {file_path.name}: Skipping malformed edge: {edge}")

    if not converted_edges:
        print(f"  ⚠️  {file_path.name}: No valid edges to convert")
        return False

    data["lemma_dag_edges"] = converted_edges

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"  ✅ {file_path.name}: Converted {len(converted_edges)} lemma_dag_edges to [[from, to]] format"
    )
    return True


def check_incomplete_proof(file_path: Path, data: dict[str, Any]) -> bool:
    """Check if proof is incomplete (missing required fields)."""
    required_fields = ["proof_id", "inputs", "outputs", "strategy", "steps"]
    missing_fields = [field for field in required_fields if field not in data or not data[field]]

    if missing_fields:
        print(f"  ⚠️  {file_path.name}: Missing required fields: {', '.join(missing_fields)}")
        return True

    return False


def main():
    """Main fix routine."""
    refined_data_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data")

    if not refined_data_dir.exists():
        print(f"Error: refined_data directory not found: {refined_data_dir}")
        return

    print("=" * 70)
    print("Fixing Refined Data Validation Errors")
    print("=" * 70)

    # 1. Fix axioms
    print("\n📝 Fixing Axioms (adding mathematical_expression)...")
    axioms_dir = refined_data_dir / "axioms"
    axiom_files = ["axiom-bounded-second-moment-perturbation.json", "axiom-rescale-function.json"]

    axioms_fixed = 0
    for filename in axiom_files:
        file_path = axioms_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  {filename}: File not found")
            continue

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if fix_axiom(file_path, data):
            axioms_fixed += 1

    print(f"\n  Total: {axioms_fixed}/{len(axiom_files)} axioms fixed")

    # 2. Fix remarks
    print("\n📝 Fixing Remarks (renaming full_text → content)...")
    remarks_dir = refined_data_dir / "remarks"
    remark_files = list(remarks_dir.glob("*.json")) if remarks_dir.exists() else []

    remarks_fixed = 0
    for file_path in remark_files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if fix_remark(file_path, data):
            remarks_fixed += 1

    print(f"\n  Total: {remarks_fixed}/{len(remark_files)} remarks fixed")

    # 3. Fix lemma_dag_edges
    print("\n📝 Fixing lemma_dag_edges (converting to [[from, to]] format)...")
    theorems_dir = refined_data_dir / "theorems"
    theorem_files_to_fix = [
        "cor-pipeline-continuity-margin-stability.json",
        "cor-chain-rule-sigma-reg-var.json",
        "cor-closed-form-lipschitz-composite.json",
        "thm-cloning-transition-operator-continuity-recorrected.json",
    ]

    edges_fixed = 0
    for filename in theorem_files_to_fix:
        file_path = theorems_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  {filename}: File not found")
            continue

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if fix_lemma_dag_edges(file_path, data):
            edges_fixed += 1

    print(f"\n  Total: {edges_fixed}/{len(theorem_files_to_fix)} files fixed")

    # 4. Check incomplete proofs
    print("\n📝 Checking Incomplete Proofs...")
    proofs_dir = refined_data_dir / "proofs"
    proof_files = list(proofs_dir.glob("*.json")) if proofs_dir.exists() else []

    incomplete_proofs = []
    for file_path in proof_files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if check_incomplete_proof(file_path, data):
            incomplete_proofs.append(file_path.name)

    print(f"\n  Total: {len(incomplete_proofs)}/{len(proof_files)} proofs are incomplete")

    if incomplete_proofs:
        print("\n  💡 Recommendation: These proof stubs should either be:")
        print(
            "     a) Completed with proper proof structure (proof_id, inputs, outputs, strategy, steps)"
        )
        print("     b) Removed if they're not needed")

    print("\n" + "=" * 70)
    print("Fix Summary:")
    print(f"  ✅ Axioms fixed: {axioms_fixed}")
    print(f"  ✅ Remarks fixed: {remarks_fixed}")
    print(f"  ✅ Lemma edges fixed: {edges_fixed}")
    print(f"  ⚠️  Incomplete proofs: {len(incomplete_proofs)}")
    print("=" * 70)

    print("\n💡 Next step: Re-run validation to check remaining errors")
    print("   python -m fragile.proofs.tools.validation \\")
    print(
        "     --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \\"
    )
    print("     --mode schema --validation-mode refined")


if __name__ == "__main__":
    main()
