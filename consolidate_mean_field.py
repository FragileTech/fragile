#!/usr/bin/env python3
"""
Consolidate all mean-field convergence documents into a single source of truth.
"""

import re
from pathlib import Path

# Source directory
SOURCE_DIR = Path("docs/source/11_mean_field_convergence")
OUTPUT_FILE = Path("algorithm/11_convergence_mean_field.md")

# Files to consolidate (in order)
STAGE_FILES = [
    "11_stage0_revival_kl.md",
    "11_stage05_qsd_regularity.md",
    "11_stage1_entropy_production.md",
    "11_stage2_explicit_constants.md",
    "11_stage3_parameter_analysis.md",
]

def read_file(filepath: Path) -> str:
    """Read file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_mathematical_content(content: str, filename: str) -> str:
    """Extract mathematical content, removing redundant headers."""
    # Remove document status and parent document references
    content = re.sub(r'\*\*Document Status\*\*:.*?\n', '', content)
    content = re.sub(r'\*\*Purpose\*\*:.*?\n\n', '', content)
    content = re.sub(r'\*\*Parent documents\*\*:.*?---', '---', content, flags=re.DOTALL)
    content = re.sub(r'\*\*Relationship to.*?\*\*:.*?\n', '', content)

    # Update cross-references (remove directory paths since everything is in one file)
    content = re.sub(r'\[([^\]]+)\]\([\.\/]*11_stage\d+[^)]*\.md\)', r'(see below)', content)
    content = re.sub(r'\[([^\]]+)\]\([\.\/]*MATHEMATICAL_REFERENCE\.md\)', r'\1', content)

    return content

def create_consolidated_document():
    """Create the consolidated document."""

    # Read the original roadmap document (keep it as the header)
    original_path = Path("algorithm") / "11_convergence_mean_field.md"
    original = read_file(original_path)

    # Start building consolidated content
    consolidated = []

    # Add header from original (up to the roadmap section)
    header_match = re.search(r'(# KL-Divergence Convergence.*?)(^## 3\. Detailed Three-Stage Roadmap)', original, re.DOTALL | re.MULTILINE)
    if header_match:
        consolidated.append(header_match.group(1))

    # Add status note
    consolidated.append("""
**Document Status**: CONSOLIDATED SINGLE SOURCE OF TRUTH

This document consolidates ALL mathematical results from the mean-field convergence analysis (Stages 0, 0.5, 1, 2, 3) into a single comprehensive reference.

**What this document contains**:
- Stage 0: Revival operator KL-properties (VERIFIED - revival is KL-expansive)
- Stage 0.5: QSD regularity properties (R1-R6, all PROVEN)
- Stage 1: Full generator entropy production framework (COMPLETE)
- Stage 2: Explicit hypocoercivity constants (COMPLETE with formulas)
- Stage 3: Parameter analysis and simulation guide (COMPLETE)
- All theorems, lemmas, definitions, and proofs
- Numerical validation procedures
- Implementation guidelines

---

""")

    # Process each stage file
    for stage_file in STAGE_FILES:
        filepath = SOURCE_DIR / stage_file
        print(f"Processing {stage_file}...")

        content = read_file(filepath)

        # Extract stage name for header
        stage_name = stage_file.replace("11_", "").replace(".md", "").replace("_", " ").title()

        consolidated.append(f"# {stage_name}\n\n")
        consolidated.append(extract_mathematical_content(content, stage_file))
        consolidated.append("\n\n---\n\n")

    # Write consolidated document
    final_content = "".join(consolidated)

    # Clean up multiple blank lines
    final_content = re.sub(r'\n\n\n+', '\n\n', final_content)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"\nConsolidated document written to: {OUTPUT_FILE}")
    print(f"Total length: {len(final_content)} characters")
    print(f"Approximate lines: {final_content.count(chr(10))}")

if __name__ == "__main__":
    create_consolidated_document()
