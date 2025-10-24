---
description: Prepare mathematical document for publication by removing all AI review meta-commentary
---

You are preparing a mathematical document for publication by removing all meta-commentary related to AI reviews, corrections, previous versions, or revision history.

## Task Overview

Clean a mathematical document to make it publication-ready for peer review, ensuring that reviewers see only the final mathematical content without any trace of:
- AI review processes (Gemini, Codex, Claude)
- Correction rounds or applied fixes
- Previous versions or revision history
- Internal development comments

The goal is to produce a document that appears as if it were written in a single, polished effort.

## Input Handling

**Parse command arguments:**

1. **If file path is provided** (e.g., `/clean_ai docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md`):
   - Use the specified file path

2. **If no parameter is provided** (e.g., `/clean_ai`):
   - Auto-detect the most recently modified markdown file in `docs/source/`
   - **EXCLUDE** files matching patterns: `*CORRECTIONS*`, `*REVIEW*`, `*REVISION*`, `*agent_output*`
   - Use `ls -t` to find the most recent file
   - Inform the user which file was selected

## Required Steps

### Step 1: Validate and Read Target File

1. Verify the file exists
2. Read the entire document
3. Inform the user of the file being processed and its size

### Step 2: Identify Meta-Commentary Patterns

Scan for these patterns that indicate meta-commentary to be removed:

**Admonition blocks** containing:
- References to "Gemini", "Codex", "Claude", "AI", "review", "reviewer"
- Phrases: "correction applied", "fixed", "previous version", "after review", "dual review"
- Discussion of changes, revisions, or improvements made

**Inline text** containing:
- "REVIEW NOTE:", "CORRECTION:", "FIXED:", "TODO (from review):"
- "As suggested by", "After feedback", "Following review"
- References to correction rounds or version numbers

**Specific patterns to flag:**
```
:::{note} **[Title containing: Review/Correction/Fixed/Meta/Previous]**
:::{important} **Clarification on [something that references reviews/corrections]**
[Any text discussing "this was changed because" or "previously we said"]
```

**Important**: DO NOT remove admonitions that provide:
- Mathematical intuition or physical interpretation
- Important caveats about theorems or assumptions
- Pedagogical explanations
- Domain-specific context (e.g., "Understanding the Derivative Structure", "Practical Consequence")

### Step 3: Use Gemini for Intelligent Cleaning

**CRITICAL**: Submit the document to Gemini 2.5 Pro for intelligent removal of meta-commentary.

Use `mcp__gemini-cli__ask-gemini` with `model: "gemini-2.5-pro"` and the following prompt:

```
You are preparing a mathematical document for publication in a peer-reviewed journal. Remove ALL meta-commentary related to AI reviews, corrections, previous versions, or development history.

REMOVE the following:
1. Admonitions (:::note, :::important, :::warning) that discuss:
   - Reviews by Gemini, Codex, Claude, or any AI system
   - Corrections applied or fixes made
   - Previous versions or what was changed
   - Feedback from reviews or dual review protocols

2. Inline comments containing:
   - "REVIEW NOTE", "CORRECTION", "FIXED", "TODO (from review)"
   - "As suggested by", "After feedback", "Following review"
   - Discussion of why something was changed or improved

3. Any text that reveals this went through multiple revision rounds

PRESERVE the following:
1. ALL mathematical content: theorems, proofs, definitions, lemmas, propositions
2. Admonitions providing mathematical context, intuition, caveats, or important notes
3. All cross-references ({prf:ref}), citations, equation numbers
4. Pedagogical explanations and physical interpretations
5. All LaTeX equations and mathematical notation
6. Document structure (sections, subsections)

IMPORTANT PRESERVATION RULES:
- If an admonition discusses both meta-content AND mathematical content, extract ONLY the mathematical content and preserve it outside the admonition
- Do not remove admonitions with titles like "Understanding X", "Practical Consequence", "Key Observation", "Main Results" - these are mathematical context
- Keep admonitions that explain "Why" something is mathematically true (not "Why" we changed it)

OUTPUT:
Return the COMPLETE cleaned document with:
- All meta-commentary removed
- All mathematical content preserved
- Smooth transitions (no abrupt gaps where content was removed)
- Proper LaTeX formatting maintained

Here is the document to clean:

[FULL DOCUMENT CONTENT]
```

**Parameters:**
- `model: "gemini-2.5-pro"` (ALWAYS use Pro, never Flash)
- `prompt: [the above template with full document inserted]`

### Step 4: Parse and Validate Gemini's Output

1. Extract the cleaned document from Gemini's response
2. Verify the output is valid markdown
3. Check that mathematical content is preserved:
   - Count theorems, lemmas, definitions before and after
   - Verify no {prf:theorem}, {prf:lemma}, {prf:definition} blocks were removed
   - Check equation blocks ($$...$$) are intact

### Step 5: Create Backup

Before overwriting, create a timestamped backup:

```bash
cp [original_file] [original_file].pre_clean_$(date +%Y%m%d_%H%M%S).bak
```

Inform the user of the backup location.

### Step 6: Apply Cleaning

1. Write the cleaned content to the original file using the Write tool
2. Verify the file was written successfully

### Step 7: Format LaTeX Math

Run the math formatting tool to ensure proper LaTeX spacing:

```bash
python src/tools/fix_math_formatting.py [file_path] --in-place
```

### Step 8: Generate Summary Report

Create a summary showing what was removed:

**Summary Format:**

```markdown
## Clean AI Summary: [filename]

**File processed**: [full path]
**Original size**: [size] lines
**Cleaned size**: [size] lines
**Lines removed**: [count]

### Changes Applied

**Backup created**: [backup path]

**Meta-commentary removed**:
- [X] Admonitions referencing AI reviews or corrections
- [X] Inline comments about fixes or previous versions
- [X] References to review feedback
- [X] Development history markers

**Mathematical content preserved**:
- ✓ All theorems, lemmas, definitions intact
- ✓ All proofs preserved
- ✓ Cross-references maintained
- ✓ LaTeX formatting corrected

### Removed Content Summary

**Admonitions removed**: [count]
Examples:
- Line [X]: Removed note about "[brief description]"
- Line [Y]: Removed important block discussing "[brief description]"

**Inline comments removed**: [count]

**Document is now publication-ready** ✅
```

Present this summary to the user.

## Important Safeguards

### Validation Checks

Before finalizing, verify:

1. **No mathematical content lost**:
   - Number of {prf:theorem}, {prf:lemma}, {prf:definition} blocks unchanged
   - All equation blocks ($$) preserved
   - Cross-references ({prf:ref}) still valid

2. **No broken formatting**:
   - No orphaned admonition markers (:::)
   - All code blocks properly closed
   - LaTeX blocks have proper spacing

3. **Smooth readability**:
   - No abrupt gaps in text flow
   - Section transitions make sense
   - No dangling references to removed content

### Error Handling

If validation fails:
1. **DO NOT overwrite** the original file
2. Inform the user of the specific validation error
3. Show the problematic section
4. Ask if the user wants to proceed anyway or abort

## Edge Cases

### Case 1: Admonition has mixed content

**Example**:
```markdown
:::note
After reviewing with Gemini, we clarified that the bound
is uniform because the velocity is bounded by V_max.
:::
```

**Action**: Extract mathematical content and preserve it as regular text:
```markdown
The bound is uniform because the velocity is bounded by V_max.
```

### Case 2: Mathematical context that mentions "previous"

**Example**:
```markdown
Unlike the previous approach in Section 2, we now use...
```

**Action**: PRESERVE - this is mathematical narrative, not meta-commentary

### Case 3: TODO markers

**Example**:
```markdown
<!-- TODO: Add reference to Lemma 4.3 -->
```

**Action**: Remove only if it references reviews/corrections. Preserve general TODOs.

## Notes

- **Always use Gemini 2.5 Pro** for intelligent context-aware cleaning
- **Create backup before modifying** original file
- **Validate mathematical content preservation** before finalizing
- **Run formatting tools** after cleaning
- The goal is a **publishable document** with no trace of development history
- Referees should see a polished, single-effort manuscript

## Example Usage

```bash
# Clean specific file
/clean_ai docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md

# Clean most recently edited file
/clean_ai
```
