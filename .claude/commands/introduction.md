---
description: Generate or improve introduction section for mathematical documents following Fragile framework standards
---

You are tasked with creating or improving the introduction section of a mathematical document in the Fragile framework. The target file path will be provided as an argument after the command.

## Task Overview

Generate a high-quality introduction following the structure established in `algorithm/03_cloning.md`. The introduction should include:

1. **Section 0: TLDR** - A concise summary of the most important results
2. **Section 1: Introduction** with subsections:
   - 1.1. Goal and Scope
   - 1.2. [Context-specific motivation section]
   - 1.3. Overview of the Proof Strategy and Document Structure

## Required Steps

### Step 1: Read and Analyze the Document

Read the entire target document to understand:
- Main theorems, lemmas, and definitions
- Proof structure and dependencies
- Key mathematical results
- Overall narrative arc
- Existing introduction (if present)

Use strategic reading with offset/limit for large files.

### Step 2: Consult Framework Context

Before drafting, consult:
- `docs/source/00_index.md` - Check related definitions and theorems
- `docs/source/00_reference.md` - Review full statements of related results
- `algorithm/03_cloning.md` (lines 0-90) - Reference the template structure

### Step 3: Draft the TLDR (Section 0)

Create a compelling TLDR section that:
- Highlights 2-4 most important results from the document
- Uses bold formatting for key concepts
- Keeps each highlight to 1-2 sentences
- Written at a level accessible to someone familiar with the framework

Format:
```markdown
## 0. TLDR

**Main Result Name**: Brief description of the result and its significance.

**Secondary Result**: Brief description.

**Key Technique**: Brief description of novel proof technique or method.
```

### Step 4: Draft Section 1.1 - Goal and Scope

Write a clear statement that:
- States the primary goal of the document
- Identifies the central mathematical object(s) of study
- Summarizes the main result(s) to be proven
- Clarifies the scope (what is included vs. deferred)
- References related framework documents

Should be 2-3 paragraphs, written with mathematical precision.

### Step 5: Draft Section 1.2 - Motivation/Context

Create a motivation section (title adapted to document content) that:
- Explains the significance within the Fragile framework
- Provides intuition for why the results matter
- Connects to physical or algorithmic interpretation
- May include admonitions for important contextual notes

Should be 2-3 paragraphs with possible admonitions.

### Step 6: Create Document Structure Diagram

Generate a Mermaid diagram showing:
- Logical flow of the proof
- Chapter/section dependencies
- Main results in each section
- How pieces build toward the main theorem

**Style requirements** (match `03_cloning.md` exactly):
- Use `graph TD` (top-down)
- Group related chapters in `subgraph` blocks
- Use descriptive node labels with `<b>` tags and `<br>` for line breaks
- Show dependencies with labeled arrows
- Apply consistent styling:

```
classDef stateStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6
classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,stroke-dasharray: 5 5,color:#f4e8d8
classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3
classDef theoremStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8
```

Apply `:::stateStyle`, `:::axiomStyle`, `:::lemmaStyle`, `:::theoremStyle` to nodes appropriately.

### Step 7: Draft Section 1.3 - Overview

Write the overview section that:
- Introduces the proof strategy at a high level
- References the Mermaid diagram
- Provides a bullet-point walkthrough of document structure
- Explains how each part contributes to the main result

Should be 1-2 paragraphs followed by structured overview.

### Step 8: Analyze Existing Document Structure

**CRITICAL**: Determine how to integrate the new sections 0 and 1:

**Case A: No existing sections 0 and 1**
- Simply insert the new TLDR and Introduction at the beginning
- Existing content starts at section 2 (no renumbering needed)

**Case B: Existing sections 0 and/or 1 contain unrelated content**
- Insert new sections 0 (TLDR) and 1 (Introduction) at the beginning
- Renumber existing section 0 → section 2
- Renumber existing section 1 → section 3
- Update ALL subsequent section numbers throughout the document
- Update ALL internal cross-references to reflect new numbering
- Search for patterns like:
  - `## N.` → `## (N+2).`
  - `### N.M.` → `### (N+2).M.`
  - `{prf:ref}` references that include section numbers
  - Text references like "in Section X" or "Chapter X"

**Case C: Existing sections 0 and 1 are already TLDR and Introduction**
- Replace them with improved versions
- No renumbering needed

### Step 9: Integration and Renumbering

Execute the integration based on the case identified:

1. **Backup approach**: Preserve old content that might be valuable in comments or admonitions
2. **Insert new sections 0 and 1** with the drafted content
3. **If renumbering is required** (Case B):
   - Use systematic find-replace for section headers
   - Update subsection numbers (e.g., `### 1.2.` → `### 3.2.`)
   - Search for and update cross-references:
     - `{prf:ref}` labels may need updates if they encode section numbers
     - Prose references like "Section 2.3" or "Chapter 4"
   - Verify Mermaid diagram references match new numbering
4. **Verify smooth transitions** between sections

### Step 10: Dual MCP Review (MANDATORY)

**CRITICAL**: After integration, submit the complete introduction (sections 0 and 1 ONLY) for dual independent review:

Submit the IDENTICAL prompt to BOTH reviewers in parallel:

**Prompt for both reviewers:**
```
Review this introduction section for a mathematical document in the Fragile framework.

Assess:
1. Clarity and accessibility of the TLDR
2. Precision and completeness of Goal and Scope
3. Accuracy of technical claims
4. Quality of the Mermaid diagram (structure, dependencies, styling)
5. Consistency with framework conventions and existing documentation
6. Pedagogical flow and readability

Check against framework documents (00_index.md, 00_reference.md) for consistency.
Provide specific feedback with severity ratings.

[PASTE COMPLETE DRAFT HERE]
```

**Execute in parallel:**
1. **Gemini**: Use `mcp__gemini-cli__ask-gemini` with `model: "gemini-2.5-pro"`
2. **Codex**: Use `mcp__codex__codex`

### Step 11: Evaluate and Compare Feedback

Critically evaluate both reviews:
- **Consensus issues** (both agree): High confidence → prioritize
- **Discrepancies** (contradictions): Verify manually against framework docs
- **Unique issues** (only one reviewer): Medium confidence → verify before accepting
- **Cross-validate**: Check specific claims against `00_index.md` and `00_reference.md`

If you disagree with feedback:
1. Document reasoning with framework references
2. Inform the user
3. Propose alternative with justification
4. Let user decide

### Step 12: Implement Revisions

Address validated feedback systematically:
- Maintain mathematical precision
- Preserve framework notation consistency
- Update cross-references as needed
- Ensure proper LaTeX formatting

### Step 13: Final Formatting Pass

Apply formatting tools from `src/tools/`:
- Ensure exactly ONE blank line before `$$` blocks
- Verify proper LaTeX notation
- Check Mermaid diagram syntax
- Validate Jupyter Book directive syntax

## Output Requirements

Present the final result to the user showing:
1. The complete TLDR and Introduction sections (0 and 1)
2. Summary of integration approach used (Case A, B, or C)
3. If renumbering was performed:
   - List of section number changes (e.g., "Section 2 → Section 4")
   - Number of cross-references updated
   - Any references that may need manual review
4. Brief explanation of the structure diagram
5. Summary of dual review feedback and how it was addressed
6. Note any areas where user input might be helpful

## Important Notes

- **Always use Gemini 2.5 Pro** (never flash) for the review
- **Always consult** `00_index.md` and `00_reference.md` before drafting
- **Match the style** of `03_cloning.md` precisely
- **Use dual MCP review** without exception
- **Maintain mathematical rigor** while being pedagogically accessible
- **Follow framework notation** conventions strictly

The goal is to create an introduction that is simultaneously rigorous, accessible, and beautifully structured, serving as an inviting gateway to the technical content that follows.
