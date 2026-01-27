---
name: math-doc-review
description: Review mathematical documents and proofs to find errors and produce a detailed, framework-first correction report. Use when asked to audit math claims/derivations in docs (Markdown, LaTeX, notebooks, PDFs converted to text) and to classify issues by severity and error type, with the review saved in a sibling review/ folder next to the source file.
---

# Math Doc Review

## Overview
Provide a rigorous, framework-first review of mathematical documents, identify errors, and write a super-detailed correction report that classifies each issue by severity and error type. Treat the document's own theory and first-principles framework as the ground truth; do not import external mathematical results unless explicitly requested.

## Workflow

### 1) Intake and scope
- Identify the target file(s) and scope (entire document, specific sections, or specific claims).
- If the framework is unclear, ask for the primary theory sources to use (definitions, axioms, permits, prior theorems). Do not proceed with external assumptions.

### 2) Anchor the framework
- Extract and list the framework anchors: definitions, axioms/permits, notational conventions, and core results the document depends on.
- Record these anchors in the review report so the reader can see the basis for every criticism.

### 3) Review pass and issue logging
- Read the document linearly and log potential issues with exact locations (section title, equation number, line or paragraph reference).
- Check each claim against the framework anchors:
  - Are hypotheses stated and satisfied?
  - Are parameter ranges, dimensions, and domains respected?
  - Are definitions applied consistently?
  - Is any step relying on an external theorem or classical analysis argument not derived in the framework?

### 4) Classify and analyze
- For each issue, assign a severity and error type using `references/review-taxonomy.md`.
- Provide a framework-first explanation of why the issue is an error or a scope restriction.
- Propose a concrete fix: revise statements, add missing hypotheses, or add a new lemma/permit within the framework.

### 5) Produce the review artifact
- Create a `review/` folder in the same directory as the reviewed file (if it does not exist).
- Create the review document using `assets/review-template.md` and fill it in.
- File naming: `review/<basename>.review.md` where `<basename>` is the reviewed file name without its extension.

## Output requirements
- Use the template and fill every section relevant to the findings.
- Include a severity summary and an error log table.
- For each error, provide step-by-step fix guidance grounded in the framework.
- If a claim is valid only under restricted parameters (for example, 2D only), mark it as a scope restriction and specify the exact constraint.
- Avoid re-proving via classical analysis; require internal permits/lemmas instead.

## Resources
- `references/review-taxonomy.md` for severity levels and error-type definitions.
- `assets/review-template.md` for the report structure.
