# Math Review Command

Invoke the Math Reviewer agent for autonomous dual-review analysis of mathematical documents.

## Instructions

You are now acting as the Math Reviewer agent. Follow the complete protocol defined in `.claude/agents/math-reviewer.md`.

**CRITICAL**: You MUST read the agent definition file first:

```
Read: .claude/agents/math-reviewer.md
```

Then execute the Math Reviewer protocol on the document specified by the user.

## Expected Input Format

The user will provide input in one of these formats:

### Format 1: Basic
```
/math_review docs/source/path/to/document.md
```

### Format 2: With Depth
```
/math_review docs/source/path/to/document.md
Depth: exhaustive
```

### Format 3: With Focus
```
/math_review docs/source/path/to/document.md
Focus: Section 2.3 non-circularity, Lemma 4.2 k-uniformity
```

### Format 4: Complete Specification
```
/math_review docs/source/path/to/document.md
Depth: thorough
Focus: Non-circular density bound proof, statistical equivalence claims
```

## Parameters

- **file_path** (required): Path to the mathematical document to review
- **depth** (optional): `quick` | `thorough` (default) | `exhaustive`
- **focus_areas** (optional): Specific sections/topics to emphasize

## Agent Protocol

After reading the agent definition, you MUST follow this autonomous workflow:

### Phase 1: Document Analysis (10-15%)
1. Validate file exists and get size
2. Map document structure (grep for section headers)
3. Identify critical sections based on depth setting
4. Extract sections strategically using Read with offset/limit

### Phase 2: Dual Review Execution (15-20%)
1. Construct comprehensive identical prompt for both reviewers
2. Submit to BOTH in parallel:
   - Gemini 2.5 Pro (model="gemini-2.5-pro")
   - Codex GPT-5 Pro (model="gpt-5-pro")
3. Wait for both responses

### Phase 3: Critical Analysis (40-50%)
1. Classify issues: Consensus / Contradiction / Unique
2. Verify claims against framework (docs/glossary.md)
3. Make evidence-based judgments
4. Re-assess severity based on verification

### Phase 4: Report Generation (20-25%)
1. Generate comprehensive report following template
2. Include all sections: comparison, tables, detailed analysis, verdict
3. Write to file: `reviewer/review_{timestamp}_{filename}.md`
4. Inform user of output location

## Output

The agent will:
1. Execute complete review autonomously
2. Write comprehensive report to file
3. Display summary to user with file path

**File Location**: `{document_dir}/reviewer/review_{YYYYMMDD_HHMM}_{doc_name}.md`

## Quality Guarantees

- ✅ Identical prompts to both reviewers
- ✅ Evidence-based judgments with framework verification
- ✅ Honest uncertainty acknowledgment
- ✅ Actionable fixes for all CRITICAL/MAJOR issues
- ✅ Template compliance for parseable output
- ✅ Framework consistency checks

## Notes

- Agent runs autonomously (no interruptions)
- Expected runtime: 10 min (quick) / 45 min (thorough) / 2 hours (exhaustive)
- Output is ~2000-10000 words depending on depth
- Multiple instances can run in parallel on different documents

---

**Now begin the Math Reviewer protocol for the document provided by the user.**
