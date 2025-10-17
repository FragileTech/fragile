---
description: Dual review protocol - Submit a document for independent review by both Gemini and Codex
---

You are conducting a **dual independent review** of a mathematical document using the protocol defined in CLAUDE.md.

**CRITICAL WORKFLOW:**

1. **Parse the file path** from the command arguments (e.g., `/review algorithm/03_cloning.md`)
2. **Read the entire document** using the Read tool
3. **Consult the mathematical index** (`docs/source/00_index.md`) to understand context and related definitions
4. **Submit for DUAL review** in parallel using IDENTICAL prompts:
   - **Gemini Review**: Use `mcp__gemini-cli__ask-gemini` with `model: "gemini-2.5-pro"` (NEVER flash)
   - **Codex Review**: Use `mcp__codex__codex` for independent second opinion
5. **CRITICAL**: Both reviewers MUST receive the EXACT SAME prompt to ensure independent, comparable feedback

**Review Prompt Template** (use verbatim for both reviewers):
```
Review this mathematical document for publication-quality rigor, consistency with the Fragile framework, and clarity.

Document: [filename]

[full document content]

Provide:
1. Critical analysis with severity ratings (CRITICAL/MAJOR/MINOR/SUGGESTION)
2. Specific issues with:
   - Location (section/line reference)
   - Problem description
   - Impact on mathematical validity
   - Suggested fix
3. Checklist of required proofs or missing justifications
4. Prioritized action plan
5. Overall assessment

Focus on:
- Mathematical rigor and completeness of proofs
- Consistency with framework axioms and definitions
- Notation consistency
- Logical gaps or unjustified claims
- Clarity of mathematical exposition
```

6. **Critical Evaluation Phase**:
   - Compare both reviews to identify:
     - **Consensus issues** (both agree): High confidence → prioritize
     - **Discrepancies** (contradict each other): Potential hallucination → verify manually
     - **Unique issues** (only one identified): Medium confidence → verify before accepting
   - **Cross-validate** ALL specific claims against `docs/source/00_index.md` and `docs/source/00_reference.md`
   - **Flag hallucinations**: Reject any claim that cannot be verified in framework documents
   - **Document your assessment** of each issue with framework references

7. **Present Results** in this format:

## Dual Review Summary for [filename]

### Comparison Overview
- **Consensus Issues**: [count] (both reviewers agree)
- **Gemini-Only Issues**: [count]
- **Codex-Only Issues**: [count]
- **Contradictions**: [count] (reviewers disagree)

### Issue Analysis Table

| # | Issue | Severity | Gemini | Codex | Claude (You) | Verification Status |
|---|-------|----------|--------|-------|--------------|---------------------|
| 1 | [brief description] | CRITICAL | [Gemini's position with quote] | [Codex's position with quote] | [Your assessment with framework refs] | ✓ Verified / ⚠ Unverified / ✗ Contradicts framework |
| 2 | ... | MAJOR | ... | ... | ... | ... |

### Detailed Issues and Proposed Fixes

#### Issue #1: [Title]
- **Location**: [section/line]
- **Severity**: [level]
- **Gemini's Analysis**: [full feedback]
- **Codex's Analysis**: [full feedback]
- **Your Assessment**: [your analysis with references to framework docs]
- **Verification**: [what you checked in 00_index.md/00_reference.md]
- **Consensus**: [AGREE/DISAGREE/PARTIAL]

**Proposed Fix**:
```
[specific implementation with code/LaTeX as needed]
```

**Rationale**: [why this fix addresses the issue while maintaining framework consistency]

---

[Repeat for each issue]

### Implementation Checklist

Priority order based on severity and verification status:

- [ ] **CRITICAL** issues (verified consensus)
- [ ] **CRITICAL** issues (single reviewer, verified)
- [ ] **MAJOR** issues (verified consensus)
- [ ] **MAJOR** issues (single reviewer, verified)
- [ ] **MINOR** issues and **SUGGESTIONS**

### Contradictions Requiring User Decision

[List any issues where you disagree with reviewers or reviewers contradict each other]

For each contradiction:
- Present all three perspectives (Gemini, Codex, yours)
- Explain your reasoning with framework references
- Ask user to make final decision

---

**IMPORTANT REMINDERS**:
- ALWAYS use `model: "gemini-2.5-pro"` for Gemini (never flash)
- Send IDENTICAL prompts to both reviewers
- Verify ALL claims against framework documents before accepting
- Flag any unverifiable claims as potential hallucinations
- Document your reasoning when disagreeing with feedback
- Let the user make final decisions on contradictions
