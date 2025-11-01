# Proof Validation - Troubleshooting

## Sketching Issues

### Issue: Proof sketch too vague

**Symptoms**: Sketch has high-level steps but no detail

**Example**:
```markdown
1. [SKETCHED] Prove convergence
2. [SKETCHED] Use spectral gap
```

**Cause**: Insufficient context provided to sketcher

**Solution**: Provide more detail
```
Load proof-sketcher agent.

Sketch: thm-keystone-principle
Document: docs/source/1_euclidean_gas/03_cloning.md
Context:
- Uses Langevin dynamics with friction γ
- Companion selection via cloning measure
- Requires irreducibility + detailed balance
Required lemmas: lem-markov-kernel, lem-spectral-gap
```

---

### Issue: Sketcher suggests wrong approach

**Symptoms**: Proof strategy doesn't match theorem structure

**Cause**: Misunderstood theorem statement or missing dependencies

**Solution**:
1. Check theorem statement is clear in markdown
2. Provide explicit strategy hint:
```
Use strategy: Lyapunov function approach (NOT spectral gap)
```

---

## Expansion Issues

### Issue: Expanded proof has gaps

**Symptoms**: Steps marked EXPANDED but lack mathematical detail

**Cause**: LLM couldn't fill in details (too complex or missing context)

**Solution**:
1. Check if proof sketch had sufficient detail
2. Provide additional context:
```
Load theorem-prover agent.

Expand: thm-keystone-principle
Additional context:
- Cloning measure defined in Definition 5.2
- Use Hölder inequality with p=2
- Reference Lemma 3.4 for boundedness
```

---

### Issue: Expansion takes too long

**Symptoms**: Theorem-prover runs for >2 hours

**Cause**: Too many steps or very complex derivations

**Solution**:
- **Break into parts**: Expand proof in sections
- **Simplify sketch**: Reduce number of steps
- **Monitor progress**: Agent should show progress updates

---

## Review Issues

### Issue: Review finds no issues

**Symptoms**: Report says "No critical issues found" but proof seems incomplete

**Cause**: Review depth insufficient or wrong focus

**Solution**: Increase depth and specify focus
```
Load math-reviewer agent.

Review: docs/source/.../document.md
Depth: exhaustive                    # ← More thorough
Focus on:
- Proof of Theorem 8.1 (lines 450-600)
- Lemma 5.3 bound derivation (lines 280-310)
- Definition 3.2 consistency with framework
```

---

### Issue: Contradictory reviews

**Symptoms**: Gemini and Codex provide conflicting assessments

**Example**:
- Gemini: "Proof is correct"
- Codex: "Fatal flaw in step 3"

**Cause**: This is actually good! Indicates need for investigation

**Solution**:
1. **Read both analyses** carefully
2. **Check framework docs**: Verify claims against `docs/glossary.md`
3. **Claude's judgment**: Review report includes evidence-based analysis
4. **Manual verification**: Double-check the specific claim

**This is a feature, not a bug** - contradictions help identify subtle issues.

---

### Issue: Review report too long

**Symptoms**: Report >10,000 words, hard to navigate

**Cause**: Exhaustive depth on large document

**Solution**: Use focused reviews
```
# Instead of reviewing entire document:
Load math-reviewer agent.
Review: docs/source/.../document.md
Depth: exhaustive

# Focus on specific sections:
Load math-reviewer agent.
Review: docs/source/.../document.md
Depth: thorough                       # ← Less exhaustive
Focus: Section 8 only (Keystone Principle proof)
```

---

### Issue: Review finds issues but no fix proposed

**Symptoms**: Issue identified but "Proposed Fix: [unclear]"

**Cause**: Issue too complex for automatic fix suggestion

**Solution**:
1. Understand the issue from both AI perspectives
2. Consult framework documents for guidance
3. Ask Claude directly:
```
Based on Issue #3 in the review report, how should I fix the non-circular density argument?
Please consult docs/glossary.md and provide a detailed fix.
```

---

## Verification Issues

### Issue: Verification script fails

**Symptoms**:
```python
AssertionError: Bound should approach 1
```

**Cause**: Mathematical claim is actually incorrect

**Solution**:
1. **Review the claim**: Is it stated correctly?
2. **Check derivation**: Go back to proof expansion
3. **Fix at source**: Update markdown document
4. **Re-verify**: Run verifier again

**This is a feature** - verifier catches incorrect claims.

---

### Issue: SymPy can't symbolically verify

**Symptoms**: Verification script times out or gives "cannot determine"

**Cause**: Claim too complex for symbolic computation

**Solution**: Use numerical verification
```
Load math-verifier agent.

Verify numerically: Eigenvalue gap bound in Lemma 5.3
Use parameter ranges: γ ∈ [0.1, 10], t ∈ [0, 100]
```

---

## Workflow Issues

### Issue: Wrong agent order

**Symptoms**: Trying to expand proof before sketching

**Cause**: Skipped sketching stage

**Solution**: Always follow order:
1. Proof Sketcher (generate strategy)
2. Theorem Prover (expand details)
3. Math Reviewer (validate rigor)
4. Math Verifier (check computations)

**Don't skip sketching** - it's the foundation.

---

### Issue: Agent can't find theorem

**Symptoms**: "Theorem thm-X not found in document"

**Cause**: Label mismatch or theorem not extracted

**Solution**:
```bash
# Check theorem exists
grep "thm-keystone-principle" docs/source/.../document.md

# Check label format (lowercase kebab-case)
# ❌ Wrong: thm-KeystonePrinciple, THM-keystone-principle
# ✅ Correct: thm-keystone-principle
```

---

### Issue: Reports in wrong location

**Symptoms**: Can't find output files

**Cause**: All reports now go to centralized `reports/` directory

**Solution**: Check expected locations
```bash
# Sketcher output
ls docs/source/.../reports/sketcher/

# Prover output
ls docs/source/.../reports/mathster/

# Reviewer output
ls docs/source/.../reports/reviewer/

# Verifier output
ls docs/source/.../reports/verifier/
```

---

## Quality Issues

### Issue: Proof sketch seems AI-generated

**Symptoms**: Generic steps, lacks specificity to theorem

**Cause**: Insufficient theorem context

**Solution**: Provide detailed theorem statement and context
- Include all hypotheses
- State conclusion explicitly
- Reference relevant framework definitions
- Specify required lemmas

---

### Issue: Review doesn't catch obvious error

**Symptoms**: You spot an error that dual-review missed

**Cause**: Error in non-mathematical content or LLM blind spot

**Solution**:
1. **Report to Claude**: "The review missed this error: [description]"
2. **Manual verification**: Always critically evaluate AI output
3. **Targeted verification**: Use math-verifier on specific claim:
```
Load math-verifier agent.

Verify specifically: [paste the problematic claim]
```

---

## Performance Issues

### Issue: Dual-review takes too long

**Symptoms**: Review runs for >2 hours

**Cause**: Large document + exhaustive depth

**Solution**:
- **Use thorough depth** (not exhaustive) for standard reviews
- **Focus on sections**: Don't review entire document
- **Process in parts**: Review sections separately

---

### Issue: LLM rate limiting

**Symptoms**: Agent pauses or reports rate limit errors

**Cause**: Too many API calls in short time

**Solution**:
- **Wait a few minutes** before retrying
- **Reduce depth**: Use quick/thorough instead of exhaustive
- **Batch processing**: Space out multiple agent runs

---

## Getting Help

If issues persist:

1. **Check agent definitions**: Detailed protocols in `.claude/agents/`
2. **Consult CLAUDE.md**: Mathematical notation and rigor requirements
3. **Check GEMINI.md**: Gemini review protocol specifics
4. **Review framework**: `docs/glossary.md` for mathematical reference
5. **Ask Claude**: Describe specific issue with context

---

**Related**:
- [QUICKSTART.md](./QUICKSTART.md) - Quick reference
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- [SKILL.md](./SKILL.md) - Complete documentation
