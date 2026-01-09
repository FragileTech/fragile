# Claude Code Guidelines for Fragile Docs

## Feynman Agent Rules

When acting as the "Feynman agent" to add explanatory content to this documentation, you MUST follow these strict rules.

### Content Classification

| Content Type | Class Required | Shown in Expert Mode? |
|--------------|----------------|----------------------|
| Formal definitions (`{prf:definition}`) | None | **YES** - always shown |
| Theorems (`{prf:theorem}`) | None | **YES** - always shown |
| Lemmas (`{prf:lemma}`) | None | **YES** - always shown |
| Corollaries (`{prf:corollary}`) | None | **YES** - always shown |
| Axioms (`{prf:axiom}`) | None | **YES** - always shown |
| Proofs (`{prf:proof}`) | None | **YES** - always shown |
| Feynman prose | `feynman-prose` | NO - hidden |
| Agent-added notes/examples | `feynman-added` | NO - hidden |
| Agent-added admonitions | `feynman-added` | NO - hidden |
| Agent-added tables (non-formal) | `feynman-added` | NO - hidden |

### What You CAN Do

1. **Write inside existing Feynman prose blocks:**
   ```markdown
   :::{div} feynman-prose
   [You can ADD content here]
   :::
   ```

2. **Create NEW Feynman prose blocks:**
   ```markdown
   :::{div} feynman-prose
   Your explanatory prose here. Write in Feynman's conversational style.
   :::
   ```

3. **Add NEW non-formal directives** (notes, examples, tips, etc.) - MUST use `feynman-added` class:
   ```markdown
   :::{note}
   :class: feynman-added
   This is an agent-added clarification that will be hidden in expert mode.
   :::
   ```

   ```markdown
   :::{admonition} Example: Computing the Metric
   :class: feynman-added tip
   Here's a worked example...
   :::
   ```

4. **Add NEW tables for intuition/examples** - MUST wrap in feynman-added div:
   ```markdown
   :::{div} feynman-added
   | Concept | Intuition | Formal Definition |
   |---------|-----------|-------------------|
   | ... | ... | ... |
   :::
   ```

5. **Add formal definitions that were MISSING** (no special class needed):
   ```markdown
   :::{prf:definition} Missing Concept Name
   :label: def-missing-concept

   Formal definition here...
   :::
   ```

### What You MUST NEVER Do

1. **NEVER delete existing content** - not formal definitions, not theorems, not prose, nothing
2. **NEVER modify text outside of Feynman prose blocks** - the formal mathematical content is immutable
3. **NEVER remove or alter existing Feynman prose** - only add to it or create new blocks
4. **NEVER add non-formal directives without the `feynman-added` class**
5. **NEVER change the structure of existing directives**

### Class Usage Examples

**Adding an example (hidden in expert mode):**
```markdown
:::{admonition} Example: Calculating Information Flow
:class: feynman-added example

Let's work through a concrete case. Suppose we have...
:::
```

**Adding a warning/note (hidden in expert mode):**
```markdown
:::{warning}
:class: feynman-added
This approximation breaks down when $\gamma \to 1$.
:::
```

**Adding a seealso box (hidden in expert mode):**
```markdown
:::{seealso}
:class: feynman-added
For background on Fisher information, see {ref}`sec-fisher-info`.
:::
```

**Adding a FORMAL definition (always shown - NO class needed):**
```markdown
:::{prf:definition} The Missing Concept
:label: def-the-missing-concept

Let $X$ be a measurable space. The **missing concept** is defined as...
:::
```

### Feynman Prose Style Guide

Write as Richard Feynman would explain physics to undergraduates:

- Start with intuition before formalism
- Use concrete examples and analogies
- Ask rhetorical questions to guide thinking
- Build concepts step by step
- Acknowledge what's tricky or counterintuitive
- Connect abstract math to physical/operational meaning

### Toggle Behavior

The "Expert Mode" toggle in the header controls visibility:

- **Full Mode**: All content visible (prose + agent-added content + formal)
- **Expert Mode**: Only formal mathematical content (definitions, theorems, proofs)

Content with class `feynman-prose` or `feynman-added` is hidden in Expert Mode.

### Verification Checklist

Before completing any edit:
- [ ] All new prose is inside `:::{div} feynman-prose` blocks
- [ ] All non-formal directives have `:class: feynman-added`
- [ ] Formal definitions/theorems have NO special class
- [ ] No existing content was deleted
- [ ] No formal definitions/theorems were modified
- [ ] Prose complements but doesn't duplicate formal content

### Quick Reference

```markdown
# Prose block (hidden in expert mode)
:::{div} feynman-prose
Conversational explanation...
:::

# Agent-added note (hidden in expert mode)
:::{note}
:class: feynman-added
Clarification...
:::

# Agent-added example (hidden in expert mode)
:::{admonition} Example Title
:class: feynman-added example
Worked example...
:::

# Agent-added table (hidden in expert mode)
:::{div} feynman-added
| Col1 | Col2 |
|------|------|
| ... | ... |
:::

# Formal definition (ALWAYS shown - no class)
:::{prf:definition} Name
:label: def-name
Formal content...
:::
```
