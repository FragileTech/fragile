# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fragile** is a FractalAI implementation exploring advanced optimization algorithms through physics-inspired stochastic search methods. The project implements multiple variants of "Gas" algorithms (Euclidean Gas, Adaptive Gas) that use particle swarms with physical dynamics (Langevin dynamics, cloning, viscous coupling) to search complex state spaces.

The codebase is a research-oriented implementation with strong mathematical foundations documented in markdown files in the `markdown/` directory. The algorithms are built on PyTorch for vectorization and Pydantic for parameter validation.

## Key Architecture

### Core Algorithm Hierarchy

1. **EuclideanGas** (`src/fragile/euclidean_gas.py`) - Base implementation
   - Implements Langevin dynamics (kinetic operator) with BAOAB integrator
   - Cloning operator with inelastic collisions and momentum conservation
   - Virtual reward mechanism for guiding exploration
   - All operations vectorized: first dimension is always N (number of walkers)

2. **AdaptiveGas** (`src/fragile/adaptive_gas.py`) - Extended version
   - Adds three adaptive mechanisms on top of EuclideanGas:
     - Adaptive force from mean-field fitness potential
     - Viscous coupling between walkers (fluid-like behavior)
     - Regularized Hessian diffusion (anisotropic noise)
   - Follows "stable backbone + adaptive perturbation" philosophy

3. **SwarmState** data structure:
   - Vectorized state representation: `x` (positions [N, d]), `v` (velocities [N, d])
   - Per-walker metadata: `reward`, `potential`, `virtual_reward`, `cum_reward`
   - Uses PyTorch tensors exclusively

### Parameter Management

All parameters use Pydantic models with validation:
- `EuclideanGasParams`: Contains `N`, `d`, `potential`, `langevin`, `cloning`
- `AdaptiveGasParams`: Extends with `adaptive` mechanism parameters
- Mathematical notation documented in docstrings matches markdown documentation

### Mathematical Documentation

The `docs/source/` directory contains rigorous mathematical specifications organized in two ways:

**1. Mathematical Glossary:**
- **`docs/glossary.md`** - Comprehensive glossary of all 683 mathematical entries (quick navigation)
  - **Use this first** when you need to find definitions, theorems, or understand the framework
  - Provides: entry type, label, tags, and source document references
  - Organized by document with cross-references for searchability
  - Fast lookup for navigating the framework structure
  - Covers both Euclidean Gas (Chapter 1, 523 entries) and Geometric Gas (Chapter 2, 160 entries)

**2. Detailed Framework Documents:**
- `01_fragile_gas_framework.md` - Core axioms and foundational definitions
- `02_euclidean_gas.md` - Euclidean Gas specification
- `03_cloning.md` - Cloning operator and Keystone Principle
- `04_convergence.md` - Kinetic operator and QSD convergence
- `05_mean_field.md` - Mean-field limit and McKean-Vlasov PDE
- `06_propagation_chaos.md` - Propagation of chaos
- `07_adaptative_gas.md` - Adaptive Viscous Fluid Model
- `08_emergent_geometry.md` - Emergent Riemannian geometry
- `09_symmetries_adaptive_gas.md` - Symmetry structure
- `10_kl_convergence/` - KL-divergence convergence and LSI theory
- `11_mean_field_convergence/` - Mean-field entropy production
- `12_gauge_theory_adaptive_gas.md` - Gauge theory formulation
- `13_fractal_set/` - Discrete spacetime and lattice QFT

**Workflow:**
1. **For quick lookup**: Use `docs/glossary.md` to find definitions, theorems, constants by tags/labels
2. **For detailed statements**: Read the source documents directly (they contain full proofs)
3. **For deep understanding**: Read the full framework documents from beginning to end
4. **For implementation**: Code mirrors mathematical notation from these documents

### Visualization and Analysis

- `src/fragile/shaolin/` - Visualization tools using HoloViz stack
  - `gas_viz.py` - Interactive visualizations of Gas algorithm dynamics
  - `stream_plots.py` - Real-time streaming visualizations
  - Parameter configuration through Shaolin's declarative UI system
  - **Always use this module for visualizations**

- `src/fragile/dataviz.py` - Basic visualization utilities (legacy, prefer shaolin)

**Visualization Stack:**
- **Primary 2D**: HoloViews + hvPlot with Bokeh backend
- **Primary 3D**: HoloViews with Plotly backend
- **Interactive dashboards**: Panel
- **DO NOT use matplotlib** - use HoloViz stack exclusively
- Always prefer `fragile.shaolin` module for all visualization needs

### Tools and Utilities

- `src/tools/` - Markdown processing tools for mathematical documentation
  - `convert_unicode_math.py` - Convert Unicode math to LaTeX
  - `convert_backticks_to_math.py` - Convert backticks to dollar signs
  - `fix_math_formatting.py` - Fix LaTeX block spacing
  - `format_math_blocks.py` - Comprehensive formatting fixes
  - `fix_complex_subscripts.py` - Handle complex subscript notation
  - `convert_mermaid_blocks.py` - Convert mermaid blocks to MyST directive format
    - Automatically runs during `make build-docs`
    - Allows editing with GitHub-flavored markdown (````mermaid`) in VSCode
    - Converts to Jupyter Book MyST directive format (` ```{mermaid} `) at build time
    - Works with both ````mermaid` and `:::mermaid` source formats
  - **`find_source_location.py`** - Interactive CLI for finding source locations
    - Find text snippets, directives, equations, or sections in markdown
    - Output SourceLocation JSON for enrichment
    - Batch processing from CSV
    - See `.claude/skills/source-location-enrichment/` for full documentation

- `src/fragile/proofs/tools/` - Mathematical entity processing tools
  - **`source_location_enricher.py`** - Enrich raw JSON with precise source locations
    - Automatically finds line ranges for extracted entities
    - Uses text matching with fallback strategies
    - Batch processing for entire corpus
    - CLI: `python src/fragile/proofs/tools/source_location_enricher.py {single|directory|batch}`
  - **`line_finder.py`** - Low-level text finding utilities
    - `find_text_in_markdown()` - Find text snippets
    - `find_directive_lines()` - Find Jupyter Book directives
    - `find_equation_lines()` - Find LaTeX equations
    - `find_section_lines()` - Find sections by heading
  - `directive_parser.py` - Extract Jupyter Book directives

- `src/fragile/proofs/utils/source_helpers.py` - SourceLocation builders
  - **`SourceLocationBuilder`** - Factory methods for creating SourceLocation objects
    - `from_markdown_location()` - Most precise (line range)
    - `from_jupyter_directive()` - Precise (directive label)
    - `from_raw_entity()` - Auto-fallback from entity data
    - `with_fallback()` - Flexible multi-level fallback
    - `from_section()` - Section-level precision
    - `minimal()` - Document-level only

### Source Location Enrichment Workflow

**MANDATORY BEFORE TRANSFORMATION**: Source locations are REQUIRED for all enriched and core math types.

**Workflow:**
1. **Extract** entities → `raw_data/` (staging types with optional source)
2. **Enrich** sources → Run enricher to populate source fields (**MANDATORY**)
3. **Transform** → `.from_raw()` requires source, errors if missing
4. **Validate** → Verify all enriched types have sources

**When enriching raw data:**

**Step 1: After extraction** - Run enricher on raw_data/ (**REQUIRED**):
```bash
# Single document
python src/fragile/proofs/tools/source_location_enricher.py directory \
    docs/source/.../raw_data \
    docs/source/.../document.md \
    document_id

# Entire corpus
python src/fragile/proofs/tools/source_location_enricher.py batch docs/source/
```

**Step 2: Manual lookup** (if needed) - Use interactive tool:
```bash
# Find text
python src/tools/find_source_location.py find-text \
    docs/source/.../document.md "text to find" -d document_id

# Find directive
python src/tools/find_source_location.py find-directive \
    docs/source/.../document.md thm-label -d document_id
```

**Step 3: Validation** - Verify enriched data:
```python
from fragile.proofs.tools.line_finder import validate_line_range, extract_lines
# Check line_range points to correct content
```

**Error handling:** If `.from_raw()` raises "requires source location":
1. Check raw JSON has `source` field populated
2. If missing, run source_location_enricher
3. If present but invalid, use find_source_location.py to fix

See `.claude/skills/source-location-enrichment/skill.md` for complete workflow documentation.

## Development Commands

All development commands are available through the Makefile. Run `make help` to see all available commands.

### Testing
```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_euclidean_gas.py

# Run with coverage
make cov

# Run without coverage (faster)
make no-cov

# Debug mode with IPython debugger
make debug

# Run doctests
make doctest
```

### Linting and Formatting
```bash
# Format and fix code
make style

# Check without fixing
make check

# Run type checking with mypy
make typing

# Run all lint checks (style + check + typing)
make lint
```

### Documentation
```bash
# Build Jupyter Book documentation (auto-converts ```mermaid to :::mermaid)
make build-docs

# Build and serve documentation
make docs

# Serve documentation (after building)
make serve-docs

# Build with Sphinx directly
make sphinx

# Manually convert mermaid blocks (usually not needed)
python src/tools/convert_mermaid_blocks.py docs/source --in-place
```

### Setup and Maintenance
```bash
# Install project with dependencies
make install

# Sync dependencies
make sync

# Clean build artifacts and caches
make clean
```

### Complete Workflow
```bash
# Run everything: lint, build docs, test
make all
```

## Testing Philosophy

Tests are organized by component:
- `test_euclidean_gas.py` - Core algorithm tests
- `test_adaptive_gas.py` - Extended algorithm tests
- `test_kinetic_operator.py` - Langevin dynamics tests
- `test_cloning_operator.py` - Cloning mechanism tests
- `test_momentum_conservation.py` - Physics conservation laws
- `test_convergence.py` - Algorithm convergence properties
- `test_parameters.py` - Pydantic parameter validation

Tests verify both algorithmic correctness and physical principles (momentum conservation, energy bounds, etc.).

## Code Conventions

### Import Style
- All imports must be absolute from `fragile.*` (enforced by ruff)
- Tools in `src/tools/` should import as `from tools.*`
- Type checking imports use `if TYPE_CHECKING:` guard

### Vectorization
- First dimension is always N (number of walkers)
- Shape conventions: `[N, d]` for positions/velocities, `[N]` for scalars
- Use PyTorch operations, avoid loops over walkers

### Mathematical Notation
- Greek letters in code should match markdown docs (e.g., `gamma`, `beta`, `epsilon_F`)
- Include equation references in docstrings when implementing specific formulas
- Use Pydantic Field descriptions to document mathematical meaning

### Ruff Configuration
- Line length: 99 characters
- Python target: 3.10 (Atari/plangym compatible)
- Strict import sorting (isort)
- Format docstring code blocks (80 char line length)
- Comprehensive ignore list with inline comments for research code (see pyproject.toml)

## Project Structure

```
src/fragile/
├── euclidean_gas.py       # Base Gas implementation
├── adaptive_gas.py        # Extended adaptive version
├── gas_parameters.py      # Parameter configurations
├── bounds.py              # State space boundaries
├── utils.py               # Shared utilities
├── random_state.py        # Reproducible randomness
├── shaolin/               # Visualization components
└── __main__.py            # CLI entry point

markdown/                  # Mathematical specifications
tests/                     # Test suite
src/tools/                 # Documentation processing tools
```

## Visualization Guidelines

**IMPORTANT**: Always use the HoloViz stack for all visualizations:

### 2D Visualizations
- Use **HoloViews** or **hvPlot** with **Bokeh backend**
- Import pattern: `import holoviews as hv; hv.extension('bokeh')`
- For pandas DataFrames: use `.hvplot()` methods
- Interactive features: use Panel for dashboards and controls

### 3D Visualizations
- Use **HoloViews** with **Plotly backend**
- Import pattern: `import holoviews as hv; hv.extension('plotly')`
- For 3D scatter, surface, mesh plots

### Module Usage
- **Always prefer** `fragile.shaolin` module for visualizations
- `fragile.shaolin.gas_viz` - Gas algorithm visualizations
- `fragile.shaolin.stream_plots` - Real-time streaming plots
- `fragile.dataviz` is legacy - avoid using it for new code

### Forbidden
- **DO NOT use matplotlib** in any new code
- If you encounter matplotlib code, refactor it to HoloViz stack
- Exception: Only if maintaining existing matplotlib-dependent code

## Mathematical Proofing and Documentation

### Document Structure and Style

The `docs/source/` directory contains rigorous mathematical specifications with the following characteristics:

#### Document Organization
- Documents are numbered sequentially (e.g., `01_fragile_gas_framework.md`, `02_euclidean_gas.md`)
- Each document builds on previous foundations
- Cross-references use Jupyter Book's `{prf:ref}` directive
- Documents can be extremely large (>400KB) - use strategic reading with offset/limit

#### Mathematical Style Requirements

**LaTeX Math Formatting:**
- Use `$` for inline math and `$$` for display math (Jupyter Book/MyST markdown)
- **CRITICAL**: Always include exactly ONE blank line before opening `$$` blocks
- Never use backticks for mathematical expressions
- Use proper LaTeX notation throughout

**Example:**
```markdown
The walker state is defined as follows:

$$
w := (x, v, s)
$$

where $x \in \mathcal{X}$ is the position.
```

**Jupyter Book Directives:**
- Use `{prf:definition}`, `{prf:theorem}`, `{prf:lemma}`, `{prf:proof}`, etc.
- Always include `:label:` for cross-referencing
- Use `{prf:ref}` for internal references
- Admonitions: `{note}`, `{important}`, `{tip}`, `{warning}`, `{dropdown}`

**Example:**
```markdown
:::{prf:theorem} Main Convergence Result
:label: thm-main-convergence

Under the stated axioms, the Euclidean Gas converges exponentially fast to a unique quasi-stationary distribution.
:::

See {prf:ref}`thm-main-convergence` for the main result.
```

#### Level of Rigor

Mathematical documents in this project target **top-tier journal standards**:
- Every claim must have a complete proof or explicit reference
- All definitions must be unambiguous and mathematically precise
- Axioms and assumptions must be stated explicitly
- Proofs should include detailed step-by-step derivations
- Physical intuition should be provided in admonitions (e.g., `{note}`, `{tip}`)
- Use pedagogical explanations to bridge intuition and rigor

### Collaborative Review Workflow with Gemini

:::{important}
**Note on GEMINI.md Customization**: You are explicitly authorized to edit [GEMINI.md](GEMINI.md) to optimize the collaborative workflow with Gemini. If you find that Gemini's review format, level of detail, or communication style could be improved to better support your work, you may modify the instructions in GEMINI.md. This includes:
- Adjusting the review output format for easier parsing
- Requesting specific types of analysis that are most useful
- Modifying the severity classification system
- Changing the structure of feedback delivery
- Adding domain-specific instructions relevant to the Fragile framework

Always inform the user when you make changes to GEMINI.md and explain your reasoning.
:::

When writing or reviewing mathematical documentation, **you MUST follow this workflow**:

#### Step 0: Consult the Mathematical Index (Prerequisite)
**ALWAYS START HERE**: Before drafting or reviewing any mathematical content, consult `docs/glossary.md` to:
- Check if related definitions, theorems, or lemmas already exist (search by tags/labels)
- Understand how your work fits into the larger framework
- Identify dependencies and cross-references
- Navigate quickly to relevant source documents
- Ensure consistency with established notation and conventions
- For full mathematical statements, refer to the source documents directly

**Example workflow:**
- Writing about LSI? → Search `docs/glossary.md` for entries tagged with `kl-convergence` or `lsi`
- Working on cloning? → Search for tags like `cloning`, `measurement`, `fitness`
- Adding a theorem? → Search for related results using labels and tags, then check full statements in source documents

#### Step 1: Draft or Modify Content
- Read relevant sections of existing documents
- Draft new content or modifications following the style requirements above
- Ensure all mathematical notation is consistent with framework conventions
- Add proper labels for cross-referencing (entries will be indexed in glossary)

#### Step 2: Dual Independent Review via MCP (Gemini + Codex)
**MANDATORY**: Before finalizing any mathematical content, submit it for review using BOTH independent reviewers:

1. **Gemini Review** - Use `mcp__gemini-cli__ask-gemini` with **model: "gemini-2.5-pro"**
2. **Codex Review** - Use `mcp__codex__codex` for independent second opinion

**CRITICAL REQUIREMENTS:**
- Both reviewers must receive the **identical prompt** to ensure independent, comparable feedback
- This dual review protocol guards against hallucinations and provides diverse perspectives
- Run both reviews in parallel when possible (single message with two tool calls)
- Always verify claims by checking against framework documents before accepting feedback

**NOTE**: Gemini will automatically consult `docs/glossary.md` as part of its review protocol (see GEMINI.md § 4).

**Prompt Templates** (use identical prompt for both reviewers):
- **Rigor check**: "Review this proof for mathematical rigor and completeness. Check all claims, verify logical steps, identify gaps, and assess whether the proof meets publication standards."
- **Consistency check**: "Verify this definition is consistent with the Fragile framework. Check against existing definitions in the framework documents, verify notation consistency, and identify any contradictions."
- **Clarity check**: "Assess if the proof structure is clear and well-organized. Evaluate pedagogical flow, identify unclear steps, and suggest improvements for readability."

Each reviewer will provide:
1. Critical analysis with severity ratings
2. Specific issues with location, problem, impact, and suggested fix
3. Checklist of required proofs
4. Prioritized action plan
5. Implementation checklist

**Review Comparison Protocol:**
1. **Consensus Issues** (both reviewers agree): High confidence → prioritize these
2. **Discrepancies** (reviewers contradict): Potential hallucination → verify manually against framework docs
3. **Unique Issues** (only one reviewer identifies): Medium confidence → verify before accepting
4. **Cross-Validation**: Always check specific claims against `docs/glossary.md` and source documents before implementing

EXTREMELY IMPORTANT: ALWAYS USE GEMINI 2.5 PRO (never flash or other variants)

#### Step 3: Critical Evaluation of Dual Feedback
**IMPORTANT**: You must critically evaluate BOTH reviewers' feedback:
- **Compare outputs**: Identify where reviewers agree (high confidence) vs. disagree (requires investigation)
- **Cross-check suggestions** against existing framework definitions in `docs/glossary.md` and source documents
- **Verify mathematical correctness** of proposed fixes by checking proofs and definitions
- **Assess preservation** of algorithmic intent and framework consistency
- **Investigate discrepancies**: When reviewers contradict, manually verify against source documents
- **Flag hallucinations**: If a claim cannot be verified in framework docs, reject it
- **If you disagree** with feedback from either or both reviewers, you MUST:
  1. Document your reasoning clearly with references to framework documents
  2. Inform the user of the disagreement
  3. Propose an alternative approach with mathematical justification
  4. Let the user make the final decision

#### Step 4: Implement Changes
- Systematically address feedback following the implementation checklist
- Maintain mathematical notation consistency
- Update cross-references as needed
- Ensure proper formatting (spacing before `$$`, correct directive syntax)

#### Step 5: Final Formatting Pass
Use the tools in `src/tools/` to ensure correct formatting:
- `convert_unicode_math.py` - Convert Unicode to LaTeX
- `convert_backticks_to_math.py` - Convert backticks to dollar signs
- `fix_math_formatting.py` - Fix LaTeX block spacing
- `format_math_blocks.py` - Comprehensive formatting fixes
- `fix_complex_subscripts.py` - Handle complex subscript notation

### Disagreement Protocol

When you disagree with feedback from either reviewer (Gemini or Codex):

**DO:**
- Explain the mathematical reasoning behind your position with specific references
- Reference specific framework axioms or established results from `docs/glossary.md` and source documents
- Verify your position by checking source documents before disagreeing
- Propose alternative solutions that maintain rigor
- Present all perspectives (yours, Gemini's, Codex's) to the user
- Note when reviewers contradict each other (strong signal for manual verification)

**DON'T:**
- Silently ignore reviewer feedback without investigation
- Implement changes you believe are mathematically incorrect
- Defer to either reviewer without understanding and verifying the issue
- Create inconsistencies with the existing framework
- Accept claims that cannot be verified in framework documents
- Assume either reviewer is always correct (both can hallucinate)

**Example Response (Single Reviewer Disagreement):**
```
I've reviewed Gemini's feedback on the convergence proof. I disagree with
suggestion #3 regarding the Lipschitz constant bound because:

1. The framework's Axiom of Bounded Displacement (def-axiom-bounded-displacement)
   already guarantees L_φ ≤ 1 through Lemma 2.3 in 01_fragile_gas_framework.md
2. Gemini's suggested bound of L_φ ≤ √(1 + λ_v) would contradict this
3. The existing proof structure relies on the tighter bound
4. Codex's review confirms the current bound is correct

Proposed alternative: Keep the current bound and add an explicit reference
to Lemma 2.3 to clarify the justification.

Would you like me to proceed with this approach?
```

**Example Response (Reviewer Contradiction):**
```
Gemini and Codex provide contradictory feedback on the compactness argument:

- Gemini claims: Sequential compactness is insufficient, need uniform bound
- Codex claims: Sequential compactness is adequate given the metric structure

I've verified against the framework documents:
- Lemma 4.2 in 05_mean_field.md establishes uniform bounds (supports Gemini)
- However, the metric structure assumption isn't stated in our theorem (Codex correct about gap)

Both reviewers identify real issues but from different angles. Proposed resolution:
1. Add explicit uniform bound hypothesis (addressing Gemini's concern)
2. State the metric structure assumption clearly (addressing Codex's concern)
3. Reference Lemma 4.2 for the uniform bound

Would you like me to implement this combined fix?
```

### Mathematical Notation Conventions

Follow these conventions from the framework documents:

**Greek Letters:**
- α (alpha): Exploitation weight for reward
- β (beta): Exploitation weight for diversity
- γ (gamma): Friction coefficient
- δ (delta): Cloning noise scale
- ε (epsilon): Regularization parameters
- η (eta): Rescale lower bound
- σ (sigma): Perturbation noise scale
- τ (tau): Time step size
- λ (lambda): Weight parameters (e.g., λ_v for velocity)

**Calligraphic Letters:**
- 𝒳 (mathcal{X}): State space
- 𝒴 (mathcal{Y}): Algorithmic space
- 𝒮 (mathcal{S}): Swarm configuration
- 𝒜 (mathcal{A}): Alive set
- 𝒟 (mathcal{D}): Dead set

**Common Operators:**
- d_𝒳: Metric on state space
- d_alg: Algorithmic distance
- φ (phi): Projection map
- ψ (psi): Squashing map
- Ψ (Psi): Operator (e.g., Ψ_clone, Ψ_kin)

### Theorem-Proof Workflow: Organic High-to-Low Level Integration

**NEW**: TheoremBox now contains an optional `proof` field that organically bridges high-level theorem validation with low-level proof development. This enables a natural workflow from theorem statement → proof sketch → proof expansion → verification.

#### Key Features

1. **TheoremBox.proof** (Optional[ProofBox]): Attach proof directly to theorem
2. **TheoremBox.proof_status**: Track progress (unproven/sketched/expanded/verified)
3. **Bidirectional navigation**: Theorem ↔ Proof references
4. **Auto-validation**: Proof validated against theorem when attached
5. **Workflow functions**:
   - `create_proof_sketch_from_theorem()`: Auto-generates proof structure
   - `attach_proof_to_theorem()`: Attaches with validation
   - `theorem.validate_proof()`: Comprehensive validation

#### Complete Workflow Example

```python
from fragile.proofs import (
    create_simple_theorem,
    create_proof_sketch_from_theorem,
    attach_proof_to_theorem,
)

# 1. Create theorem (high-level specification)
theorem = create_simple_theorem(
    label="thm-kl-convergence",
    name="KL Convergence Rate",
    output_type=TheoremOutputType.PROPERTY,
    input_objects=["obj-discrete-system"],
)

# Add property requirements
theorem = theorem.model_copy(update={
    'properties_required': {
        'obj-discrete-system': ['prop-lipschitz', 'prop-bounded']
    }
})

# 2. Generate proof sketch (auto-creates structure from theorem)
proof = create_proof_sketch_from_theorem(
    theorem=theorem,
    objects=objects,  # Dictionary of MathematicalObject instances
    strategy="LSI → Grönwall → exponential rate",
    num_steps=5
)

# 3. Attach & validate proof
theorem = attach_proof_to_theorem(
    theorem=theorem,
    proof=proof,
    objects=objects,
    validate=True  # Auto-validates against theorem spec
)

# 4. Check status
print(theorem.proof_status)  # "sketched"
print(theorem.has_proof())   # True
print(theorem.is_proven())   # False (steps not fully expanded)

# 5. Validate proof comprehensively
validation = theorem.validate_proof(objects)
print_validation_result(validation)

# 6. Expand proof steps (with LLM)
for step in theorem.proof.get_sketched_steps():
    # LLM expands step with full mathematical derivation...
    expanded_step = expand_step_with_llm(step)

# 7. Check completion
print(theorem.is_proven())   # True (all steps expanded)
print(theorem.proof_status)  # "expanded" or "verified"
```

#### Convenience Methods

```python
# Check proof status
theorem.has_proof()      # → bool: Does theorem have attached proof?
theorem.is_proven()      # → bool: Are all proof steps fully expanded?

# Attach proof directly (alternative to attach_proof_to_theorem)
theorem_with_proof = theorem.attach_proof(proof)

# Validate proof
validation_result = theorem.validate_proof(objects)
# Returns ProofValidationResult with:
#   - is_valid: bool
#   - mismatches: List[ProofTheoremMismatch]
#   - warnings: List[str]
```

#### Bidirectional Navigation

```python
# Theorem → Proof
proof_id = theorem.proof.proof_id

# Proof → Theorem (back-reference)
theorem_label = proof.theorem.label

# Full navigation
assert theorem.proof.theorem.label == theorem.label  # ✓
```

#### Status Progression

Proof status updates automatically based on step completion:

1. **unproven**: No proof attached (theorem.proof is None)
2. **sketched**: Proof attached, some/all steps in SKETCHED status
3. **expanded**: All steps expanded, not yet verified
4. **verified**: All steps verified after dual review

#### Complete Example

See `examples/theorem_proof_workflow.py` for a comprehensive demonstration including:
- Creating mathematical objects with properties
- Defining theorems with property requirements
- Auto-generating proof sketches from theorems
- Attaching proofs with validation
- Checking proof status and completion
- Comprehensive dataflow validation

### Common Mathematical Tasks

**Adding a New Theorem:**
1. Read related sections to understand context
2. Draft theorem with proper `{prf:theorem}` directive
3. Write complete proof with detailed steps
4. Submit to Gemini for rigor review
5. Address feedback critically
6. Run formatting tools
7. Update cross-references

**Extending Existing Proofs:**
1. Read the full proof context (use offset/limit for large files)
2. Identify the specific gap or extension point
3. Draft additional content maintaining proof flow
4. Verify consistency with framework axioms
5. Submit extension to Gemini for review
6. Implement feedback after critical evaluation

**Fixing Mathematical Errors:**
1. Understand the error and its propagation
2. Check all dependent results
3. Draft correction with justification
4. Submit to Gemini for verification
5. Update all affected downstream results
6. Run formatting tools on modified files

## Notes

- This is research code: prioritize mathematical correctness and clarity over performance
- Breaking changes to algorithm behavior should be documented in markdown files first
- All visualizations use HoloViz stack (HoloViews/hvPlot + Panel) with Bokeh backend for 2D and Plotly for 3D
- PyTorch is used for vectorization, not for autodiff/gradients
- The project uses UV for dependency management and Hatch for environment management
- Python 3.10 is required for Atari/plangym compatibility
- Lint and docs environments can use Python 3.12 (no Atari dependency)
- **Mathematical documentation requires collaborative review with Gemini via MCP**
- **Always critically evaluate Gemini's feedback before implementation**
- **Use formatting tools in src/tools/ to ensure consistent mathematical notation**
