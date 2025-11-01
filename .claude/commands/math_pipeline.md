---
description: Autonomous proof development pipeline - Process documents or folders through Proof Sketcher → Theorem Prover → Math Reviewer workflow
---

# Autonomous Math Pipeline

You are executing the **Autonomous Math Proof Development Pipeline**, a sophisticated system that processes mathematical documents or entire folders, automatically developing complete, publication-ready proofs for all theorems and lemmas using the three-agent workflow:

**Proof Sketcher** → **Theorem Prover** → **Math Reviewer** → **Integration**

This pipeline is designed to run autonomously for hours or days without user intervention, handling dependencies, iterating for quality, and integrating results into source documents.

---

## Pipeline Configuration

Based on design decisions, the pipeline operates with:

- **Integration Strategy**: **Hybrid**
  - Auto-edit documents for high-confidence proofs (rigor ≥ 9/10)
  - Generate separate proof files for manual review (8 ≤ rigor < 9)
  - Report proofs below threshold (rigor < 8) as needing refinement

- **Dependency Handling**: **Auto-resolve**
  - Automatically sketch and prove missing lemmas recursively
  - Topologically sort theorems by dependencies
  - Build complete dependency chains before main results

- **Quality Standards**: **Auto-iterate**
  - Re-expand proofs until rigor ≥ 8/10
  - Maximum 3 iterations per theorem
  - Focus on specific gaps identified by Math Reviewer

- **Scope**: **Flexible**
  - If argument is a folder: Process all documents in folder
  - If argument is a single document: Process only that document
  - Automatically detect and skip generated files

---

## Input Handling

### Parse Command Arguments

The command accepts either a folder path or a single document path:

**Format 1: Process entire folder**
```
/math_pipeline docs/source/1_euclidean_gas
```

**Format 2: Process single document**
```
/math_pipeline docs/source/1_euclidean_gas/09_kl_convergence.md
```

**Format 3: Resume interrupted pipeline**
```
/math_pipeline docs/source/1_euclidean_gas
```
(Will detect existing `pipeline_state.json` and resume from last checkpoint)

### Determine Scope

1. **Check if path is a file or directory**:
   ```bash
   if [ -f "$PATH" ]; then
       # Single document mode
   elif [ -d "$PATH" ]; then
       # Folder mode
   fi
   ```

2. **Single Document Mode**:
   - Process only the specified document
   - Create state file in document's directory
   - All output folders (sketcher/, proofs/, reviewer/) created in document's directory

3. **Folder Mode**:
   - Find all `.md` files in the folder
   - Exclude generated files matching patterns:
     - `*CORRECTIONS*`
     - `*REVIEW*`
     - `*REVISION*`
     - `*agent_output*`
     - `*backup*`
     - `pipeline_*.md`
   - Create state file in folder root
   - All output folders created in folder root

---

## PHASE 0: Initialization & Discovery

### Step 0.1: Validate Input and Check for Resume

1. **Validate the provided path**:
   - Verify it exists
   - Determine if file or directory
   - Inform user of detected mode

2. **Check for existing state file**:
   - Look for `pipeline_state.json` in the target directory
   - If found:
     - Read existing state
     - Check status: `in_progress`, `completed`, `interrupted`
     - Ask user if they want to resume or start fresh
     - If resume: Load existing state and skip to Phase 1
   - If not found: Create new state file

3. **Create output directories** (if they don't exist):
   ```bash
   mkdir -p sketcher/
   mkdir -p mathster/
   mkdir -p reviewer/
   ```

### Step 0.2: Document Discovery

**For Single Document Mode**:
1. Read the single document
2. Extract all theorems, lemmas, propositions
3. Proceed to Step 0.3

**For Folder Mode**:
1. Find all markdown files:
   ```bash
   find $FOLDER -maxdepth 1 -name "*.md" -type f \
     ! -name "*CORRECTIONS*" \
     ! -name "*REVIEW*" \
     ! -name "*REVISION*" \
     ! -name "*agent_output*" \
     ! -name "*backup*" \
     ! -name "pipeline_*.md"
   ```

2. For each file:
   - Read the document (use strategic reading with offset/limit for large files)
   - Extract metadata: filename, size, sections
   - Add to document list

### Step 0.3: Theorem Extraction

For each document in scope:

1. **Search for theorem-like directives**:
   - Use grep to find patterns:
     - `{prf:theorem}`
     - `{prf:lemma}`
     - `{prf:proposition}`
     - `{prf:corollary}`

2. **Extract theorem metadata**:
   - Label (e.g., `:label: thm-main-result`)
   - Title/name
   - Location (line number)
   - Document path

3. **Check for existing proofs**:
   - After each theorem directive, search for `{prf:proof}` within next 50 lines
   - If found: Mark as "has_proof"
   - If not found: Mark as "needs_proof"
   - If found but incomplete (contains "TODO", "SKETCH", "INCOMPLETE"): Mark as "needs_proof"

4. **Create theorem record**:
   ```json
   {
     "label": "thm-kl-convergence-euclidean",
     "type": "theorem",
     "document": "09_kl_convergence.md",
     "document_path": "docs/source/1_euclidean_gas/09_kl_convergence.md",
     "line_number": 245,
     "title": "N-Particle Exponential KL-Convergence",
     "has_proof": false,
     "needs_proof": true,
     "status": "pending",
     "dependencies": [],
     "attempts": 0
   }
   ```

### Step 0.4: Dependency Graph Construction

1. **For each theorem needing proof**:
   - Read the theorem statement and surrounding context (±100 lines)
   - Search for `{prf:ref}` references within the theorem statement and proof context
   - Extract referenced labels (e.g., `{prf:ref}\`lemma-fisher-info\``)

2. **Build dependency list**:
   - For each reference:
     - Check if it's another theorem/lemma in our processing list
     - Add to dependencies array
     - Check if referenced lemma has a proof
     - If not: Add referenced lemma to processing queue

3. **Detect missing dependencies**:

   **Critical**: Verify each dependency against existing framework proofs!

   For each referenced lemma/theorem:

   a. **Search `docs/glossary.md`**:
      ```bash
      grep -A 5 "label: {dependency_label}" docs/glossary.md
      ```
      - If found: Extract source document path
      - Read source document to check for proof

   b. **Search `docs/reference.md`**:
      ```bash
      grep -A 10 "{dependency_label}" docs/reference.md
      ```
      - Check for proof location reference

   c. **Search all documents in current scope**:
      - Use glob to find all markdown files
      - Grep for the dependency label
      - Check if proof exists

   d. **Classification**:
      - **Dependency satisfied**: Proof exists in framework → No action needed
      - **In current pipeline**: Will be proven during this pipeline → Add to execution order
      - **Truly missing**: Not found anywhere → Mark for auto-resolution

   e. **Update dependency record**:
      ```json
      {
        "label": "lemma-wasserstein-contraction",
        "status": "satisfied_externally",
        "proof_location": "docs/source/1_euclidean_gas/04_wasserstein_contraction.md:245"
      }
      ```
      or
      ```json
      {
        "label": "lemma-entropy-bound",
        "status": "truly_missing",
        "needs_resolution": true
      }
      ```

4. **Topological sort**:
   - Order theorems so lemmas are proven before dependent theorems
   - Detect circular dependencies:
     - If found: Report error and exclude that subgraph
     - Circular dependencies indicate a logical error in the framework
   - Create execution order list

### Step 0.5: Create Execution Plan

1. **Calculate estimates**:
   - Per theorem: ~3-5 hours average (sketch 45min, expand 2-4h, review 30min)
   - Account for iterations (average 1.5 iterations per theorem)
   - Missing lemmas: Add recursive time estimates
   - Total estimated time

2. **Generate execution plan**:
   ```json
   {
     "total_theorems": 15,
     "needs_proof": 12,
     "has_proof": 3,
     "missing_lemmas": 4,
     "execution_order": [
       "lemma-bounded-displacement",
       "lemma-fisher-info",
       "lemma-wasserstein-contraction",
       "lemma-entropy-bound",
       "thm-kinetic-lsi",
       "thm-cloning-contraction",
       "thm-kl-convergence-euclidean"
     ],
     "estimated_time_hours": 42,
     "estimated_completion": "2025-10-26T09:30:00"
   }
   ```

3. **Present plan to user**:
   ```markdown
   ## Math Pipeline Execution Plan

   **Target**: docs/source/1_euclidean_gas (Folder Mode)
   **Documents**: 12 files
   **Theorems/Lemmas to prove**: 15
   **Missing dependencies to resolve**: 4

   **Estimated total time**: 42 hours
   **Estimated completion**: 2025-10-26 09:30:00

   **Execution order** (topologically sorted):
   1. lemma-bounded-displacement (09_kl_convergence.md) - Missing dependency
   2. lemma-fisher-info (09_kl_convergence.md) - Missing dependency
   3. lemma-wasserstein-contraction (04_wasserstein_contraction.md)
   4. lemma-entropy-bound (09_kl_convergence.md) - Missing dependency
   5. thm-kinetic-lsi (09_kl_convergence.md) - Depends on lemma-fisher-info
   6. thm-cloning-contraction (03_cloning.md) - Depends on lemma-wasserstein-contraction
   7. thm-kl-convergence-euclidean (09_kl_convergence.md) - Main result
   ...

   **Pipeline will run autonomously. Progress saved to pipeline_state.json.**
   **You can interrupt and resume at any time.**

   Starting pipeline in 3 seconds...
   ```

### Step 0.6: Initialize State File

Create `pipeline_state.json` in the target directory:

```json
{
  "version": "1.0",
  "mode": "folder",
  "target_path": "docs/source/1_euclidean_gas",
  "start_time": "2025-10-24T15:30:00",
  "last_update": "2025-10-24T15:30:00",
  "status": "initializing",
  "execution_plan": { ... },
  "theorems": [ ... ],
  "statistics": {
    "total_theorems": 15,
    "completed": 0,
    "in_progress": 0,
    "pending": 15,
    "failed": 0,
    "auto_integrated": 0,
    "manual_review": 0
  },
  "current_theorem": null,
  "errors": [],
  "resume_instructions": "Run '/math_pipeline docs/source/1_euclidean_gas' to resume"
}
```

Update status to `in_progress` and proceed to Phase 1.

---

## PHASE 1: Proof Development Loop

This phase processes each theorem in dependency order, sketching and expanding proofs with quality iteration.

### Main Loop Structure

```
FOR each theorem in execution_plan.execution_order:
    1. Check if already completed (in state file)
    2. Update state: current_theorem = theorem_label, status = "processing"
    3. Resolve missing dependencies (recursive)
    4. Execute proof workflow (sketch → expand → review)
    5. Quality iteration (if rigor < 8/10)
    6. Update state: mark as completed
    7. Save state file (checkpoint)
END FOR
```

### Step 1.1: Pre-Theorem Checks

Before processing each theorem:

1. **Load current state** from `pipeline_state.json`

2. **Check if theorem already completed**:
   - If status = "completed": Skip to next theorem
   - If status = "failed" after 3 attempts: Skip and log warning
   - If status = "in_progress": Resume from last stage

3. **Update state**:
   ```json
   {
     "current_theorem": "thm-kl-convergence-euclidean",
     "current_stage": "resolving_dependencies",
     "statistics": { "in_progress": 1 }
   }
   ```

4. **Progress report** (output to user):
   ```markdown
   ===================================================================
   PROCESSING THEOREM [3/15]: thm-kl-convergence-euclidean
   ===================================================================
   Document: 09_kl_convergence.md (line 245)
   Title: N-Particle Exponential KL-Convergence
   Dependencies: lemma-fisher-info, lemma-wasserstein-contraction
   Attempt: 1/3

   [Checking dependencies...]
   ```

### Step 1.2: Resolve Missing Dependencies

For each dependency in theorem's dependency list:

1. **Check dependency status**:
   - Search state file for dependency theorem
   - If status = "completed": Continue
   - If status = "pending" or not found: **Verify it's truly missing before auto-resolving**

2. **Verify dependency is truly missing**:

   **Critical step**: Before marking a lemma as "missing", search the framework for existing proofs!

   a. **Search `docs/glossary.md`**:
      ```bash
      grep -A 5 "label: {lemma_label}" docs/glossary.md
      ```
      - Check if lemma exists in glossary
      - Extract source document path
      - If found: Read source document to check for proof

   b. **Search `docs/reference.md`**:
      ```bash
      grep -A 10 "{lemma_label}" docs/reference.md
      ```
      - Check for full statement and proof location
      - If proof location specified: Verify proof exists

   c. **Search previous documents in chapter**:
      - For lemma in document N, search documents 1 through N-1
      - Use grep to find `{prf:theorem}` or `{prf:lemma}` with matching label
      - Example:
        ```bash
        grep -l ":label: {lemma_label}" docs/source/1_euclidean_gas/0*.md
        ```

   d. **Check for proof in referenced document**:
      - Read the document where lemma is defined
      - Search for `{prf:proof}` directive after lemma statement
      - If proof exists: **Dependency is satisfied, not missing!**

   e. **Verification outcome**:
      ```
      IF proof found in glossary/reference/previous docs:
          Log: "✓ Dependency {lemma_label} already proven in {source_doc}"
          Mark as "dependency_satisfied_externally"
          Update theorem record with proof location
          RETURN success (no auto-resolution needed)
      ELSE:
          Log: "⚠ Dependency {lemma_label} not found - truly missing"
          Proceed to auto-resolution
      ```

3. **Recursive dependency resolution** (only for truly missing lemmas):
   ```
   FUNCTION resolve_dependency(lemma_label):
       # First: Verify it's truly missing (Step 2 above)
       IF lemma has existing proof in framework:
           Log: "✓ Found existing proof for {lemma_label} in {source}"
           RETURN success

       # Only auto-resolve if truly missing
       IF lemma is in current pipeline and completed:
           RETURN success
       ELSE:
           Log: "⚠ Auto-resolving truly missing lemma: {lemma_label}"
           Log: "  (Not found in: glossary, reference, previous documents)"
           Apply full proof workflow to lemma (Steps 1.3-1.6)
           IF successful:
               RETURN success
           ELSE:
               Log error: "Failed to resolve dependency: {lemma_label}"
               Mark main theorem as "blocked"
               RETURN failure
   ```

4. **Circular dependency detection**:
   - Maintain stack of currently-processing theorems
   - If lemma depends on a theorem already in stack: **Circular dependency**
   - Report error and abort that theorem branch

5. **Update progress**:
   ```markdown
   [Dependency Resolution]
   ✓ lemma-fisher-info (completed previously in this pipeline)
   🔍 lemma-entropy-bound (checking framework...)
     → Searching glossary... not found
     → Searching reference... not found
     → Searching previous documents (01-08)... not found
     ⚠ Truly missing - auto-resolving...
     → Launching sub-pipeline for lemma-entropy-bound...
     → [Sub-pipeline output...]
     ✓ lemma-entropy-bound resolved successfully

   🔍 lemma-wasserstein-contraction (checking framework...)
     → Searching glossary... found!
     → Source: docs/source/1_euclidean_gas/04_wasserstein_contraction.md
     → Reading source document... proof exists at line 245!
     ✓ Dependency satisfied (existing proof in 04_wasserstein_contraction.md)

   ✓ All dependencies satisfied

   [Proceeding to proof development...]
   ```

### Step 1.3: Launch Proof Sketcher Agent

1. **Update state**:
   ```json
   {
     "current_stage": "sketching",
     "last_update": "2025-10-24T15:45:00"
   }
   ```

2. **Prepare Proof Sketcher prompt**:
   ```markdown
   Load instructions from: .claude/agents/proof-sketcher.md

   Sketch proof for theorem: {theorem_label}
   Document: {document_path}
   Depth: thorough

   **Context**:
   - This is part of an autonomous pipeline
   - Dependencies already proven: {dependency_list}
   - Focus on creating actionable strategy suitable for expansion
   ```

3. **Launch agent using Task tool**:
   ```python
   Task(
       description=f"Sketch proof for {theorem_label}",
       subagent_type="general-purpose",
       prompt=sketcher_prompt
   )
   ```

4. **Wait for agent completion**:
   - Agent outputs sketch to: `sketcher/sketch_{timestamp}_proof_{document_name}.md`
   - Parse agent output to find sketch file path
   - Verify sketch file was created successfully

5. **Validate sketch**:
   - Read sketch file
   - Check for required sections:
     - Theorem Statement
     - Proof Strategy Comparison
     - Detailed Proof Sketch
     - Expansion Roadmap
   - Extract proof steps (typically 3-7 steps)
   - Extract estimated expansion time

6. **Update theorem record**:
   ```json
   {
     "sketch_file": "sketcher/sketch_20251024_1545_proof_09_kl_convergence.md",
     "sketch_timestamp": "2025-10-24T15:45:00",
     "proof_steps": 6,
     "status": "sketched"
   }
   ```

7. **Progress report**:
   ```markdown
   ✓ Proof sketch completed successfully
   File: sketcher/sketch_20251024_1545_proof_09_kl_convergence.md
   Strategy: Entropy-transport Lyapunov method
   Steps: 6
   Estimated expansion time: 3-4 hours

   [Proceeding to proof expansion...]
   ```

### Step 1.4: Launch Theorem Prover Agent

1. **Update state**:
   ```json
   {
     "current_stage": "expanding",
     "last_update": "2025-10-24T15:50:00"
   }
   ```

2. **Prepare Theorem Prover prompt**:
   ```markdown
   Load instructions from: .claude/agents/theorem-prover.md

   Expand proof sketch:
   {sketch_file_path}

   **Configuration**:
   - Depth: standard (Annals of Mathematics rigor)
   - Focus: All epsilon-delta complete, all constants explicit
   - This is attempt {attempt_number}/3
   {if attempt > 1: Include focus areas from previous review}
   ```

3. **Launch agent using Task tool**:
   ```python
   Task(
       description=f"Expand proof for {theorem_label}",
       subagent_type="general-purpose",
       prompt=theorem_prover_prompt
   )
   ```

4. **Wait for agent completion**:
   - Agent outputs complete proof to: `proofs/proof_{timestamp}_{theorem_label}.md`
   - Parse agent output to find proof file path
   - Verify proof file was created successfully
   - This may take 2-4 hours per theorem

5. **Validate proof structure**:
   - Read proof file
   - Check for required sections:
     - Theorem Statement
     - Proof Expansion Comparison
     - Framework Dependencies
     - Complete Rigorous Proof
     - Verification Checklist
     - Publication Readiness Assessment
   - Extract publication assessment score

6. **Update theorem record**:
   ```json
   {
     "proof_file": "mathster/proof_20251024_1850_thm_kl_convergence_euclidean.md",
     "proof_timestamp": "2025-10-24T18:50:00",
     "expansion_time_minutes": 180,
     "status": "expanded",
     "attempts": 1
   }
   ```

7. **Progress report**:
   ```markdown
   ✓ Proof expansion completed successfully
   File: proofs/proof_20251024_1850_thm_kl_convergence_euclidean.md
   Length: 1247 lines
   Expansion time: 3 hours 0 minutes

   [Proceeding to validation...]
   ```

### Step 1.5: Launch Math Reviewer Agent

1. **Update state**:
   ```json
   {
     "current_stage": "reviewing",
     "last_update": "2025-10-24T18:55:00"
   }
   ```

2. **Prepare Math Reviewer prompt**:
   ```markdown
   Load instructions from: .claude/agents/math-reviewer.md

   Review the complete proof at:
   {proof_file_path}

   **Focus**:
   - Mathematical rigor and completeness
   - Publication readiness (Annals of Mathematics standard)
   - Framework consistency
   - All epsilon-delta arguments complete
   - All measure theory justified
   - All constants explicit

   Depth: thorough
   ```

3. **Launch agent using Task tool**:
   ```python
   Task(
       description=f"Review proof for {theorem_label}",
       subagent_type="general-purpose",
       prompt=math_reviewer_prompt
   )
   ```

4. **Wait for agent completion**:
   - Agent outputs review to: `reviewer/review_{timestamp}_{proof_filename}.md`
   - Parse agent output to find review file path
   - Verify review file was created successfully

5. **Extract rigor assessment**:
   - Read review file
   - Find "Publication Readiness Assessment" section
   - Extract numerical scores:
     - Mathematical Rigor: X/10
     - Completeness: X/10
     - Clarity: X/10
     - Overall Score: X/10
   - Extract verdict: "MEETS STANDARD" / "MINOR REVISIONS" / "MAJOR REVISIONS"
   - Extract specific issues (CRITICAL, MAJOR, MINOR)

6. **Update theorem record**:
   ```json
   {
     "review_file": "reviewer/review_20251024_1900_proof_thm_kl_convergence.md",
     "review_timestamp": "2025-10-24T19:00:00",
     "rigor_score": 9.0,
     "completeness_score": 9.0,
     "clarity_score": 8.5,
     "overall_score": 8.8,
     "verdict": "MEETS STANDARD",
     "critical_issues": 0,
     "major_issues": 0,
     "minor_issues": 2,
     "status": "reviewed"
   }
   ```

7. **Progress report**:
   ```markdown
   ✓ Proof review completed successfully
   File: reviewer/review_20251024_1900_proof_thm_kl_convergence.md

   [Review Assessment]
   Mathematical Rigor: 9/10
   Completeness: 9/10
   Clarity: 8.5/10
   Overall Score: 8.8/10

   Verdict: ✅ MEETS ANNALS OF MATHEMATICS STANDARD
   Critical Issues: 0
   Major Issues: 0
   Minor Issues: 2

   ✓ Proof meets quality threshold (score ≥ 8/10)
   → Marked for auto-integration (score ≥ 9/10)
   ```

### Step 1.6: Quality Iteration

If the overall rigor score is below 8/10, iterate:

1. **Check iteration count**:
   - If attempts ≥ 3: Mark as "needs_manual_refinement" and move to next theorem
   - Otherwise: Proceed with iteration

2. **Extract focus areas from review**:
   - Parse review file for CRITICAL and MAJOR issues
   - Extract specific gaps:
     - Incomplete epsilon-delta arguments (with locations)
     - Missing measure theory justifications
     - Unjustified constants
     - Unhandled edge cases
   - Create focus list

3. **Prepare focused re-expansion**:
   ```markdown
   Load instructions from: .claude/agents/theorem-prover.md

   Expand proof sketch:
   {sketch_file_path}

   **Configuration**:
   - Depth: maximum (address all gaps)
   - This is iteration {attempt_number}/3

   **Focus Areas** (from Math Reviewer feedback):
   - Step 4, line 245: Complete epsilon-delta for limit as N→∞
   - Step 5, line 320: Verify Fubini condition 2 explicitly
   - All steps: Track constant C_Fisher with explicit formula
   - Edge case: Handle k=1 (single walker) scenario
   ```

4. **Re-launch Theorem Prover** with focus areas (return to Step 1.4)

5. **Increment attempt counter**:
   ```json
   {
     "attempts": 2,
     "iteration_history": [
       {"attempt": 1, "score": 7.5, "issues": ["incomplete epsilon-delta", "missing Fubini"]},
       {"attempt": 2, "score": 8.8, "issues": ["minor notation inconsistency"]}
     ]
   }
   ```

6. **Progress report for iteration**:
   ```markdown
   ⚠ Proof below quality threshold (score 7.5/10 < 8.0/10)

   [Identified Gaps]
   - CRITICAL: Step 4 limit lacks complete epsilon-delta argument
   - MAJOR: Fubini condition 2 not explicitly verified
   - MAJOR: Constant C_Fisher formula missing

   → Re-expanding with focused attention (attempt 2/3)...

   [Re-expansion in progress...]
   ```

### Step 1.7: Mark as Completed

Once the proof meets quality standards (score ≥ 8/10):

1. **Determine integration strategy**:
   - If overall_score ≥ 9.0: Mark for **auto-integration**
   - If 8.0 ≤ overall_score < 9.0: Mark for **manual review**
   - If overall_score < 8.0 after 3 attempts: Mark as **needs_refinement**

2. **Update theorem record**:
   ```json
   {
     "status": "completed",
     "integration_status": "ready_for_auto_integration",
     "completion_timestamp": "2025-10-24T19:05:00",
     "total_time_minutes": 195
   }
   ```

3. **Update state file statistics**:
   ```json
   {
     "statistics": {
       "completed": 3,
       "in_progress": 0,
       "pending": 12
     },
     "current_theorem": null
   }
   ```

4. **Save state file** (checkpoint for resume)

5. **Progress report**:
   ```markdown
   ===================================================================
   ✅ THEOREM COMPLETED: thm-kl-convergence-euclidean
   ===================================================================
   Total time: 3 hours 15 minutes
   Attempts: 2
   Final score: 8.8/10
   Integration: Auto-integration (high confidence)

   Files generated:
   - Sketch: sketcher/sketch_20251024_1545_proof_09_kl_convergence.md
   - Proof: proofs/proof_20251024_1850_thm_kl_convergence_euclidean.md
   - Review: reviewer/review_20251024_1900_proof_thm_kl_convergence.md

   [Proceeding to next theorem...]
   [3/15 theorems completed, 12 remaining]
   [Estimated time remaining: 39 hours]
   ```

### Step 1.8: Loop Continue

Repeat Steps 1.1-1.7 for each theorem in execution order until all are processed.

After all theorems processed, proceed to Phase 2.

---

## PHASE 2: Integration & Document Editing

This phase integrates completed proofs back into the original documents.

### Step 2.1: Prepare Integration List

1. **Load state file** to get all completed theorems

2. **Filter by integration status**:
   - Auto-integration list: `integration_status = "ready_for_auto_integration"` (score ≥ 9/10)
   - Manual review list: `integration_status = "ready_for_manual_review"` (8/10 ≤ score < 9/10)
   - Needs refinement list: `integration_status = "needs_refinement"` (score < 8/10)

3. **Group by document**:
   ```json
   {
     "09_kl_convergence.md": [
       {"label": "thm-kl-convergence", "integration": "auto"},
       {"label": "lemma-fisher-info", "integration": "manual"}
     ],
     "03_cloning.md": [
       {"label": "thm-keystone", "integration": "auto"}
     ]
   }
   ```

4. **Integration plan**:
   ```markdown
   ## Integration Plan

   **Auto-Integration** (9 theorems with rigor ≥ 9/10):
   - docs/source/1_euclidean_gas/09_kl_convergence.md: 3 theorems
   - docs/source/1_euclidean_gas/03_cloning.md: 2 theorems
   - docs/source/1_euclidean_gas/04_wasserstein_contraction.md: 4 theorems

   **Manual Review** (3 theorems with 8 ≤ rigor < 9):
   - docs/source/1_euclidean_gas/09_kl_convergence.md: 2 lemmas
   - docs/source/1_euclidean_gas/06_convergence.md: 1 theorem

   **Needs Refinement** (0 theorems):
   (None - all proofs met quality threshold)

   [Starting auto-integration...]
   ```

### Step 2.2: Auto-Integration Workflow

For each theorem marked for auto-integration:

#### Step 2.2.1: Backup Original Document

1. **Create timestamped backup**:
   ```bash
   cp {document_path} {document_path}.backup_$(date +%Y%m%d_%H%M%S)
   ```

2. **Record backup location** in state file:
   ```json
   {
     "backups": [
       {
         "original": "docs/source/1_euclidean_gas/09_kl_convergence.md",
         "backup": "docs/source/1_euclidean_gas/09_kl_convergence.md.backup_20251024_190500",
         "timestamp": "2025-10-24T19:05:00"
       }
     ]
   }
   ```

#### Step 2.2.2: Locate Theorem in Document

1. **Read original document** using Read tool

2. **Find theorem statement**:
   - Search for theorem label: `:label: {theorem_label}`
   - Record exact line number
   - Extract full theorem directive block (from `:::{prf:theorem}` to `:::`)

3. **Check for existing proof**:
   - Search for `{prf:proof}` directive within next 100 lines
   - If found:
     - Extract existing proof
     - Determine if it's complete or just a sketch
     - If complete: Consider whether to replace (check if ours is higher quality)
   - If not found: Proceed with insertion

#### Step 2.2.3: Extract Complete Proof

1. **Read proof file** from `proofs/` directory

2. **Extract Section IV: Complete Rigorous Proof**:
   - This section contains the publication-ready proof
   - Parse from `## IV. Complete Rigorous Proof` to next `##` section
   - Extract the `{prf:proof}` block with all content

3. **Verify proof structure**:
   - Ensure it starts with `:::{prf:proof}`
   - Ensure it ends with `:::`
   - Check for proper LaTeX formatting (blank line before `$$`)
   - Verify all cross-references are valid

#### Step 2.2.4: Integrate Proof into Document

1. **Determine insertion point**:
   - Immediately after the theorem statement's closing `:::`
   - Add exactly one blank line between theorem and proof

2. **Construct integrated content**:
   ```markdown
   :::{prf:theorem} N-Particle Exponential KL-Convergence
   :label: thm-kl-convergence-euclidean

   [Theorem statement...]
   :::

   :::{prf:proof}
   [Complete rigorous proof content extracted from proofs/ file...]
   :::
   ```

3. **Use Edit tool to insert proof**:
   - If no existing proof:
     ```python
     Edit(
         file_path=document_path,
         old_string=theorem_block,  # Just the theorem
         new_string=theorem_block + "\n\n" + proof_block  # Theorem + proof
     )
     ```

   - If existing proof (replace):
     ```python
     Edit(
         file_path=document_path,
         old_string=theorem_block + existing_proof_block,
         new_string=theorem_block + "\n\n" + new_proof_block
     )
     ```

4. **Verify edit succeeded**:
   - Re-read the document
   - Verify proof was inserted at correct location
   - Check that theorem statement is unchanged
   - Ensure no formatting corruption

#### Step 2.2.5: Update Cross-References

1. **Check if proof references other results**:
   - Search for `{prf:ref}` directives in integrated proof
   - Verify each reference resolves correctly:
     - Check if referenced label exists in framework
     - Verify it's in `docs/glossary.md`

2. **Update document-internal references**:
   - If proof references "Step 3" or "Equation (4.2)", verify numbering is correct
   - If document sections were added, update section numbers in references

3. **Mark any broken references** for manual review

#### Step 2.2.6: Format LaTeX Math Blocks

1. **Run math formatting tool**:
   ```bash
   python src/tools/fix_math_formatting.py {document_path} --in-place
   ```

2. **Verify formatting**:
   - Ensure exactly ONE blank line before all `$$` blocks
   - Check proper LaTeX syntax
   - Verify no broken math delimiters

#### Step 2.2.7: Update Theorem Status

```json
{
  "label": "thm-kl-convergence-euclidean",
  "integration_status": "auto_integrated",
  "integration_timestamp": "2025-10-24T19:10:00",
  "integrated_into": "docs/source/1_euclidean_gas/09_kl_convergence.md",
  "backup_file": "docs/source/1_euclidean_gas/09_kl_convergence.md.backup_20251024_190500"
}
```

#### Step 2.2.8: Progress Report

```markdown
✓ Auto-integrated: thm-kl-convergence-euclidean
  Document: 09_kl_convergence.md (line 245)
  Proof length: 847 lines
  Backup: 09_kl_convergence.md.backup_20251024_190500

[9/9 auto-integrations completed]
```

### Step 2.3: Manual Review Proofs

For theorems marked for manual review (8 ≤ score < 9):

1. **Leave proofs in proofs/ directory** (do not auto-integrate)

2. **Create integration guide** for user:
   ```markdown
   ## Proofs for Manual Review

   These proofs meet publication standards (rigor ≥ 8/10) but are recommended for
   manual review before integration due to:
   - Complexity of the result
   - Novel proof techniques
   - Minor notation inconsistencies
   - Score below high-confidence threshold (9/10)

   ### lemma-fisher-info (Score: 8.5/10)
   - **Document**: docs/source/1_euclidean_gas/09_kl_convergence.md (line 187)
   - **Proof file**: proofs/proof_20251024_1645_lemma_fisher_info.md
   - **Review file**: reviewer/review_20251024_1700_proof_lemma_fisher_info.md
   - **Minor issues**:
     - Notation for Fisher information uses both I[ρ] and ℱ(ρ)
     - Edge case k=1 could be more explicit

   **Integration instructions**:
   1. Review proof file and assessment
   2. Address minor issues if desired
   3. Copy Section IV (Complete Rigorous Proof) into source document
   4. Insert after theorem statement at line 187
   5. Format with: python src/tools/fix_math_formatting.py {file} --in-place

   [Repeat for each manual-review proof...]
   ```

3. **Update status**:
   ```json
   {
     "integration_status": "ready_for_manual_review",
     "manual_review_guide": "pipeline_manual_review_guide.md"
   }
   ```

### Step 2.4: Update State File

After all integrations:

```json
{
  "status": "integration_complete",
  "statistics": {
    "total_theorems": 15,
    "completed": 15,
    "auto_integrated": 9,
    "manual_review": 3,
    "needs_refinement": 0,
    "failed": 0
  },
  "integration_summary": {
    "documents_modified": 5,
    "backups_created": 5,
    "proofs_integrated": 9,
    "proofs_for_review": 3,
    "total_lines_added": 7234
  }
}
```

---

## PHASE 3: Final Validation

### Step 3.1: Validate Modified Documents

For each document that was auto-edited:

1. **Launch Math Reviewer for complete validation**:
   ```markdown
   Load instructions from: .claude/agents/math-reviewer.md

   Review the complete document at:
   {document_path}

   **Focus**:
   - Verify all integrated proofs are correct
   - Check for broken cross-references
   - Ensure section transitions are smooth
   - Verify LaTeX formatting is correct
   - Check for any integration artifacts

   Depth: thorough
   ```

2. **Wait for validation review**:
   - Review saved to: `reviewer/review_{timestamp}_{document_name}_final.md`

3. **Check for integration errors**:
   - Parse review for CRITICAL issues related to integration
   - Common integration errors:
     - Broken references after proof insertion
     - Section numbering conflicts
     - Duplicate theorem labels
     - LaTeX formatting issues

4. **If errors found**:
   - Report errors to user
   - Provide restoration instructions:
     ```markdown
     ⚠ Integration errors detected in 09_kl_convergence.md

     To restore original:
     cp 09_kl_convergence.md.backup_20251024_190500 09_kl_convergence.md

     Issues found:
     - Line 456: Broken reference to {prf:ref}`lemma-not-found`
     - Line 789: Duplicate section numbering (two Section 4.3)

     Please manually review and fix, or restore backup.
     ```

### Step 3.2: Cross-Reference Validation

1. **Build list of all references** in modified documents:
   - Grep for `{prf:ref}` patterns
   - Extract all referenced labels

2. **Verify each reference**:
   - Check if label exists in `docs/glossary.md`
   - Check if label exists in local document
   - Flag broken references

3. **Report broken references**:
   ```markdown
   ## Cross-Reference Validation

   **Total references checked**: 247
   **Valid references**: 245
   **Broken references**: 2

   ### Broken References

   1. docs/source/1_euclidean_gas/09_kl_convergence.md:456
      Reference: {prf:ref}`lemma-auxillary-bound`
      Issue: Label not found in framework
      Suggested fix: Check for typo (possibly `lemma-auxiliary-bound`)

   2. docs/source/1_euclidean_gas/04_wasserstein_contraction.md:123
      Reference: {prf:ref}`thm-main-convergence-v2`
      Issue: Label exists but in different document (07_mean_field.md)
      Suggested fix: Add relative path: {prf:ref}`../07_mean_field.md#thm-main-convergence-v2`
   ```

### Step 3.3: LaTeX Formatting Validation

1. **Run formatting check** on all modified documents:
   ```bash
   for doc in {modified_documents}; do
       python src/tools/fix_math_formatting.py $doc --check-only
   done
   ```

2. **Report formatting issues**:
   - Missing blank lines before `$$`
   - Unclosed math delimiters
   - Invalid LaTeX commands

3. **Auto-fix if possible**:
   ```bash
   python src/tools/fix_math_formatting.py $doc --in-place
   ```

---

## PHASE 4: Summary Report & Completion

### Step 4.1: Generate Comprehensive Summary

Create `pipeline_report_{timestamp}.md` in the target directory:

```markdown
# Math Pipeline Completion Report

**Pipeline ID**: pipeline_20251024_153000
**Target**: docs/source/1_euclidean_gas (Folder Mode)
**Start time**: 2025-10-24 15:30:00
**End time**: 2025-10-26 08:45:00
**Total elapsed**: 41 hours 15 minutes

---

## Executive Summary

✅ **Pipeline completed successfully**

- Processed 12 documents
- Proved 15 theorems/lemmas
- Auto-integrated 9 proofs (rigor ≥ 9/10)
- 3 proofs ready for manual review (8 ≤ rigor < 9)
- 0 failed theorems
- 5 documents modified and validated

---

## Statistics

### Proof Development

| Metric | Count |
|--------|-------|
| Total theorems/lemmas | 15 |
| Completed | 15 |
| Auto-integrated | 9 |
| Manual review | 3 |
| Needs refinement | 0 |
| Failed | 0 |
| Missing lemmas resolved | 4 |

### Quality Scores

| Score Range | Count | Percentage |
|-------------|-------|------------|
| 9.0-10.0 (Excellent) | 9 | 60% |
| 8.0-8.9 (Good) | 6 | 40% |
| 7.0-7.9 (Needs work) | 0 | 0% |
| < 7.0 (Inadequate) | 0 | 0% |

**Average rigor score**: 8.9/10

### Time Breakdown

| Phase | Time | Percentage |
|-------|------|------------|
| Initialization | 15 min | 0.6% |
| Proof sketching | 675 min | 27.3% |
| Proof expansion | 1620 min | 65.5% |
| Proof review | 135 min | 5.5% |
| Integration | 30 min | 1.2% |

**Average time per theorem**: 165 minutes (2.75 hours)

---

## Detailed Results by Document

### 09_kl_convergence.md

**Theorems processed**: 5

1. ✅ **lemma-fisher-info** (Manual Review - Score: 8.5/10)
   - Sketch: sketcher/sketch_20251024_1545_proof_09_kl_convergence.md
   - Proof: proofs/proof_20251024_1645_lemma_fisher_info.md
   - Review: reviewer/review_20251024_1700_proof_lemma_fisher_info.md
   - Status: Ready for manual review and integration
   - Time: 2h 15min (1 attempt)

2. ✅ **lemma-wasserstein-contraction** (Auto-Integrated - Score: 9.2/10)
   - Sketch: sketcher/sketch_20251024_1715_proof_09_kl_convergence.md
   - Proof: proofs/proof_20251024_1845_lemma_wasserstein_contraction.md
   - Review: reviewer/review_20251024_1900_proof_lemma_wasserstein.md
   - Status: ✓ Auto-integrated (line 187)
   - Backup: 09_kl_convergence.md.backup_20251024_190500
   - Time: 2h 45min (1 attempt)

3. ✅ **lemma-entropy-bound** (Auto-Integrated - Score: 9.0/10)
   - [Missing dependency - auto-resolved]
   - Sketch: sketcher/sketch_20251024_2015_proof_09_kl_convergence.md
   - Proof: proofs/proof_20251024_2200_lemma_entropy_bound.md
   - Status: ✓ Auto-integrated (line 215)
   - Time: 3h 10min (1 attempt)

4. ✅ **thm-kinetic-lsi** (Auto-Integrated - Score: 9.5/10)
   - Sketch: sketcher/sketch_20251024_2315_proof_09_kl_convergence.md
   - Proof: proofs/proof_20251025_0200_thm_kinetic_lsi.md
   - Review: reviewer/review_20251025_0215_proof_thm_kinetic_lsi.md
   - Status: ✓ Auto-integrated (line 345)
   - Time: 3h 15min (1 attempt)

5. ✅ **thm-kl-convergence-euclidean** (Auto-Integrated - Score: 9.3/10)
   - [Main result - depends on all above lemmas]
   - Sketch: sketcher/sketch_20251025_0245_proof_09_kl_convergence.md
   - Proof: proofs/proof_20251025_0600_thm_kl_convergence_euclidean.md
   - Review: reviewer/review_20251025_0615_proof_thm_kl_convergence.md
   - Status: ✓ Auto-integrated (line 567)
   - Time: 3h 45min (1 attempt)

**Document status**: ✅ Modified and validated
**Backup**: 09_kl_convergence.md.backup_20251024_190500
**Total proofs integrated**: 4
**Total lines added**: 2847

[Repeat for each document...]

---

## Integration Summary

### Documents Modified (Auto-Integration)

| Document | Theorems | Lines Added | Backup |
|----------|----------|-------------|--------|
| 09_kl_convergence.md | 4 | 2847 | .backup_20251024_190500 |
| 03_cloning.md | 2 | 1523 | .backup_20251025_0700 |
| 04_wasserstein_contraction.md | 3 | 1892 | .backup_20251025_1200 |

**Total**: 5 documents, 9 proofs, 7234 lines

### Proofs for Manual Review

The following proofs meet publication standards (rigor ≥ 8/10) but are recommended
for manual review before integration:

1. **lemma-fisher-info** (Score: 8.5/10)
   - Document: 09_kl_convergence.md (line 187)
   - Proof: proofs/proof_20251024_1645_lemma_fisher_info.md
   - Review: reviewer/review_20251024_1700_proof_lemma_fisher_info.md
   - Reason: Minor notation inconsistency (uses both I[ρ] and ℱ(ρ))

2. **lemma-companion-selection** (Score: 8.3/10)
   - Document: 03_cloning.md (line 423)
   - Proof: proofs/proof_20251025_1030_lemma_companion_selection.md
   - Review: reviewer/review_20251025_1045_proof_lemma_companion.md
   - Reason: Edge case analysis could be more detailed

3. **thm-wasserstein-smoothing** (Score: 8.7/10)
   - Document: 04_wasserstein_contraction.md (line 289)
   - Proof: proofs/proof_20251025_1345_thm_wasserstein_smoothing.md
   - Review: reviewer/review_20251025_1400_proof_thm_wasserstein.md
   - Reason: Complex technical result, recommend verification

**Integration guide**: See `pipeline_manual_review_guide.md` for detailed instructions.

---

## Files Generated

### By Category

**Proof Sketches**: 15 files in `sketcher/`
**Complete Proofs**: 15 files in `proofs/`
**Reviews**: 18 files in `reviewer/` (15 proof reviews + 3 final document reviews)
**Backups**: 5 files (original documents before integration)
**State Files**: `pipeline_state.json`
**Reports**: This file

### Disk Usage

- Sketches: 1.2 MB
- Proofs: 8.7 MB
- Reviews: 3.4 MB
- Total: 13.3 MB

---

## Validation Results

### Cross-Reference Check

✅ **247 references validated**
- Valid: 245 (99.2%)
- Broken: 2 (0.8%)

See "Cross-Reference Validation" section above for broken reference details.

### LaTeX Formatting Check

✅ **All modified documents pass formatting validation**
- Math blocks: Properly formatted
- Blank lines: Correct
- Delimiters: Balanced

### Final Document Reviews

All auto-integrated documents passed final Math Reviewer validation:

1. ✅ 09_kl_convergence.md - No critical issues
2. ✅ 03_cloning.md - No critical issues
3. ✅ 04_wasserstein_contraction.md - 1 minor notation issue (acceptable)

---

## Dependency Resolution

### Dependencies Verified and Satisfied

The pipeline verified all dependencies against existing framework proofs before auto-resolving:

**✓ Satisfied Externally** (proofs already exist in framework):
- **lemma-wasserstein-contraction**: Found in 04_wasserstein_contraction.md (line 245)
- **lemma-fisher-info-basic**: Found in 09_kl_convergence.md (line 123)
- **thm-foster-lyapunov**: Found in 06_convergence.md (line 456)
- **lemma-qsd-existence**: Found in 02_euclidean_gas.md (line 789)

These dependencies were NOT re-proven (existing proofs used).

**✓ In Current Pipeline** (proven during this pipeline run):
- **lemma-fisher-info** (thm-kinetic-lsi depends on this)
- **lemma-entropy-bound** (thm-kl-convergence depends on this)

### Missing Lemmas Auto-Resolved

The pipeline automatically proved these **truly missing** dependencies (not found in glossary, reference, or previous documents):

1. **lemma-entropy-bound** (09_kl_convergence.md)
   - Verification: ⚠ Not found in glossary → Not found in reference → Not found in docs 01-08
   - Status: Truly missing
   - Required by: thm-kl-convergence-euclidean
   - Resolved in: 3h 10min
   - Score: 9.0/10
   - Now available for future use

2. **lemma-bounded-displacement** (04_wasserstein_contraction.md)
   - Verification: ⚠ Not found in framework
   - Status: Truly missing
   - Required by: thm-wasserstein-contraction
   - Resolved in: 2h 35min
   - Score: 9.1/10
   - Now available for future use

**Note**: The pipeline performed comprehensive verification (glossary.md, reference.md, all previous documents) before auto-resolving. This prevents duplicate work and ensures consistency with existing framework proofs.

### Dependency Graph

```
lemma-velocity-bound
    ├─→ lemma-bounded-displacement
    │       └─→ thm-wasserstein-contraction
    └─→ lemma-momentum-conservation
            └─→ thm-keystone-principle

lemma-fisher-info
    └─→ thm-kinetic-lsi
            └─→ thm-kl-convergence-euclidean

lemma-entropy-bound
    └─→ thm-kl-convergence-euclidean

lemma-wasserstein-contraction
    └─→ thm-kl-convergence-euclidean
```

---

## Iteration History

### Quality Improvements Through Iteration

| Theorem | Attempt 1 Score | Attempt 2 Score | Attempt 3 Score | Final |
|---------|-----------------|-----------------|-----------------|-------|
| thm-wasserstein-contraction | 7.8 | 8.9 | - | 8.9 |
| lemma-companion-selection | 7.5 | 8.3 | - | 8.3 |
| All others | ≥8.0 | - | - | First attempt |

**2 theorems** required iteration (13% of total)
**Average improvement**: +1.1 points per iteration

---

## Performance Metrics

### Agent Efficiency

| Agent | Invocations | Avg Time | Success Rate |
|-------|-------------|----------|--------------|
| Proof Sketcher | 15 | 45 min | 100% |
| Theorem Prover | 17 (2 re-expansions) | 95 min | 100% |
| Math Reviewer | 18 (15 proofs + 3 docs) | 7.5 min | 100% |

### Pipeline Efficiency

- **Parallelization**: 4 independent theorem branches processed in parallel
- **Wasted work**: 0% (no failed proofs)
- **Average rigor on first attempt**: 8.7/10
- **Iteration rate**: 13% of theorems required re-expansion

---

## Recommendations

### For Manual Review Proofs

1. Review `pipeline_manual_review_guide.md` for integration instructions
2. Prioritize by score (highest first) for fastest integration
3. All proofs in this category are publication-ready; review is optional

### For Future Pipelines

Based on this run:

1. **Preprocessing**: Consider pre-solving common lemmas (velocity bounds, Fisher info)
   to speed up future pipelines
2. **Quality threshold**: Current threshold (8/10) is appropriate; 87% of proofs
   scored ≥9/10 on first attempt
3. **Time estimates**: Use 2.75 hours per theorem for planning (observed average)
4. **Parallelization**: Pipeline achieved 4x speedup through parallel processing;
   consider increasing parallelism for larger folders

---

## Next Steps

### Immediate Actions

1. ✅ Pipeline completed successfully - no immediate action required

2. **Optional**: Integrate manual-review proofs
   - Follow guide in `pipeline_manual_review_guide.md`
   - Estimated time: 30 minutes per proof

3. **Optional**: Address 2 broken cross-references
   - See validation report above for details
   - Estimated time: 10 minutes

### Build Documentation

To build the updated documentation with new proofs:

```bash
make build-docs
```

### Commit Changes (Optional)

The pipeline has modified 5 source documents. To commit:

```bash
git add docs/source/1_euclidean_gas/
git commit -m "Add complete proofs via math pipeline

- Proved 15 theorems/lemmas
- Auto-integrated 9 high-confidence proofs (rigor ≥9/10)
- 3 proofs ready for manual review
- Average rigor score: 8.9/10

Generated by autonomous math pipeline
Pipeline ID: pipeline_20251024_153000
Duration: 41h 15min"
```

---

## Backup and Recovery

### Restore Original Documents

If you need to restore any document to its pre-pipeline state:

```bash
# Restore specific document
cp docs/source/1_euclidean_gas/09_kl_convergence.md.backup_20251024_190500 \
   docs/source/1_euclidean_gas/09_kl_convergence.md

# Restore all modified documents
for backup in docs/source/1_euclidean_gas/*.backup_*; do
    original="${backup%.backup_*}"
    cp "$backup" "$original"
done
```

### Rerun Pipeline

To rerun the pipeline (will skip completed theorems):

```bash
/math_pipeline docs/source/1_euclidean_gas
```

To start fresh (delete state file first):

```bash
rm docs/source/1_euclidean_gas/pipeline_state.json
/math_pipeline docs/source/1_euclidean_gas
```

---

## Conclusion

✅ **Mission accomplished!**

The autonomous math pipeline successfully processed the entire `1_euclidean_gas` chapter,
developing complete, publication-ready proofs for all 15 theorems and lemmas. All proofs
meet or exceed the Annals of Mathematics standard, with an average rigor score of 8.9/10.

**Total autonomous runtime**: 41 hours 15 minutes
**Human intervention required**: 0 times
**Success rate**: 100%

The pipeline demonstrated robust dependency resolution, quality iteration, and safe
integration. All modified documents have been validated and backed up.

**The Euclidean Gas chapter is now mathematically complete and publication-ready.**

---

**Generated by**: Autonomous Math Pipeline v1.0
**Pipeline ID**: pipeline_20251024_153000
**Report timestamp**: 2025-10-26 08:45:00
**State file**: pipeline_state.json
```

### Step 4.2: Update State File to Completed

```json
{
  "status": "completed",
  "completion_timestamp": "2025-10-26T08:45:00",
  "total_elapsed_minutes": 2475,
  "report_file": "pipeline_report_20251024_153000.md"
}
```

### Step 4.3: Final Output to User

```markdown
===================================================================
✅ AUTONOMOUS MATH PIPELINE COMPLETED SUCCESSFULLY
===================================================================

**Target**: docs/source/1_euclidean_gas (Folder Mode)
**Duration**: 41 hours 15 minutes
**Start**: 2025-10-24 15:30:00
**End**: 2025-10-26 08:45:00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Results Summary

✅ **15/15 theorems completed** (100% success rate)
✅ **9 proofs auto-integrated** (rigor ≥ 9/10)
⚠ **3 proofs ready for manual review** (8 ≤ rigor < 9)
✅ **0 failed proofs**
✅ **4 missing lemmas auto-resolved**

**Average rigor score**: 8.9/10
**Average time per theorem**: 2.75 hours

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Documents Modified

✅ 09_kl_convergence.md (4 proofs, 2847 lines added)
✅ 03_cloning.md (2 proofs, 1523 lines added)
✅ 04_wasserstein_contraction.md (3 proofs, 1892 lines added)

**Total**: 5 documents, 9 proofs, 7234 lines

All modifications validated ✓
All backups created ✓
All cross-references verified ✓ (2 minor issues flagged)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Files Generated

📁 **sketcher/** - 15 proof strategy sketches (1.2 MB)
📁 **proofs/** - 15 complete rigorous proofs (8.7 MB)
📁 **reviewer/** - 18 validation reviews (3.4 MB)
📋 **pipeline_report_20251024_153000.md** - Full detailed report
📋 **pipeline_manual_review_guide.md** - Integration guide for 3 manual-review proofs
💾 **pipeline_state.json** - Complete pipeline state

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Next Steps

1. **Read the full report**: pipeline_report_20251024_153000.md

2. **(Optional) Integrate manual-review proofs**:
   - See: pipeline_manual_review_guide.md
   - 3 proofs with rigor scores 8.3-8.7/10
   - All publication-ready, review is optional

3. **(Optional) Fix 2 broken cross-references**:
   - Details in report under "Cross-Reference Validation"

4. **Build documentation**:
   make build-docs

5. **Commit changes** (if satisfied):
   git add docs/source/1_euclidean_gas/
   git commit -m "Add complete proofs via math pipeline (15 theorems, avg rigor 8.9/10)"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Restoration (if needed)

To restore any document to pre-pipeline state:

cp {document}.backup_20251024_190500 {document}

All original documents backed up safely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ **The Euclidean Gas chapter is now mathematically complete and publication-ready.**

Thank you for using the Autonomous Math Pipeline!
```

---

## Error Handling and Resume Capability

### Graceful Error Handling

The pipeline includes comprehensive error handling at each stage:

#### Agent Failures

**Symptom**: Agent doesn't complete, returns error, or produces invalid output

**Recovery**:
1. Log error to state file:
   ```json
   {
     "errors": [
       {
         "timestamp": "2025-10-24T18:30:00",
         "stage": "theorem_prover",
         "theorem": "thm-kl-convergence",
         "error": "Agent timed out after 4 hours",
         "recovery_action": "retry_once"
       }
     ]
   }
   ```

2. **Retry once** (different model variant if available)

3. If second failure: Mark theorem as "failed", increment attempts counter

4. Continue to next theorem (don't abort pipeline)

5. Report failure in final summary

#### Missing File Errors

**Symptom**: Expected sketch/proof/review file not found

**Recovery**:
1. Check if agent actually completed (parse output)
2. Search for file with similar timestamp
3. If found: Use it
4. If not found: Retry agent invocation
5. If retry fails: Mark as failed and continue

#### Integration Errors

**Symptom**: Edit tool fails, document becomes corrupted, references break

**Recovery**:
1. **Immediate restoration**:
   ```bash
   cp {document}.backup_20251024_190500 {document}
   ```

2. Log error with details

3. Mark theorem as "integration_failed" (proof is still valid in proofs/ folder)

4. Continue to next theorem

5. Report in manual review section

#### Circular Dependency Detection

**Symptom**: Theorem A depends on Lemma B, which depends on Theorem A

**Recovery**:
1. Detect cycle during topological sort

2. Log error:
   ```json
   {
     "error": "circular_dependency",
     "cycle": ["thm-A", "lemma-B", "thm-A"],
     "resolution": "excluded_from_pipeline"
   }
   ```

3. Exclude entire cycle from execution plan

4. Report in final summary as "Requires manual resolution"

5. Continue with remaining theorems

### Resume Capability

The pipeline can be interrupted at any time and resumed from the last checkpoint.

#### Interruption Scenarios

1. **User interrupts** (Ctrl+C, timeout, context limit)
2. **System failure** (crash, network loss, etc.)
3. **Deliberate pause** (user wants to check progress)

#### Resume Protocol

When `/math_pipeline` is invoked:

1. **Check for existing state file**:
   ```bash
   if [ -f "pipeline_state.json" ]; then
       # State file exists - possible resume
   fi
   ```

2. **Read state file**:
   ```json
   {
     "status": "interrupted",
     "current_theorem": "thm-kl-convergence",
     "current_stage": "expanding",
     "statistics": {
       "completed": 8,
       "in_progress": 1,
       "pending": 6
     }
   }
   ```

3. **Determine resume point**:
   - If status = "in_progress" and current_theorem is set:
     - Check current_stage
     - Resume from that stage
   - If status = "interrupted":
     - Show progress summary
     - Ask user: Resume or Start Fresh?

4. **Resume logic by stage**:

   **Stage: sketching**
   - Check if sketch file exists
   - If yes: Skip to expanding stage
   - If no: Restart sketching

   **Stage: expanding**
   - Check if proof file exists
   - If yes: Skip to reviewing stage
   - If no: Restart expanding

   **Stage: reviewing**
   - Check if review file exists
   - If yes: Check rigor score and proceed to quality iteration or completion
   - If no: Restart reviewing

   **Stage: integrating**
   - Check if backup exists
   - Check if integration was completed
   - If completed: Skip
   - If not: Restart integration

5. **Continue from next theorem**:
   - Update state: current_theorem = next_theorem_in_plan
   - Proceed with normal loop

#### Resume Example

```markdown
===================================================================
RESUMING INTERRUPTED PIPELINE
===================================================================

**State file found**: pipeline_state.json
**Last update**: 2025-10-24 18:45:00 (2 hours ago)
**Status**: interrupted during "expanding" stage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Progress So Far

✅ **8/15 theorems completed**
⏸ **1/15 in progress** (thm-kl-convergence - expanding)
⏳ **6/15 pending**

Completed theorems:
1. ✅ lemma-bounded-displacement (Score: 9.1/10, integrated)
2. ✅ lemma-fisher-info (Score: 8.5/10, manual review)
3. ✅ lemma-wasserstein-contraction (Score: 9.2/10, integrated)
... [list continues]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Resume Plan

**Current theorem**: thm-kl-convergence-euclidean
**Current stage**: expanding (Theorem Prover running)
**Elapsed time**: 1h 30min of 3-4h estimated

Checking for partial output...
✓ Sketch completed: sketcher/sketch_20251024_1545_proof_09_kl_convergence.md
⚠ Proof expansion in progress (no output file yet)

**Resume action**: Restart expansion stage for thm-kl-convergence-euclidean
(Previous expansion progress lost, but sketch preserved)

**Estimated time to completion**: 33 hours

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Resuming pipeline in 3 seconds...

[Restarting Theorem Prover for thm-kl-convergence-euclidean...]
```

### State File Corruption Recovery

**Symptom**: `pipeline_state.json` is corrupted or invalid JSON

**Recovery**:
1. Attempt to parse JSON
2. If parsing fails:
   - Backup corrupted file: `pipeline_state.json.corrupted_20251024`
   - Search for most recent valid backup: `pipeline_state.json.backup_*`
   - If backup found: Use it
   - If no backup: Scan output directories (sketcher/, proofs/, reviewer/) to reconstruct state
3. Rebuild state file from discovered artifacts
4. Resume from reconstructed state

---

## Advanced Features

### Parallel Processing

For independent theorems (no dependencies), the pipeline can process multiple theorems simultaneously:

1. **Detect independence**:
   - During topological sort, identify theorems with no mutual dependencies
   - Group into "parallel batches"

2. **Launch agents in parallel**:
   ```python
   # Launch 3 Proof Sketcher agents simultaneously
   Task(description="Sketch thm-A", ...)
   Task(description="Sketch thm-B", ...)
   Task(description="Sketch thm-C", ...)

   # Wait for all to complete before proceeding
   ```

3. **Parallelism limits**:
   - Maximum 4 concurrent theorems (to avoid overwhelming agents)
   - Prioritize by estimated complexity (simpler theorems first)

4. **Progress tracking**:
   - State file tracks each theorem's status independently
   - Report shows parallel processing in action

### Incremental Integration

For very large folders, the pipeline can integrate proofs incrementally rather than waiting for all theorems to complete:

1. **After each theorem completion**:
   - If rigor ≥ 9/10: Immediately integrate into document
   - Update state file
   - Continue to next theorem

2. **Benefits**:
   - Reduces risk of data loss
   - User can see progress in real-time
   - Documents become progressively more complete

3. **Trade-off**:
   - More frequent file I/O
   - Slightly slower than batch integration
   - But much safer for long-running pipelines

### Smart Iteration

When a proof scores below threshold, the pipeline uses intelligent focus to address specific gaps:

1. **Parse Math Reviewer feedback**:
   - Extract CRITICAL and MAJOR issues
   - Categorize by type:
     - Incomplete epsilon-delta (location + specific limit)
     - Missing measure theory justification (operation + condition)
     - Unjustified constant (symbol + required formula)
     - Unhandled edge case (scenario + required analysis)

2. **Construct targeted focus prompt**:
   ```markdown
   Focus on:
   - Step 4, Substep 4.2, line 245: Add complete epsilon-delta for lim_{N→∞} ...
     Current gap: Limit stated but not proven with explicit δ(ε) construction
     Required: For all ε>0, find δ(ε) such that |N-N₀|<δ ⇒ |f(N)-L|<ε

   - Step 5, line 320: Verify Fubini condition 2 (integrability of |f|)
     Current gap: Fubini applied without verifying integrability
     Required: Show ∫∫|f(x,y)| dμ dν < ∞

   - All steps: Track constant C_Fisher with explicit formula
     Current gap: Appears as O(1) in multiple places
     Required: Explicit formula C_Fisher = ... in terms of framework parameters
   ```

3. **Agent receives precise instructions**:
   - Knows exactly which lines to fix
   - Knows what type of argument is needed
   - Can preserve other parts of proof unchanged

4. **Convergence**:
   - Typically 1-2 iterations sufficient
   - Rarely requires all 3 allowed attempts

---

## Usage Examples

### Example 1: Process Single Document

```bash
/math_pipeline docs/source/1_euclidean_gas/09_kl_convergence.md
```

**Output**:
```markdown
===================================================================
AUTONOMOUS MATH PIPELINE - SINGLE DOCUMENT MODE
===================================================================

Target: docs/source/1_euclidean_gas/09_kl_convergence.md
Mode: Single document processing

[Analyzing document...]
Document size: 2847 lines
Theorems/lemmas found: 5
  - lemma-fisher-info (line 187): needs proof
  - lemma-wasserstein-contraction (line 245): needs proof
  - lemma-entropy-bound (line 315): needs proof
  - thm-kinetic-lsi (line 456): needs proof
  - thm-kl-convergence-euclidean (line 678): needs proof

[Building dependency graph...]
Dependencies detected:
  thm-kinetic-lsi depends on: lemma-fisher-info
  thm-kl-convergence depends on: lemma-fisher-info, lemma-wasserstein-contraction, lemma-entropy-bound

[Creating execution plan...]
Execution order:
  1. lemma-fisher-info
  2. lemma-wasserstein-contraction
  3. lemma-entropy-bound
  4. thm-kinetic-lsi
  5. thm-kl-convergence-euclidean

Estimated total time: 14 hours

[Starting pipeline...]
```

### Example 2: Process Entire Folder

```bash
/math_pipeline docs/source/1_euclidean_gas
```

**Output**: (See Phase 0 execution plan example above)

### Example 3: Resume Interrupted Pipeline

```bash
/math_pipeline docs/source/1_euclidean_gas
```

**Output**: (See Resume Example above)

### Example 4: Process Subfolder

```bash
/math_pipeline docs/source/1_euclidean_gas/10_kl_convergence
```

If the path is a folder (even a subfolder), processes all documents in that folder.

---

## Configuration Options

The command behavior can be customized by modifying the configuration section at the top of the command file:

```markdown
## Pipeline Configuration

- **Integration Strategy**: Hybrid (modify to "auto_all" or "manual_all" if desired)
- **Dependency Handling**: Auto-resolve (modify to "skip" if you want manual control)
- **Quality Standards**: Auto-iterate (modify iteration limit if needed)
- **Parallelism**: 4 concurrent theorems (adjust based on system resources)
```

---

## Important Notes

### Safety and Validation

1. **Backups are always created** before any document is modified
2. **All integrations are validated** by Math Reviewer
3. **Cross-references are verified** after integration
4. **State is saved after each theorem** (can resume at any time)
5. **Errors never abort the pipeline** (continue to next theorem)

### Time and Resources

1. **This pipeline is designed to run for hours or days** without human intervention
2. **Each theorem takes 2-4 hours on average** (sketching, expansion, review, iteration)
3. **Agents run in separate contexts** (no token concerns in main pipeline)
4. **State file enables resume** after interruptions
5. **Progress reports every 30 minutes** show current status

### Quality Standards

1. **Minimum threshold**: 8/10 (Annals of Mathematics standard)
2. **Auto-integration threshold**: 9/10 (high confidence)
3. **Maximum iterations**: 3 attempts per theorem
4. **All proofs validated** by Math Reviewer before integration

### Limitations

1. **Cannot handle theorems outside the Fragile framework** (agents won't have context)
2. **Circular dependencies are excluded** (require manual resolution)
3. **Very complex theorems** may hit agent limits (mark for manual proof)
4. **Notation inconsistencies** across documents may cause issues (should be uniform)

---

## Troubleshooting

### Pipeline seems stuck

**Solution**: Check `pipeline_state.json` for current theorem and stage. Agents can run for 2-4 hours per theorem.

### State file corrupted

**Solution**: Pipeline auto-recovers by scanning output directories. See "State File Corruption Recovery" above.

### Integration failed

**Solution**: Restore from backup (automatically created). Proof is still valid in proofs/ folder for manual integration.

### Agent failed multiple times

**Solution**: Theorem marked as "failed" in state, pipeline continues. Can retry manually after pipeline completes.

### Wrong rigor score

**Solution**: Review the review file manually. Math Reviewer assessment is usually accurate, but you can override by manually integrating proofs.

### Want to change quality threshold

**Solution**: Modify "Quality Standards" configuration at top of command file. Default 8/10 is recommended.

---

## Related Commands

- **/dual_review**: Review existing proofs interactively
- **/clean_ai**: Remove AI meta-commentary after integration
- **/introduction**: Generate introduction sections for documents

## Related Agents

- **.claude/agents/proof-sketcher.md**: Sketch individual theorem proofs
- **.claude/agents/theorem-prover.md**: Expand individual proof sketches
- **.claude/agents/math-reviewer.md**: Review individual proofs or documents

---

**Note**: This is the autonomous pipeline. For manual control of individual theorems, use the agents directly via the Task tool.

---

**End of Math Pipeline Command**
