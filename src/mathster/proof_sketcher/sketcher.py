#!/usr/bin/env python3
"""Structured proof sketch models plus DSPy signatures aligned with sketch.json."""

from __future__ import annotations

import json
from typing import Callable, Literal

import dspy
from pydantic import BaseModel, Field

from mathster.proof_sketcher.sketch_strategist import (
    SketchStrategy,
    SketchStrategist,
    get_label_data,
)
from mathster.claude_tool import sync_ask_claude

__all__ = [
    "ProofType",
    "ProofSketchStatus",
    "DependencyType",
    "LemmaDifficulty",
    "FrameworkVerificationState",
    "CircularReasoningState",
    "KeyAssumptionsState",
    "CrossValidationState",
    "ProofStatement",
    "StrategyOption",
    "VerificationStatus",
    "RecommendedApproach",
    "StrategySynthesis",
    "DependencyEntry",
    "LemmaToProve",
    "UncertainAssumption",
    "MissingOrUncertainDependencies",
    "DependencyLedger",
    "ProofStep",
    "DetailedProof",
    "TechnicalDeepDive",
    "ValidationChecklist",
    "AlternativeApproach",
    "FutureWork",
    "ExpansionTask",
    "ExpansionPhase",
    "ExpansionRoadmap",
    "CrossReferences",
    "ProofSketch",
    "ProofStatementSignature",
    "ProofStatementAgent",
    "StrategySynthesisSignature",
    "StrategySynthesisAgent",
    "DependencyLedgerSignature",
    "DetailedProofSignature",
    "DetailedProofAgent",
    "TechnicalDeepDivesSignature",
    "TechnicalDeepDiveAgent",
    "ValidationChecklistSignature",
    "ValidationChecklistAgent",
    "AlternativeApproachesSignature",
    "AlternativeApproachesAgent",
    "FutureWorkSignature",
    "FutureWorkAgent",
    "ExpansionRoadmapSignature",
    "ExpansionRoadmapAgent",
    "CrossReferencesSignature",
    "CrossReferencesAgent",
    "ProofSketchSignature",
    "ProofSketchAgent",
    "DependencyPlanningSignature",
    "DependencyLedgerAgentSignature",
    "StrategyComparisonSignature",
    "StrategySynthesizer",
    "DependencyLedgerAgent",
    "configure_dependency_plan_tool",
    "setup_missing_dependency_tool",
    "setup_verified_dependency_tool",
]


ProofType = Literal["Theorem", "Proposition", "Corollary", "Lemma"]
ProofSketchStatus = Literal["Sketch", "Draft", "Ready for Expansion", "Completed"]
DependencyType = Literal["Axiom", "Theorem", "Lemma", "Definition", "Constant"]
LemmaDifficulty = Literal["Easy", "Medium", "Hard"]
FrameworkVerificationState = Literal["Verified", "Partially Verified", "Needs Verification"]
CircularReasoningState = Literal["No circularity detected", "Potential circularity to check"]
KeyAssumptionsState = Literal["All assumptions are standard", "Requires new, unproven assumption"]
CrossValidationState = Literal[
    "Consensus between strategists", "Discrepancies noted", "Single strategist only"
]


class ProofStatement(BaseModel):
    """Plain-language and formal versions of the target result."""

    formal: str = Field(..., description="Verbatim formal statement (LaTeX/Markdown allowed).")
    informal: str = Field(
        ..., description="Intuitive explanation of the result's meaning and impact."
    )


class StrategyOption(BaseModel):
    """Candidate proof strategy with strengths/weaknesses."""

    strategist: str = Field(..., description="Origin of the idea (agent/model/human).")
    method: str = Field(..., description="Concise technique label.")
    keySteps: list[str] = Field(
        default_factory=list, description="Ordered milestones required for the approach."
    )
    strengths: list[str] = Field(
        default_factory=list, description="Advantages for selecting this strategy."
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Risks or drawbacks that need mitigation."
    )


class VerificationStatus(BaseModel):
    """Status audit for the recommended strategy."""

    frameworkDependencies: FrameworkVerificationState = Field(
        ..., description="Are cited dependencies confirmed within the framework?"
    )
    circularReasoning: CircularReasoningState = Field(
        ..., description="Assessment of circularity risks."
    )
    keyAssumptions: KeyAssumptionsState = Field(
        ..., description="Do we rely only on established assumptions?"
    )
    crossValidation: CrossValidationState = Field(
        ..., description="Were multiple strategists consulted?"
    )


class RecommendedApproach(BaseModel):
    """Chosen approach plus rationale and verification state."""

    chosenMethod: str = Field(..., description="Final technique selection.")
    rationale: str = Field(
        ..., description="Why this approach is preferred over other candidates."
    )
    verificationStatus: VerificationStatus = Field(
        ..., description="Self-audit of dependency and logic soundness."
    )


class StrategySynthesis(BaseModel):
    """Complete strategy evaluation block."""

    strategies: list[StrategyOption] = Field(
        ..., description="All candidate strategies that were evaluated."
    )
    recommendedApproach: RecommendedApproach = Field(
        ..., description="Final approach plus justification."
    )


class DependencyEntry(BaseModel):
    """Single verified dependency used inside the proof."""

    type: DependencyType = Field(..., description="Directive category for this dependency.")
    label: str = Field(..., description="Framework label (e.g., thm-main-convergence).")
    sourceDocument: str = Field(..., description="Document path or identifier.")
    purpose: str = Field(..., description="What this dependency provides to the proof.")
    usedInSteps: list[str] = Field(
        ..., description="Proof step identifiers where this dependency is consumed."
    )


class LemmaToProve(BaseModel):
    """Placeholder lemma required to complete the proof."""

    name: str = Field(..., description="Temporary lemma name (e.g., 'Lemma A').")
    statement: str = Field(..., description="Mathematical statement to prove.")
    justification: str = Field(..., description="Why this lemma is necessary.")
    difficulty: LemmaDifficulty = Field(..., description="Subjective difficulty rating.")


class UncertainAssumption(BaseModel):
    """Assumption that still needs formal verification."""

    name: str | None = Field(
        default=None, description="Optional short name for cross-reference tracking."
    )
    statement: str = Field(..., description="Precise assumption being used.")
    justification: str = Field(
        ..., description="Why this assumption is viewed as uncertain or implicit."
    )
    resolutionPath: str = Field(
        ..., description="Plan to formalize or validate the assumption."
    )


class MissingOrUncertainDependencies(BaseModel):
    """Outstanding lemmas or assumptions that must be resolved."""

    lemmasToProve: list[LemmaToProve] = Field(
        default_factory=list, description="List of lemmas that still need proofs."
    )
    uncertainAssumptions: list[UncertainAssumption] = Field(
        default_factory=list,
        description="Assumptions whose validity still needs justification.",
    )


class DependencyLedger(BaseModel):
    """Combined accounting of proven and pending dependencies."""

    verifiedDependencies: list[DependencyEntry] = Field(
        default_factory=list, description="All references currently justified within the framework."
    )
    missingOrUncertainDependencies: MissingOrUncertainDependencies | None = Field(
        default=None, description="Work remaining to firm up the dependency graph."
    )


class ProofStep(BaseModel):
    """Single structured step appearing inside detailedProof.steps."""

    stepNumber: int = Field(..., description="1-based identifier for ordering.")
    title: str = Field(..., description="Short heading for this step.")
    goal: str = Field(..., description="Objective for the step.")
    action: str = Field(..., description="Mathematical operations that will be executed.")
    justification: str = Field(..., description="Why the action is valid.")
    expectedResult: str = Field(..., description="Result once the step is completed.")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Labels of dependencies invoked inside this step.",
    )
    potentialIssues: str | None = Field(
        default=None, description="Risks, gaps, or caveats specific to this step."
    )


class DetailedProof(BaseModel):
    """Narrative plus structured steps for the full proof skeleton."""

    overview: str = Field(..., description="High-level description of the proof flow.")
    topLevelOutline: list[str] = Field(
        ..., description="Ordered list summarizing the proof's major phases."
    )
    steps: list[ProofStep] = Field(..., description="Detailed proof steps with metadata.")
    conclusion: str = Field(..., description="Final statement (Q.E.D.).")


class TechnicalDeepDive(BaseModel):
    """Detailed analysis of a challenging sub-problem."""

    challengeTitle: str = Field(..., description="Short name for the technical challenge.")
    difficultyDescription: str = Field(..., description="Why this part is delicate.")
    proposedSolution: str = Field(..., description="Outline of the technique to resolve it.")
    mathematicalDetail: str | None = Field(
        default=None, description="Key calculations or rigor checkpoints."
    )
    references: list[str] = Field(
        default_factory=list,
        description="Supporting framework labels or literature citations.",
    )


class ValidationChecklist(BaseModel):
    """Self-audit of proof completeness and rigor."""

    logicalCompleteness: bool = Field(False, description="All steps logically connected?")
    hypothesisUsage: bool = Field(False, description="All hypotheses invoked appropriately?")
    conclusionDerivation: bool = Field(False, description="Conclusion follows rigorously?")
    frameworkConsistency: bool = Field(False, description="Consistent with framework axioms?")
    noCircularReasoning: bool = Field(False, description="No hidden circular logic?")
    constantTracking: bool = Field(False, description="Universal constants tracked correctly?")
    edgeCases: bool = Field(False, description="Edge cases addressed or flagged?")
    regularityAssumptions: bool = Field(False, description="Regularity needs explicitly stated?")


class AlternativeApproach(BaseModel):
    """Documented but unselected proof strategy."""

    name: str = Field(..., description="Identifier for the alternative approach.")
    approach: str = Field(..., description="Summary of the method/idea.")
    pros: list[str] = Field(default_factory=list, description="Advantages.")
    cons: list[str] = Field(default_factory=list, description="Disadvantages.")
    whenToConsider: str = Field(..., description="Conditions under which this approach is useful.")


class FutureWork(BaseModel):
    """Outstanding directions related to the sketch."""

    remainingGaps: list[str] = Field(default_factory=list, description="Known gaps to close.")
    conjectures: list[str] = Field(default_factory=list, description="Related conjectures.")
    extensions: list[str] = Field(default_factory=list, description="Potential extensions.")


class ExpansionTask(BaseModel):
    """Single task appearing in the expansion roadmap."""

    taskName: str = Field(..., description="Label for the work item.")
    strategy: str | None = Field(
        default=None, description="Proposed strategy for executing this task."
    )
    difficulty: str | None = Field(
        default=None, description="Subjective difficulty indicator or notes."
    )


class ExpansionPhase(BaseModel):
    """Phase of the proof expansion plan."""

    phaseTitle: str = Field(..., description="Name of this roadmap phase.")
    estimatedTime: str = Field(..., description="Estimated time budget for the phase.")
    tasks: list[ExpansionTask] = Field(..., description="Tasks required within this phase.")


class ExpansionRoadmap(BaseModel):
    """Project plan for converting the sketch into a publication-ready proof."""

    phases: list[ExpansionPhase] = Field(..., description="Ordered phases covering the expansion.")
    totalEstimatedTime: str = Field(..., description="Sum of the expected time for completion.")


class CrossReferences(BaseModel):
    """Links to other artifacts in the mathematical ecosystem."""

    theoremsUsed: list[str] = Field(default_factory=list, description="Referenced theorems.")
    definitionsUsed: list[str] = Field(default_factory=list, description="Referenced definitions.")
    axiomsUsed: list[str] = Field(default_factory=list, description="Referenced axioms.")
    relatedProofs: list[str] = Field(default_factory=list, description="Related proof labels.")
    downstreamConsequences: list[str] = Field(
        default_factory=list, description="Results depending on the sketch."
    )


class ProofSketch(BaseModel):
    """Top-level schema mirroring src/mathster/proof_sketcher/sketch.json."""

    title: str = Field(..., description="Short descriptive theorem title.")
    label: str = Field(..., description="Framework label of the statement.")
    type: ProofType = Field(..., description="Directive type classification.")
    source: str = Field(..., description="Source file path or document pointer.")
    date: str = Field(..., description="Creation date in YYYY-MM-DD.")
    status: ProofSketchStatus = Field(..., description="Current lifecycle state.")
    statement: ProofStatement = Field(..., description="Formal + informal statements.")
    strategySynthesis: StrategySynthesis = Field(
        ..., description="Evaluated strategies plus chosen approach."
    )
    dependencies: DependencyLedger = Field(
        ..., description="Verified dependencies plus outstanding work."
    )
    detailedProof: DetailedProof = Field(..., description="Structured proof outline.")
    validationChecklist: ValidationChecklist = Field(
        ..., description="Self-audit of proof completeness."
    )
    technicalDeepDives: list[TechnicalDeepDive] = Field(
        default_factory=list, description="Optional list of technical challenge analyses."
    )
    alternativeApproaches: list[AlternativeApproach] = Field(
        default_factory=list, description="Other viable strategies that were recorded."
    )
    futureWork: FutureWork | None = Field(
        default=None, description="Related gaps, conjectures, or extensions."
    )
    expansionRoadmap: ExpansionRoadmap | None = Field(
        default=None, description="Plan for promoting the sketch to a full proof."
    )
    crossReferences: CrossReferences | None = Field(
        default=None, description="Pointers to other framework assets."
    )
    specialNotes: str | None = Field(
        default=None, description="Meta-commentary, reviewer notes, or version info."
    )


class ProofStatementSignature(dspy.Signature):
    """Extract structured formal/informal statements for the target result."""

    theorem_label = dspy.InputField(desc="Framework label (e.g., thm-main-convergence).")
    theorem_statement = dspy.InputField(
        desc="Raw theorem/proposition statement as written in the source."
    )
    context_notes = dspy.InputField(
        desc="Optional background context or operator instructions.", optional=True
    )

    statement: ProofStatement = dspy.OutputField(
        desc="Structured ProofStatement containing formal/informal descriptions."
    )


class ProofStatementAgent(dspy.Module):
    """Chain-of-thought agent that emits ProofStatement objects."""

    def __init__(self) -> None:
        schema = json.dumps(ProofStatement.model_json_schema(), indent=2)
        instructions = f"""
Produce both the formal and informal statements for the provided theorem.

Guidelines:
- Use theorem_label/theorem_statement/context to restate the result precisely.
- The formal statement should be publication-ready; preserve hypotheses.
- The informal statement should explain intuition, significance, or geometric meaning.

Return ONLY JSON matching:
{schema}
"""
        signature = ProofStatementSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        theorem_label: str,
        theorem_statement: str,
        context_notes: str = "",
    ) -> dspy.Prediction:
        return self.agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            context_notes=context_notes or "",
        )


class StrategySynthesisSignature(dspy.Signature):
    """Generate the complete strategy comparison block."""

    theorem_label = dspy.InputField(desc="Framework label of the target statement.")
    theorem_statement = dspy.InputField(desc="Formal statement with relevant hypotheses.")
    strategy_notes = dspy.InputField(
        desc="Optional notes on candidate strategies or prior attempts.", optional=True
    )

    strategySynthesis: StrategySynthesis = dspy.OutputField(
        desc="Evaluated strategies plus the recommended approach."
    )


class StrategySynthesisAgent(dspy.Module):
    """Chain-of-thought agent that generates StrategySynthesis from scratch."""

    def __init__(self) -> None:
        schema = json.dumps(StrategySynthesis.model_json_schema(), indent=2)
        instructions = f"""
You explore multiple proof strategies for the given theorem.

Steps:
1. Brainstorm ≥2 feasible methods leveraging known framework tools.
2. Summarize each strategy (strategist/method/keySteps/strengths/weaknesses).
3. Select a recommended approach with rationale and verificationStatus:
   - frameworkDependencies: degree of established support
   - circularReasoning: flag if suspicious dependencies
   - keyAssumptions: whether the approach introduces new assumptions
   - crossValidation: consensus between strategists?

Return ONLY JSON matching:
{schema}
"""
        signature = StrategySynthesisSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        theorem_label: str,
        theorem_statement: str,
        strategy_notes: str = "",
    ) -> dspy.Prediction:
        return self.agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            strategy_notes=strategy_notes or "",
        )


class DependencyLedgerSignature(dspy.Signature):
    """Summarize verified and missing dependencies from a proof sketch narrative."""

    proof_sketch_text = dspy.InputField(
        desc="Narrative sketch or step outline referencing framework results."
    )
    registry_context = dspy.InputField(
        desc="Optional JSON or text describing available registry entries.", optional=True
    )

    dependencies: DependencyLedger = dspy.OutputField(
        desc="Structured ledger of verified dependencies and outstanding work."
    )


class DetailedProofSignature(dspy.Signature):
    """Build the structured detailedProof object from outlines and notes."""

    theorem_statement = dspy.InputField(desc="Formal theorem/proposition statement.")
    strategy_summary = dspy.InputField(
        desc="Summary of the preferred strategy or top-level outline."
    )
    dependency_notes = dspy.InputField(
        desc="Key dependencies or lemmas each step must reference.", optional=True
    )

    detailedProof: DetailedProof = dspy.OutputField(
        desc="Structured DetailedProof with overview, outline, steps, and conclusion."
    )


class DetailedProofAgent(dspy.Module):
    """Chain-of-thought agent that emits DetailedProof objects."""

    def __init__(self) -> None:
        schema = json.dumps(DetailedProof.model_json_schema(), indent=2)
        instructions = f"""
You produce the DetailedProof section of the proof sketch.

Input fields:
- theorem_statement: formal claim with hypotheses.
- strategy_summary: textual summary of the selected proof strategy.
- dependency_notes: optional text describing key lemmas or references.

Reasoning steps:
1. Derive a narrative overview referencing main mechanisms.
2. Create a topLevelOutline (3-7 bullet items) covering each proof phase.
3. For each outline item, craft a structured ProofStep with fields:
   * stepNumber (1-based, consistent order)
   * title (concise)
   * goal (what must be established)
   * action (mathematical actions/operations)
   * justification (why valid, referencing lemmas/axioms)
   * expectedResult (state the derived fact)
   * dependencies (list of labels or plan references)
   * potentialIssues (optional risk notes)
4. Finish with a conclusion string tying the steps back to the theorem.

Output ONLY JSON matching:
{schema}
"""
        signature = DetailedProofSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        theorem_statement: str,
        strategy_summary: str,
        dependency_notes: str = "",
    ) -> dspy.Prediction:
        return self.agent(
            theorem_statement=theorem_statement,
            strategy_summary=strategy_summary,
            dependency_notes=dependency_notes or "",
        )


class TechnicalDeepDivesSignature(dspy.Signature):
    """Extract technical deep dives from the proof sketch."""

    proof_sketch_text = dspy.InputField(
        desc="Full proof sketch text or section that mentions technical challenges."
    )
    focus_areas = dspy.InputField(
        desc="Optional hints specifying which challenges to emphasize.", optional=True
    )

    deep_dives: list[TechnicalDeepDive] = dspy.OutputField(
        desc="List of TechnicalDeepDive objects documenting the hardest sub-problems."
    )


class TechnicalDeepDiveAgent(dspy.Module):
    """Chain-of-thought agent that fills TechnicalDeepDiveSignature."""

    def __init__(self) -> None:
        instructions = f"""
You extract ALL meaningful technical challenges from a proof sketch.

Process:
1. Read the proof sketch carefully and identify distinct obstacles (regularity, uniform bounds, coupling, etc.).
2. For each challenge provide:
   - challengeTitle (concise)
   - difficultyDescription (why hard, with quantitative/qualitative detail)
   - proposedSolution (tools/lemmas/strategy to resolve)
   - mathematicalDetail (optional but preferred: key estimates, inequalities, or plan)
   - references (framework labels or literature)
3. Prioritize most critical obstacles first. 3-5 entries typical.

Return ONLY a JSON array conforming to:
{json.dumps(TechnicalDeepDive.model_json_schema(), indent=2)}
"""
        signature = TechnicalDeepDivesSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        proof_sketch_text: str,
        focus_areas: str = "",
    ) -> dspy.Prediction:
        return self.agent(
            proof_sketch_text=proof_sketch_text,
            focus_areas=focus_areas or "",
        )


class ValidationChecklistSignature(dspy.Signature):
    """Assess completion of the validation checklist."""

    proof_sketch_text = dspy.InputField(desc="Full proof sketch for self-audit.")
    self_review_notes = dspy.InputField(
        desc="Optional reviewer feedback guiding the checklist.", optional=True
    )

    checklist: ValidationChecklist = dspy.OutputField(
        desc="Boolean checklist confirming rigor/completeness checkpoints."
    )


class ValidationChecklistAgent(dspy.Module):
    """Predictor that fills ValidationChecklist booleans."""

    def __init__(self) -> None:
        schema = json.dumps(ValidationChecklist.model_json_schema(), indent=2)
        instructions = f"""
Review the provided proof sketch text (and optional reviewer notes) and evaluate
each checklist item (True/False). Set True only if evidence clearly supports it.

Return ONLY JSON matching:
{schema}
"""
        signature = ValidationChecklistSignature.with_instructions(instructions)
        self.agent = dspy.Predict(signature)

    def forward(
        self,
        proof_sketch_text: str,
        self_review_notes: str = "",
    ) -> dspy.Prediction:
        return self.agent(
            proof_sketch_text=proof_sketch_text,
            self_review_notes=self_review_notes or "",
        )


class AlternativeApproachesSignature(dspy.Signature):
    """Summarize alternative strategies that were considered but not selected."""

    theorem_statement = dspy.InputField(desc="Formal theorem statement.")
    rejected_ideas = dspy.InputField(
        desc="Notes about strategies that were considered but ultimately rejected."
    )

    alternatives: list[AlternativeApproach] = dspy.OutputField(
        desc="Structured list of alternate approaches with pros/cons."
    )


class AlternativeApproachesAgent(dspy.Module):
    """Chain-of-thought agent that records alternative strategies."""

    def __init__(self) -> None:
        schema = json.dumps(AlternativeApproach.model_json_schema(), indent=2)
        instructions = f"""
From the theorem statement and rejected_ideas text, extract 2-4 alternative approaches.
Each entry must include name, approach summary, pros, cons, and whenToConsider.

Return ONLY a JSON array of objects matching:
{schema}
"""
        signature = AlternativeApproachesSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        theorem_statement: str,
        rejected_ideas: str,
    ) -> dspy.Prediction:
        return self.agent(
            theorem_statement=theorem_statement,
            rejected_ideas=rejected_ideas,
        )


class FutureWorkSignature(dspy.Signature):
    """Capture remaining gaps, conjectures, and extensions."""

    open_questions = dspy.InputField(
        desc="Narrative description of gaps, conjectures, or follow-up directions."
    )

    futureWork: FutureWork = dspy.OutputField(
        desc="Structured FutureWork object derived from the open questions."
    )


class FutureWorkAgent(dspy.Module):
    """Chain-of-thought agent that structures future work directions."""

    def __init__(self) -> None:
        schema = json.dumps(FutureWork.model_json_schema(), indent=2)
        instructions = f"""
Analyze the provided open questions/gaps and categorize them into remainingGaps,
conjectures, and extensions. Provide concrete, concise bullet-style text.

Return ONLY JSON matching:
{schema}
"""
        signature = FutureWorkSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(self, open_questions: str) -> dspy.Prediction:
        return self.agent(open_questions=open_questions)


class ExpansionRoadmapSignature(dspy.Signature):
    """Plan the expansion roadmap for formalizing the proof sketch."""

    workstream_notes = dspy.InputField(
        desc="Project management notes or TODO list for finishing the proof."
    )
    constraints = dspy.InputField(
        desc="Optional constraints such as timeline, staffing, or sequencing requirements.",
        optional=True,
    )

    roadmap: ExpansionRoadmap = dspy.OutputField(
        desc="ExpansionRoadmap object with phases, tasks, and total timeline."
    )


class ExpansionRoadmapAgent(dspy.Module):
    """Chain-of-thought agent that drafts the expansion roadmap."""

    def __init__(self) -> None:
        schema = json.dumps(ExpansionRoadmap.model_json_schema(), indent=2)
        instructions = f"""
Convert workstream notes (plus optional constraints) into a phased roadmap.
Each phase should include tasks with taskName (and optional strategy/difficulty).
Set totalEstimatedTime consistent with phase estimates.

Return ONLY JSON matching:
{schema}
"""
        signature = ExpansionRoadmapSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        workstream_notes: str,
        constraints: str = "",
    ) -> dspy.Prediction:
        return self.agent(
            workstream_notes=workstream_notes,
            constraints=constraints or "",
        )


class CrossReferencesSignature(dspy.Signature):
    """Extract cross-reference metadata from the sketch."""

    proof_sketch_text = dspy.InputField(
        desc="Proof sketch prose referencing other theorems/definitions/etc."
    )

    crossReferences: CrossReferences = dspy.OutputField(
        desc="Structured lists of referenced theorems, definitions, axioms, and downstream links."
    )


class CrossReferencesAgent(dspy.Module):
    """Chain-of-thought agent that records cross references."""

    def __init__(self) -> None:
        schema = json.dumps(CrossReferences.model_json_schema(), indent=2)
        instructions = f"""
From the provided proof sketch text, list referenced framework artifacts.
Populate each array with unique labels (strings). Use downstreamConsequences to
describe results enabled by the theorem.

Return ONLY JSON matching:
{schema}
"""
        signature = CrossReferencesSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(self, proof_sketch_text: str) -> dspy.Prediction:
        return self.agent(proof_sketch_text=proof_sketch_text)


class ProofSketchSignature(dspy.Signature):
    """Produce a complete ProofSketch object aligned with sketch.json."""

    title_hint = dspy.InputField(desc="Concise human-readable theorem title.")
    theorem_label = dspy.InputField(desc="Framework label (e.g., thm-main-convergence).")
    theorem_type = dspy.InputField(
        desc='Type of statement ("Theorem", "Proposition", "Corollary", "Lemma").'
    )
    theorem_statement = dspy.InputField(desc="Formal statement, including hypotheses.")
    document_source = dspy.InputField(desc="File path or reference to the source text.")
    creation_date = dspy.InputField(desc="Date string YYYY-MM-DD when the sketch is created.")
    proof_status = dspy.InputField(
        desc='Lifecycle state (Sketch, Draft, Ready for Expansion, Completed).'
    )
    framework_context = dspy.InputField(
        desc="Optional background information or available results.", optional=True
    )
    operator_notes = dspy.InputField(
        desc="Optional evaluator instructions, constraints, or review goals.", optional=True
    )

    sketch: ProofSketch = dspy.OutputField(
        desc="Complete proof sketch object ready to validate against sketch.json."
    )


class ProofSketchAgent(dspy.Module):
    """End-to-end orchestrator that assembles ProofSketch via modular agents."""

    def __init__(self) -> None:
        super().__init__()
        self.statement_agent = ProofStatementAgent()
        self.sketch_strategist_primary = SketchStrategist()
        self.sketch_strategist_secondary = SketchStrategist()
        self.strategy_synthesizer = StrategySynthesizer()
        self.dependency_agent = DependencyLedgerAgent()
        self.detailed_proof_agent = DetailedProofAgent()
        self.deep_dive_agent = TechnicalDeepDiveAgent()
        self.validation_agent = ValidationChecklistAgent()
        self.alternative_agent = AlternativeApproachesAgent()
        self.future_work_agent = FutureWorkAgent()
        self.roadmap_agent = ExpansionRoadmapAgent()
        self.cross_refs_agent = CrossReferencesAgent()

    def forward(
        self,
        title_hint: str,
        theorem_label: str,
        theorem_type: str,
        theorem_statement: str,
        document_source: str,
        creation_date: str,
        proof_status: str,
        framework_context: str = "",
        operator_notes: str = "",
    ) -> dspy.Prediction:
        framework_context = framework_context or ""
        operator_notes = operator_notes or ""

        # 1. Statement
        statement = self.statement_agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            context_notes=framework_context,
        ).statement

        # 2. Generate two strategies via SketchStrategist with different guidance
        primary_notes = (
            operator_notes + " | Primary strategist: Claude Sonnet 4.5 focusing on geometric insight."
        ).strip()
        secondary_notes = (
            operator_notes + " | Secondary strategist: GPT-5 Codex focusing on analytic bounds."
        ).strip()

        primary_strategy = self.sketch_strategist_primary(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            framework_context=framework_context,
            operator_notes=primary_notes,
        ).strategy

        secondary_strategy = self.sketch_strategist_secondary(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            framework_context=framework_context,
            operator_notes=secondary_notes,
        ).strategy

        # 3. Strategy synthesis
        strategy_synthesis = self.strategy_synthesizer(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            primary_strategy=primary_strategy,
            secondary_strategy=secondary_strategy,
            evaluation_notes=operator_notes,
        ).strategySynthesis

        strategy_context_text = _strategy_context(strategy_synthesis)

        # 4. Dependencies
        combined_notes = "\n\n".join(
            [
                theorem_statement,
                _describe_strategy(primary_strategy),
                _describe_strategy(secondary_strategy),
                strategy_context_text,
            ]
        )

        dependency_ledger = self.dependency_agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            strategy_synthesis=strategy_synthesis,
            sketch_notes=combined_notes,
        ).dependency_ledger

        dependency_notes = json.dumps(dependency_ledger.model_dump(), indent=2)

        # 5. Detailed proof
        option = _select_recommended_option(strategy_synthesis)
        detailed_proof = self.detailed_proof_agent(
            theorem_statement=theorem_statement,
            strategy_summary=strategy_context_text,
            dependency_notes=dependency_notes,
        ).detailedProof

        proof_text = "\n\n".join(
            [
                statement.formal,
                statement.informal,
                strategy_context_text,
                detailed_proof.overview,
                "\n".join(step.action for step in detailed_proof.steps),
            ]
        )

        # 6. Technical deep dives
        technical_deep_dives = self.deep_dive_agent(
            proof_sketch_text=proof_text,
            focus_areas="Extract hardest steps, dependencies, and regularity issues.",
        ).deep_dives

        # 7. Validation checklist
        validation_checklist = self.validation_agent(
            proof_sketch_text=proof_text,
            self_review_notes=operator_notes,
        ).checklist

        # 8. Alternate approaches (based on non-selected strategy)
        selected_method = strategy_synthesis.recommendedApproach.chosenMethod.strip()
        other_strategy = (
            secondary_strategy if secondary_strategy.method.strip() == selected_method else primary_strategy
        )
        rejected_ideas = _describe_strategy(other_strategy)
        alternative_approaches = self.alternative_agent(
            theorem_statement=theorem_statement,
            rejected_ideas=rejected_ideas,
        ).alternatives

        # 9. Future work from dependency gaps
        missing_text = _format_missing_dependencies_text(dependency_ledger)
        future_work = self.future_work_agent(open_questions=missing_text).futureWork

        # 10. Expansion roadmap
        roadmap_notes = _roadmap_notes_from_future_work(future_work)
        expansion_roadmap = self.roadmap_agent(
            workstream_notes=roadmap_notes,
            constraints=operator_notes,
        ).roadmap

        # 11. Cross references
        cross_references = self.cross_refs_agent(proof_sketch_text=proof_text).crossReferences

        # Assemble final ProofSketch
        proof_sketch = ProofSketch(
            title=title_hint,
            label=theorem_label,
            type=theorem_type,
            source=document_source,
            date=creation_date,
            status=proof_status,
            statement=statement,
            strategySynthesis=strategy_synthesis,
            dependencies=dependency_ledger,
            detailedProof=detailed_proof,
            validationChecklist=validation_checklist,
            technicalDeepDives=technical_deep_dives,
            alternativeApproaches=alternative_approaches,
            futureWork=future_work,
            expansionRoadmap=expansion_roadmap,
            crossReferences=cross_references,
            specialNotes=(
                "Strategies generated via dual SketchStrategist passes (Claude + GPT) "
                f"with operator guidance: {operator_notes or 'n/a'}"
            ),
        )

        return dspy.Prediction(sketch=proof_sketch)


class DependencyPlanningSignature(dspy.Signature):
    """Plan dependency auditing steps before assembling the ledger."""

    theorem_label = dspy.InputField(desc="Framework label of the target statement.")
    theorem_statement = dspy.InputField(desc="Formal theorem/proposition statement.")
    strategy_summary = dspy.InputField(
        desc="Narrative summary of the selected proof strategy."
    )
    rationale = dspy.InputField(desc="Justification for the chosen method.")
    key_steps_text = dspy.InputField(
        desc="Ordered bullet list of key steps (one per line)."
    )

    plan: str = dspy.OutputField(
        desc="Numbered plan describing how to confirm verified dependencies and identify missing ones."
    )


class StrategyComparisonSignature(dspy.Signature):
    """Chain-of-thought signature for comparing two sketch strategies."""

    theorem_label = dspy.InputField(desc="Framework label of the target statement.")
    theorem_statement = dspy.InputField(
        desc="Formal statement (with hypotheses) for which the strategies were produced."
    )
    primary_strategy_json = dspy.InputField(
        desc="JSON string representation of the first SketchStrategy."
    )
    secondary_strategy_json = dspy.InputField(
        desc="JSON string representation of the second SketchStrategy."
    )
    evaluation_notes = dspy.InputField(
        desc="Optional reviewer guidance or evaluation criteria.", optional=True
    )

    strategySynthesis: StrategySynthesis = dspy.OutputField(
        desc="Structured StrategySynthesis object selecting the stronger approach."
    )


class StrategySynthesizer(dspy.Module):
    """Use a ChainOfThought agent to synthesize StrategySynthesis from two strategies."""

    def __init__(self) -> None:
        instructions = f"""
You are comparing two independently generated proof strategies for the same theorem.

Input:
- theorem_label/theorem_statement identify the target result.
- primary_strategy_json / secondary_strategy_json are JSON objects that follow sketch_strategy.json.
- evaluation_notes provide optional reviewer emphasis (e.g., prefer LSI tools, avoid circular reasoning).

Required reasoning steps:
1. Parse both strategy JSON objects.
2. Evaluate each along the framework criteria:
   - keySteps coverage and coherence
   - strengths vs weaknesses (technical feasibility)
   - availability of frameworkDependencies (are cited results sufficient?)
   - severity of technicalDeepDives and confidenceScore
3. Decide which strategy is superior or note if both converge to the same plan.
4. Populate StrategySynthesis:
   - strategies: two StrategyOption entries mirroring the original strategist/method/keySteps/strengths/weaknesses.
   - recommendedApproach: choose the best method and justify it in 2-3 sentences referencing concrete factors.
   - verificationStatus:
       * frameworkDependencies ∈ {list(FrameworkVerificationState.__args__)}
       * circularReasoning ∈ {list(CircularReasoningState.__args__)}
       * keyAssumptions ∈ {list(KeyAssumptionsState.__args__)}
       * crossValidation ∈ {list(CrossValidationState.__args__)}
     Base these fields on the comparative analysis (e.g., if both strategies agree, mark consensus).

Output format:
Return ONLY a JSON object that matches StrategySynthesis model schema:
{json.dumps(StrategySynthesis.model_json_schema(), indent=2)}
"""
        signature = StrategyComparisonSignature.with_instructions(instructions)
        self.agent = dspy.ChainOfThought(signature)

    def forward(
        self,
        theorem_label: str,
        theorem_statement: str,
        primary_strategy: "SketchStrategy",
        secondary_strategy: "SketchStrategy",
        evaluation_notes: str = "",
    ) -> dspy.Prediction:
        primary_json = primary_strategy.model_dump_json()
        secondary_json = secondary_strategy.model_dump_json()
        return self.agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            primary_strategy_json=primary_json,
            secondary_strategy_json=secondary_json,
            evaluation_notes=evaluation_notes or "",
        )


def _load_strategy_synthesis(strategy_json: str) -> StrategySynthesis:
    if not strategy_json:
        raise ValueError("Strategy synthesis JSON must be provided.")
    try:
        return StrategySynthesis.model_validate_json(strategy_json)
    except Exception as exc:  # noqa: BLE001 - surface underlying validation
        raise ValueError("Invalid strategy_synthesis_json supplied to tool.") from exc


def _select_recommended_option(strategy: StrategySynthesis) -> StrategyOption:
    if not strategy.strategies:
        raise ValueError("StrategySynthesis must contain at least one strategy option.")
    chosen = strategy.recommendedApproach.chosenMethod.strip()
    for option in strategy.strategies:
        if option.method.strip() == chosen:
            return option
    return strategy.strategies[0]


def _format_key_steps(option: StrategyOption) -> str:
    if not option.keySteps:
        return "No key steps were provided."
    return "\n".join(f"{idx}. {step}" for idx, step in enumerate(option.keySteps, start=1))


def _strategy_summary(option: StrategyOption, rationale: str) -> str:
    strengths = ", ".join(option.strengths) if option.strengths else "Not specified"
    weaknesses = ", ".join(option.weaknesses) if option.weaknesses else "Not specified"
    return (
        f"Method: {option.method}\n"
        f"Strengths: {strengths}\n"
        f"Weaknesses: {weaknesses}\n"
        f"Rationale: {rationale}"
    )


def _strategy_context(strategy: StrategySynthesis) -> str:
    option = _select_recommended_option(strategy)
    context_lines = [
        f"Chosen method: {strategy.recommendedApproach.chosenMethod}",
        f"Rationale: {strategy.recommendedApproach.rationale}",
        "Key steps:",
    ]
    context_lines.extend(f"- {step}" for step in option.keySteps)
    if option.strengths:
        context_lines.append(f"Strengths: {', '.join(option.strengths)}")
    if option.weaknesses:
        context_lines.append(f"Weaknesses: {', '.join(option.weaknesses)}")
    return "\n".join(context_lines)


def _describe_strategy(strategy: SketchStrategy) -> str:
    steps = "\n".join(f"- {step}" for step in strategy.keySteps) if strategy.keySteps else "None"
    strengths = ", ".join(strategy.strengths) if strategy.strengths else "None"
    weaknesses = ", ".join(strategy.weaknesses) if strategy.weaknesses else "None"
    deps = strategy.frameworkDependencies
    dep_count = len(deps.theorems) + len(deps.lemmas) + len(deps.axioms) + len(deps.definitions)
    return (
        f"Strategist: {strategy.strategist}\n"
        f"Method: {strategy.method}\n"
        f"Summary: {strategy.summary}\n"
        f"Key Steps:\n{steps}\n"
        f"Strengths: {strengths}\n"
        f"Weaknesses: {weaknesses}\n"
        f"Dependencies referenced: {dep_count}\n"
        f"Confidence: {strategy.confidenceScore}"
    )


def _format_missing_dependencies_text(ledger: DependencyLedger) -> str:
    missing = ledger.missingOrUncertainDependencies
    if missing is None:
        return "All dependencies currently verified; no missing lemmas recorded."
    parts: list[str] = []
    if missing.lemmasToProve:
        parts.append("Lemmas To Prove:")
        for lemma in missing.lemmasToProve:
            parts.append(
                f"- {lemma.name} ({lemma.difficulty}): {lemma.statement}. Justification: {lemma.justification}"
            )
    if missing.uncertainAssumptions:
        parts.append("Uncertain Assumptions:")
        for assumption in missing.uncertainAssumptions:
            parts.append(
                f"- {assumption.name or 'Unnamed'}: {assumption.statement}. Justification: {assumption.justification}. "
                f"Resolution: {assumption.resolutionPath}"
            )
    return "\n".join(parts) if parts else "No missing dependencies were identified."


def _roadmap_notes_from_future_work(future_work: FutureWork | None) -> str:
    if future_work is None:
        return "No future work currently recorded."
    parts = []
    if future_work.remainingGaps:
        parts.append("Remaining Gaps:\n" + "\n".join(f"- {gap}" for gap in future_work.remainingGaps))
    if future_work.conjectures:
        parts.append("Conjectures:\n" + "\n".join(f"- {conj}" for conj in future_work.conjectures))
    if future_work.extensions:
        parts.append("Extensions:\n" + "\n".join(f"- {ext}" for ext in future_work.extensions))
    return "\n\n".join(parts) if parts else "Future work list currently empty."


def configure_dependency_plan_tool() -> Callable[[str, str, str], str]:
    """Create Chain-of-Thought tool that outlines dependency auditing steps."""

    instructions = """
You are planning how to extract a DependencyLedger for a proof sketch.
Build a SHORT numbered plan (3-6 steps) that:
1. Maps the chosen strategy's key steps to expected framework dependencies.
2. Notes which registries or documents must be consulted.
3. Identifies where new lemmas/assumptions may be required.
Mention concrete labels or step identifiers when possible.
"""
    signature = DependencyPlanningSignature.with_instructions(instructions)
    agent = dspy.ChainOfThought(signature)

    def run_plan(theorem_label: str, theorem_statement: str, strategy_synthesis_json: str) -> str:
        strategy = _load_strategy_synthesis(strategy_synthesis_json)
        option = _select_recommended_option(strategy)
        plan_result = agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            strategy_summary=_strategy_summary(option, strategy.recommendedApproach.rationale),
            rationale=strategy.recommendedApproach.rationale,
            key_steps_text=_format_key_steps(option),
        )
        return plan_result.plan if hasattr(plan_result, "plan") else str(plan_result)

    return run_plan


def setup_verified_dependency_tool() -> Callable[[str, str, str, str, str], str]:
    """Create a tool that queries Claude for verified dependency entries."""

    schema = json.dumps(DependencyEntry.model_json_schema(), indent=2)
    claude_instructions = f"""
You enumerate VERIFIED framework dependencies for a proof strategy.
Return a JSON array where each item matches this schema:
{schema}

Guidelines:
- Use actual framework labels (thm-*, lem-*, def-*, ax-*)
- Specify sourceDocument paths when known (e.g., docs/source/1_euclidean_gas/...).
- usedInSteps should reference the plan or key steps (e.g., "Plan Step 2", "Key Step 3").
- Only include dependencies that already exist in the framework.
"""

    def gather(
        theorem_label: str,
        theorem_statement: str,
        strategy_synthesis_json: str,
        plan_text: str,
        sketch_notes: str = "",
    ) -> str:
        strategy = _load_strategy_synthesis(strategy_synthesis_json)
        context = _strategy_context(strategy)
        prompt = f"""
Target theorem: {theorem_label}
Statement:
{theorem_statement}

Strategy context (selected approach):
{context}

Dependency plan:
{plan_text}

Additional notes:
{sketch_notes or 'None'}

List VERIFIED dependencies as JSON array.
"""
        raw = sync_ask_claude(prompt, model="sonnet", system_prompt=claude_instructions)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # noqa: B904
            raise ValueError(f"Claude returned invalid dependency JSON: {raw}") from exc
        if not isinstance(data, list):
            raise ValueError("Claude dependency response must be a JSON array.")
        validated = [DependencyEntry.model_validate(item).model_dump() for item in data]
        return json.dumps(validated, indent=2)

    return gather


def setup_missing_dependency_tool() -> Callable[[str, str, str, str, str], str]:
    """Create a tool that identifies missing lemmas or uncertain assumptions."""

    schema = json.dumps(MissingOrUncertainDependencies.model_json_schema(), indent=2)
    claude_instructions = f"""
You identify missing lemmas or uncertain assumptions required by a proof strategy.
Return a JSON object matching:
{schema}

Guidelines:
- lemmasToProve: list each needed lemma with name, statement, justification, difficulty.
- uncertainAssumptions: list implicit assumptions plus resolution paths.
- Use strings that reference plan/key steps for context (e.g., "Needed for Step 4 bootstrap").
"""

    def identify(
        theorem_label: str,
        theorem_statement: str,
        strategy_synthesis_json: str,
        plan_text: str,
        sketch_notes: str = "",
    ) -> str:
        strategy = _load_strategy_synthesis(strategy_synthesis_json)
        context = _strategy_context(strategy)
        prompt = f"""
Target theorem: {theorem_label}
Statement:
{theorem_statement}

Strategy context:
{context}

Dependency plan:
{plan_text}

Additional notes:
{sketch_notes or 'None'}

List NEW lemmas or uncertain assumptions still needed.
"""
        raw = sync_ask_claude(prompt, model="sonnet", system_prompt=claude_instructions)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # noqa: B904
            raise ValueError(f"Claude returned invalid missing-dependency JSON: {raw}") from exc
        validated = MissingOrUncertainDependencies.model_validate(data)
        return validated.model_dump_json(indent=2)

    return identify


class DependencyLedgerAgentSignature(dspy.Signature):
    """Signature for the DependencyLedger ReAct agent."""

    theorem_label = dspy.InputField(desc="Framework label of the target statement.")
    theorem_statement = dspy.InputField(desc="Formal theorem/proposition statement.")
    strategy_synthesis_json = dspy.InputField(
        desc="StrategySynthesis JSON produced by StrategySynthesizer."
    )
    sketch_notes = dspy.InputField(
        desc="Optional partial proof sketch text or reviewer notes.", optional=True
    )

    dependency_ledger: DependencyLedger = dspy.OutputField(
        desc="Structured ledger containing verified and missing dependencies."
    )


class DependencyLedgerAgent(dspy.Module):
    """ReAct agent that uses planning + Claude tools to build a DependencyLedger."""

    def __init__(self) -> None:
        super().__init__()
        self._plan_impl = configure_dependency_plan_tool()
        self._verified_impl = setup_verified_dependency_tool()
        self._missing_impl = setup_missing_dependency_tool()

        def plan_dependencies_tool(
            theorem_label: str, theorem_statement: str, strategy_synthesis_json: str
        ) -> str:
            """Plan numbered steps to audit dependencies before extraction."""
            print("running plan_dependencies_tool")
            return self._plan_impl(theorem_label, theorem_statement, strategy_synthesis_json)

        def gather_verified_dependencies_tool(
            theorem_label: str,
            theorem_statement: str,
            strategy_synthesis_json: str,
            plan_text: str,
            sketch_notes: str,
        ) -> str:
            """Ask Claude to list VERIFIED framework dependencies cited by the plan."""
            print("running gather_verified_dependencies_tool")
            return self._verified_impl(
                theorem_label, theorem_statement, strategy_synthesis_json, plan_text, sketch_notes
            )

        def identify_missing_dependencies_tool(
            theorem_label: str,
            theorem_statement: str,
            strategy_synthesis_json: str,
            plan_text: str,
            sketch_notes: str,
        ) -> str:
            """Ask Claude to enumerate lemmas-to-prove and uncertain assumptions still needed."""
            print("running identify_missing_dependencies_tool")
            return self._missing_impl(
                theorem_label, theorem_statement, strategy_synthesis_json, plan_text, sketch_notes
            )

        def lookup_label_tool(label: str) -> str:
            """Fetch existing registry data for a framework label (definition, theorem, etc.)."""
            print("running lookup_label_tool")
            return get_label_data(label)

        instructions = f"""
You are constructing the DependencyLedger AFTER the strategy has been selected but BEFORE full proof expansion.
Follow this workflow:
1. Call plan_dependencies_tool(...) to create a numbered plan linking steps to expected dependencies.
2. Use gather_verified_dependencies_tool(...) (multiple times if needed) to enumerate confirmed dependencies.
3. Call lookup_label_tool(label) whenever you must inspect an existing registry entry.
4. Call identify_missing_dependencies_tool(...) to capture lemmas-to-prove and uncertain assumptions.
5. Populate DependencyLedger with:
   - verifiedDependencies: entries returned from the verified dependency tool.
   - missingOrUncertainDependencies: JSON from the missing-dependency tool (ensure empty arrays if nothing is missing).

Always reference specific plan/key steps in usedInSteps fields.
Return ONLY JSON matching:
{json.dumps(DependencyLedger.model_json_schema(), indent=2)}
"""
        signature = DependencyLedgerAgentSignature.with_instructions(instructions)
        self.agent = dspy.ReAct(
            signature,
            tools=[
                plan_dependencies_tool,
                gather_verified_dependencies_tool,
                identify_missing_dependencies_tool,
                lookup_label_tool,
            ],
            max_iters=4,
        )

    def forward(
        self,
        theorem_label: str,
        theorem_statement: str,
        strategy_synthesis: StrategySynthesis | str,
        sketch_notes: str = "",
    ) -> dspy.Prediction:
        strategy_json = (
            strategy_synthesis
            if isinstance(strategy_synthesis, str)
            else strategy_synthesis.model_dump_json()
        )
        return self.agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            strategy_synthesis_json=strategy_json,
            sketch_notes=sketch_notes or "",
        )
