#!/usr/bin/env python3
"""DSPy signature and helper models for schema-driven sketch generation."""

from __future__ import annotations

import json
import logging
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from mathster.claude_tool import sync_ask_claude

logger = logging.getLogger(__name__)


__all__ = [
    "ConfidenceScore",
    "FrameworkDependencies",
    "FrameworkDependenciesSignature",
    "SketchStrategist",
    "SketchStrategy",
    "SketchStrategyItems",
    "SketchStrategySignature",
    "StrategyDependency",
    "TechnicalDeepDive",
    "TechnicalDeepDiveSignature",
    "configure_get_strategy_items",
    "setup_get_framework_dependencies",
    "setup_get_technical_deep_dive",
]


ConfidenceScore = Literal["High", "Medium", "Low"]


class StrategyDependency(BaseModel):
    """Single dependency entry inside the frameworkDependencies block."""

    label: str = Field(
        ...,
        description="Unique identifier (e.g., thm-main-convergence, def-capped-volume).",
    )
    document: str = Field(
        ...,
        description="Path or document title where this dependency is defined.",
    )
    purpose: str = Field(
        ...,
        description="Brief explanation of what this dependency contributes to the proof strategy.",
    )
    usedInSteps: list[str] = Field(
        default_factory=list,
        description="Optional list of keySteps identifiers or short names that rely on this dependency.",
    )


class FrameworkDependencies(BaseModel):
    """Grouped dependency lists aligned with sketch_strategy.json."""

    theorems: list[StrategyDependency] = Field(
        default_factory=list,
        description="Theorems required to execute the strategy.",
    )
    lemmas: list[StrategyDependency] = Field(
        default_factory=list,
        description="Lemmas required to execute the strategy.",
    )
    axioms: list[StrategyDependency] = Field(
        default_factory=list,
        description="Axioms/assumptions the strategy depends on.",
    )
    definitions: list[StrategyDependency] = Field(
        default_factory=list,
        description="Definitions needed for notation or concepts used in the proof.",
    )


class FrameworkDependenciesSignature(dspy.Signature):
    """Signature for extracting framework dependencies from proof sketches."""

    prompt: str = dspy.InputField(
        desc="Natural language proof sketch or mathematical discussion containing framework dependencies."
    )
    dependencies: FrameworkDependencies = dspy.OutputField(
        desc="Structured FrameworkDependencies object with categorized theorems, lemmas, axioms, definitions."
    )


class SketchStrategy(BaseModel):
    """Pydantic mirror of sketch_strategy.json for convenient validation."""

    strategist: str = Field(
        ...,
        description="Agent/model responsible for the proposal (e.g., GPT-5 Codex).",
    )
    method: str = Field(
        ...,
        description="Concise, high-level name for the proof technique.",
    )
    summary: str = Field(
        ...,
        description="Narrative overview explaining the approach and logical flow.",
    )
    keySteps: list[str] = Field(
        ...,
        description="Ordered list of major stages required to complete the proof.",
    )
    strengths: list[str] = Field(
        ...,
        description="Primary advantages of pursuing this strategy.",
    )
    weaknesses: list[str] = Field(
        ...,
        description="Limitations, risks, or pain points to investigate.",
    )
    frameworkDependencies: FrameworkDependencies = Field(
        ...,
        description="Structured list of supporting framework assets grouped by type.",
    )
    technicalDeepDives: list[TechnicalDeepDive] = Field(
        default_factory=list,
        description="Optional deep dives into the most demanding sub-problems.",
    )
    confidenceScore: ConfidenceScore = Field(
        ...,
        description='Self-assessed viability score ("High", "Medium", or "Low").',
    )


class SketchStrategySignature(dspy.Signature):
    """Collect all data required to emit a SketchStrategy object."""

    theorem_label = dspy.InputField(
        desc="Framework label of the target statement (e.g., thm-main-convergence)."
    )
    theorem_statement = dspy.InputField(
        desc="Formal statement (with hypotheses) for which the strategy is being produced."
    )
    framework_context = dspy.InputField(
        desc="Optional background: prior results, parameter regimes, or operator notes.",
        optional=True,
    )
    operator_notes = dspy.InputField(
        desc="Optional evaluator instructions (preferred tools, constraints, pitfalls).",
        optional=True,
    )

    strategy: SketchStrategy = dspy.OutputField(
        desc="JSON-serializable object conforming to sketch_strategy.json."
    )


class TechnicalDeepDive(BaseModel):
    """Detailed look at hard sub-problems in the proposed approach."""

    challengeTitle: str = Field(..., description="Short name describing the technical obstacle.")
    difficultyDescription: str = Field(
        ...,
        description="Explanation of why this challenge is difficult or delicate.",
    )
    proposedSolution: str = Field(
        ...,
        description="Sketch of the argument or tool that resolves the difficulty.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Optional citations to framework results or literature backing the solution.",
    )


class TechnicalDeepDiveSignature(dspy.Signature):
    """Signature for extracting technical challenges from proof sketches."""

    prompt: str = dspy.InputField(
        desc="Natural language proof sketch or mathematical discussion containing technical challenges."
    )
    deep_dives: list[TechnicalDeepDive] = dspy.OutputField(
        desc="List of TechnicalDeepDive objects identifying obstacles, difficulties, and proposed solutions."
    )


def setup_get_technical_deep_dive(lm: dspy.LM = None, claude_model="sonnet"):
    """Helper to create a tool that extracts all TechnicalDeepDive objects from a proof sketch.

    Returns:
        Callable that takes a prompt string and returns a list of TechnicalDeepDive objects.
        This function can be used as a tool in agentic workflows (e.g., DSPy ReAct).
    """
    CLAUDE_INSTRUCTIONS = f"""
You are a mathematical proof analysis expert. Your task is to extract ALL TechnicalDeepDive
objects from proof sketches or mathematical discussions.

A TechnicalDeepDive identifies ONE specific technical obstacle in a proof strategy and provides:
1. challengeTitle: A concise name for the obstacle (e.g., "Uniform LSI Constant", "Boundary Regularity")
2. difficultyDescription: Why this challenge is difficult or requires careful treatment
3. proposedSolution: A sketch of the mathematical argument or tool that resolves it
4. references: Optional list of framework labels (e.g., "thm-bakry-emery", "lem-holder-bound")
   or literature citations that support the solution

Guidelines:
- Identify ALL distinct technical challenges in the proof sketch (typically 2-5)
- Create one TechnicalDeepDive per challenge
- Be mathematically precise in difficulty descriptions
- Provide actionable solution sketches, not vague statements
- Reference specific framework results when available
- Order by importance (most critical first)
- Return a list of valid JSON objects matching the TechnicalDeepDive schema

Common types of challenges to identify:
- Uniform bounds across parameters (e.g., N-uniformity)
- Regularity/smoothness requirements
- Compactness/tightness arguments
- Convergence rate estimates
- Functional inequality constants (LSI, Poincaré)
- Coupling constructions
- Tensorization/factorization steps

Input will be a natural language description or proof sketch.
Output must be a list of structured TechnicalDeepDive objects following the schema exactly and nothing else:
{TechnicalDeepDive.model_json_schema()}
"""
    INSTRUCTIONS = "Use your ask_claude tool to extract detailed TechnicalDeepDive objects from the provided proof sketch or discussion."
    signature = TechnicalDeepDiveSignature.with_instructions(INSTRUCTIONS)

    def ask_claude(prompt: str) -> str:
        """Give claude detailed instructions to gather any information you need about the project."""
        return sync_ask_claude(prompt, model=claude_model, system_prompt=CLAUDE_INSTRUCTIONS)

    agent = dspy.ReAct(signature=signature, tools=[ask_claude])
    lm = lm or dspy.settings.lm

    def get_technical_deep_dives(prompt: str) -> list[TechnicalDeepDive]:
        """Extract all TechnicalDeepDive objects from a proof sketch or mathematical discussion.

        This function analyzes mathematical text to identify and structure ALL technical obstacles,
        their difficulties, proposed solutions, and supporting references.

        Args:
            prompt: Natural language description of a proof sketch, mathematical argument,
                   or discussion containing technical challenges to be extracted.

        Returns:
            List of TechnicalDeepDive objects, each with structured fields:
                - challengeTitle: Short descriptive name
                - difficultyDescription: Why the challenge is hard
                - proposedSolution: Sketch of resolution approach
                - references: Optional framework labels or citations

            Ordered by importance (most critical challenges first).

        Example:
            >>> prompt = '''
            ... The proof has several technical challenges:
            ... 1. We need uniform LSI constants across N, which is delicate due to
            ...    cloning-induced correlations. We use tensorization (thm-lsi-tensorization)
            ...    with conditional independence (lem-cloning-conditional-independence).
            ... 2. Establishing boundary regularity requires careful Hölder estimates.
            ...    We apply the barrier method from lem-euclidean-boundary-holder.
            ... 3. The convergence rate must be uniform in the time step τ.
            ...    This follows from Grönwall's lemma with explicit constants.
            ... '''
            >>> deep_dives = get_technical_deep_dives(prompt)
            >>> len(deep_dives)
            3
            >>> print(deep_dives[0].challengeTitle)
            'Uniform LSI Constant Across N'
            >>> print(deep_dives[1].challengeTitle)
            'Boundary Regularity via Hölder Estimates'
        """
        with dspy.context(lm=lm):
            result = agent.run(prompt=prompt)
        return result.deep_dives if hasattr(result, "deep_dives") else result

    get_technical_deep_dives.agent = agent  # type: ignore[attr-defined]

    return get_technical_deep_dives


def setup_get_framework_dependencies(lm: dspy.LM = None, claude_model="sonnet"):
    """Helper to create a tool that extracts FrameworkDependencies from a proof sketch.

    Returns:
        Callable that takes a prompt string and returns a FrameworkDependencies object.
        This function can be used as a tool in agentic workflows (e.g., DSPy ReAct).
    """
    CLAUDE_INSTRUCTIONS = f"""
You are a mathematical proof dependency analyzer. Your task is to extract ALL framework dependencies
from proof sketches or mathematical discussions, organized by type.

A FrameworkDependencies object contains four lists:
1. theorems: List[StrategyDependency] - Major results required for the proof
2. lemmas: List[StrategyDependency] - Supporting technical results
3. axioms: List[StrategyDependency] - Fundamental assumptions or axioms
4. definitions: List[StrategyDependency] - Definitions needed for notation/concepts

Each StrategyDependency has:
- label: Unique identifier (e.g., "thm-main-convergence", "def-capped-volume", "lem-holder-bound")
- document: Path or document title where defined (e.g., "09_kl_convergence", "01_fragile_gas_framework")
- purpose: Brief explanation of what this contributes to the proof strategy
- usedInSteps: Optional list of proof step identifiers that use this dependency

Guidelines:
- Extract ALL dependencies mentioned in the proof sketch
- Categorize correctly:
  * theorems: Major results, main theorems (prefix: thm-)
  * lemmas: Technical lemmas, propositions, corollaries (prefix: lem-, prop-, cor-)
  * axioms: Fundamental assumptions, axioms (prefix: ax-, axiom-)
  * definitions: Definitions, notation (prefix: def-)
- Extract accurate labels matching framework convention (e.g., "thm-bakry-emery", "def-lsi-constant")
- Provide clear, concise purpose statements explaining the dependency's role
- Map dependencies to proof steps via usedInSteps when steps are mentioned
- If document is unknown, use descriptive title (e.g., "KL Convergence Theory")
- Order each list by importance/usage order

Common framework documents:
- 01_fragile_gas_framework: Core axioms and definitions
- 02_euclidean_gas: Euclidean Gas specification
- 03_cloning: Cloning operator theory
- 05_kinetic_contraction: Kinetic operator convergence
- 06_convergence.md: Uniform in N convergence in total variation

Input will be a natural language proof sketch or mathematical discussion.
Output must be a structured FrameworkDependencies object following the schema exactly and nothing else:
{FrameworkDependencies.model_json_schema()}
"""
    INSTRUCTIONS = "Use your ask_claude tool to extract all framework dependencies (theorems, lemmas, axioms, definitions) from the provided proof sketch."
    signature = FrameworkDependenciesSignature.with_instructions(INSTRUCTIONS)

    def ask_claude(prompt: str) -> str:
        """Give claude detailed instructions to gather framework dependency information."""
        return sync_ask_claude(prompt, model=claude_model, system_prompt=CLAUDE_INSTRUCTIONS)

    agent = dspy.ReAct(signature=signature, tools=[ask_claude])
    lm = lm or dspy.settings.lm

    def get_framework_dependencies(prompt: str) -> FrameworkDependencies:
        """Extract FrameworkDependencies from a proof sketch or mathematical discussion.

        This function analyzes mathematical text to identify and categorize ALL framework
        dependencies (theorems, lemmas, axioms, definitions) required for the proof.

        Args:
            prompt: Natural language description of a proof sketch, mathematical argument,
                   or discussion containing framework dependencies to be extracted.

        Returns:
            FrameworkDependencies object with four categorized lists:
                - theorems: Major results required
                - lemmas: Supporting technical results
                - axioms: Fundamental assumptions
                - definitions: Notation and concept definitions

            Each dependency includes label, document, purpose, and usedInSteps.

        Example:
            >>> prompt = '''
            ... The proof relies on several key results:
            ... 1. The main LSI theorem (thm-lsi-target from 09_kl_convergence) establishes
            ...    the log-Sobolev inequality with constant λ, used in steps 2-3.
            ... 2. We need the tensorization lemma (lem-tensorization) for independence.
            ... 3. The axiom of bounded displacement (ax-bounded-displacement from
            ...    01_fragile_gas_framework) ensures Lipschitz bounds.
            ... 4. We use the definition of relative entropy (def-relative-entropy).
            ... '''
            >>> deps = get_framework_dependencies(prompt)
            >>> len(deps.theorems)
            1
            >>> deps.theorems[0].label
            'thm-lsi-target'
            >>> len(deps.lemmas)
            1
            >>> len(deps.axioms)
            1
            >>> len(deps.definitions)
            1
        """
        with dspy.context(lm=lm):
            result = agent.run(prompt=prompt)
        return result.dependencies if hasattr(result, "dependencies") else result

    get_framework_dependencies.agent = agent  # type: ignore[attr-defined]

    return get_framework_dependencies


def configure_get_strategy_items(lm: dspy.LM = None):
    """Helper to create a tool that generates SketchStrategyItems using chain-of-thought reasoning.

    Returns:
        Callable that takes theorem context and returns a SketchStrategyItems object.
        Uses ChainOfThought reasoning to develop proof strategy items.
    """
    INSTRUCTIONS = f"""
You are a mathematical proof strategy expert. Generate a structured SketchStrategyItems object
that outlines a proof approach for the given theorem.

Think step-by-step about the proof structure before generating the strategy items.

SketchStrategyItems contains 6 components:
1. strategist: Agent/model responsible for the proposal (e.g., "Claude Sonnet 4.5", "GPT-5 Codex")
2. method: Concise, high-level name for the proof technique (e.g., "LSI + Grönwall Iteration", "Coupling + Tensorization")
3. summary: Narrative overview explaining the approach and logical flow (2-4 sentences)
4. keySteps: Ordered list of major stages required to complete the proof (typically 3-7 steps)
5. strengths: Primary advantages of pursuing this strategy (list of 2-5 items)
6. weaknesses: Limitations, risks, or pain points to investigate (list of 2-5 items)

Guidelines for chain-of-thought reasoning:
- First, analyze the theorem statement to identify what needs to be proven
- Consider the framework context and prior results available
- Think about operator notes for constraints or preferred approaches
- Reason about multiple possible proof strategies, then select the most promising
- For the method: Be concise yet descriptive (3-6 words max)
- For the summary: Explain the high-level strategy flow, not technical details
- For keySteps: Identify major milestones (not too granular, not too vague)
- For strengths: Focus on why this approach is viable/promising
- For weaknesses: Identify technical challenges that need careful treatment
- Consider trade-offs between different approaches

Common proof strategies in the framework:
- LSI-based convergence (Bakry-Émery, Grönwall iteration)
- Coupling arguments (Wasserstein contraction, tensorization)
- Hypocoercivity methods (entropy dissipation, spectral gaps)
- Mean-field limits (propagation of chaos, exchangeability)
- Functional inequalities (Poincaré, log-Sobolev, transport-entropy)

Output must be a structured SketchStrategy object following the schema exactly:
{SketchStrategy.model_json_schema()}
"""
    signature = dspy.Signature(
        "theorem_label: str, theorem_statement: str, framework_context: str, operator_notes: str -> strategy_items: SketchStrategy"
    ).with_instructions(INSTRUCTIONS)

    agent = dspy.ChainOfThought(signature)
    lm = lm or dspy.settings.lm

    def get_strategy_items(
        theorem_label: str,
        theorem_statement: str,
        framework_context: str = "",
        operator_notes: str = "",
    ) -> SketchStrategy:
        """Generate SketchStrategy using chain-of-thought reasoning.

        This function uses structured reasoning to develop a proof strategy, analyzing the
        theorem statement, available framework results, and constraints to produce a coherent
        approach with identified strengths and weaknesses.

        Args:
            theorem_label: Framework label of the target statement (e.g., "thm-main-convergence")
            theorem_statement: Formal statement with hypotheses for which strategy is being produced
            framework_context: Optional background - prior results, parameter regimes, available tools
            operator_notes: Optional evaluator instructions - preferred tools, constraints, known pitfalls

        Returns:
            SketchStrategy object with 6 components:
                - strategist: Agent/model name
                - method: Concise technique name
                - summary: Narrative overview
                - keySteps: Ordered major stages
                - strengths: Advantages of this approach
                - weaknesses: Limitations and challenges

        Example:
            >>> label = "thm-kl-convergence-euclidean"
            >>> statement = '''
            ... Under Axiom (Confining Potential) and Axiom (QSD Log-Concave),
            ... the Euclidean Gas with cloning converges exponentially fast in
            ... relative entropy to the unique QSD.
            ... '''
            >>> context = "Available: thm-lsi-target, lem-cloning-contraction, thm-grönwall"
            >>> notes = "Prefer LSI-based approach, avoid direct Lyapunov construction"
            >>> items = get_strategy_items(label, statement, context, notes)
            >>> print(items.method)
            'LSI + Grönwall Iteration'
            >>> len(items.keySteps)
            5
            >>> print(items.keySteps[0])
            'Establish uniform LSI constant for the target measure'
        """
        with dspy.context(lm=lm):
            result = agent(
                theorem_label=theorem_label,
                theorem_statement=theorem_statement,
                framework_context=framework_context or "",
                operator_notes=operator_notes or "",
            )
        return result.strategy_items if hasattr(result, "strategy_items") else result

    get_strategy_items.agent = agent  # type: ignore[attr-defined]

    return get_strategy_items


def get_label_data(label: str) -> str:
    """Retrieve entity data for a given framework label from the registries."""
    from mathster.registry import search as registry_search

    label = label.strip()
    if not label:
        msg = "Label must be a non-empty string."
        raise ValueError(msg)

    selected_stage = "auto"
    registry_source = None
    entity: dict | None = None

    if selected_stage in {"auto", "preprocess"}:
        entity = registry_search.get_preprocess_label(label)
        if entity is not None:
            registry_source = "preprocess"

    if entity is None and selected_stage in {"auto", "directives"}:
        entity = registry_search.get_directive_label(label)
        if entity is not None:
            registry_source = "directives"

    if entity is None:
        if selected_stage == "preprocess":
            scope = "the preprocess registry"
        elif selected_stage == "directives":
            scope = "the directives registry"
        else:
            scope = "either registry"
        raise ValueError(f"Label '{label}' was not found in {scope}.")

    assert registry_source is not None  # For type checkers

    return json.dumps(entity, indent=2, sort_keys=True)


class SketchStrategist(dspy.Module):
    """Orchestrating module that generates complete SketchStrategy using ReAct reasoning.

    This module coordinates multiple specialized tools to build a comprehensive proof sketch strategy:
    1. Generates basic strategy items (method, steps, strengths, weaknesses)
    2. Extracts framework dependencies (theorems, lemmas, axioms, definitions)
    3. Identifies technical deep dives (challenging sub-problems)
    4. Assesses confidence score
    5. Assembles all components into a complete SketchStrategy

    The module uses a ReAct agent that reasons about how to use the available tools
    and orchestrates the workflow to produce a coherent, well-structured proof strategy.

    Example:
        >>> strategist = SketchStrategist()
        >>> result = strategist(
        ...     theorem_label="thm-kl-convergence",
        ...     theorem_statement="Under confining potential, Gas converges to QSD.",
        ...     framework_context="Available: LSI theory, cloning lemmas",
        ...     operator_notes="Prefer LSI-based approach",
        ... )
        >>> print(result.strategy.method)
        'LSI + Grönwall Iteration'
        >>> len(result.strategy.frameworkDependencies.theorems)
        3
    """

    def __init__(self, max_iters: int = 5, lm: dspy.LM = None, claude_model="sonnet") -> None:
        """Initialize the SketchStrategist with all required tools.

        Args:
            max_iters: Maximum iterations for ReAct agent
            lm: Language model to use (defaults to dspy.settings.lm)
        """
        super().__init__()
        self.lm = lm or dspy.settings.lm

        # Initialize the three specialized tool functions
        self._get_strategy_items_impl = configure_get_strategy_items(lm=self.lm)
        self._get_framework_dependencies_impl = setup_get_framework_dependencies(
            lm=self.lm,
            claude_model=claude_model,
        )
        self._get_technical_deep_dives_impl = setup_get_technical_deep_dive(
            lm=self.lm, claude_model=claude_model
        )

        # Expose the underlying DSPy predictors so Refine/trace tooling can map them.
        try:
            self.strategy_items_agent = self._get_strategy_items_impl.agent
        except AttributeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Strategy items tool is missing its DSPy agent reference") from exc

        try:
            self.framework_dependencies_agent = self._get_framework_dependencies_impl.agent
        except AttributeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Framework dependency tool is missing its DSPy agent reference") from exc

        try:
            self.technical_deep_dives_agent = self._get_technical_deep_dives_impl.agent
        except AttributeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Technical deep-dives tool is missing its DSPy agent reference") from exc

        # Create tool wrappers for the ReAct agent
        def get_strategy_items_tool(
            theorem_label: str, theorem_statement: str, framework_context: str, operator_notes: str
        ) -> str:
            """Generate basic strategy items (method, summary, steps, strengths, weaknesses).

            This should be called FIRST to establish the high-level proof approach.

            Args:
                theorem_label: Framework label (e.g., thm-main-convergence)
                theorem_statement: Formal theorem statement
                framework_context: Available results and background
                operator_notes: Constraints and preferences

            Returns:
                JSON string of SketchStrategyItems
            """
            logger.debug(
                "Tool call: get_strategy_items_tool(theorem_label=%s, statement_len=%d, context_len=%d, notes_len=%d)",
                theorem_label,
                len(theorem_statement),
                len(framework_context),
                len(operator_notes),
            )
            result = self._get_strategy_items_impl(
                theorem_label, theorem_statement, framework_context, operator_notes
            )
            logger.debug("Tool result: get_strategy_items_tool returned method=%s", result.method)
            return result.model_dump_json()

        def get_framework_dependencies_tool(proof_sketch: str) -> str:
            """Extract all framework dependencies from the proof sketch.

            Call this AFTER get_strategy_items_tool to identify required theorems,
            lemmas, axioms, and definitions.

            Args:
                proof_sketch: Narrative proof sketch containing dependency references

            Returns:
                JSON string of FrameworkDependencies (theorems, lemmas, axioms, definitions)
            """
            logger.debug(
                "Tool call: get_framework_dependencies_tool(sketch_len=%d)",
                len(proof_sketch),
            )
            result = self._get_framework_dependencies_impl(proof_sketch)
            logger.debug(
                "Tool result: get_framework_dependencies_tool found %d theorems, %d lemmas, %d axioms, %d definitions",
                len(result.theorems),
                len(result.lemmas),
                len(result.axioms),
                len(result.definitions),
            )
            return result.model_dump_json()

        def get_technical_deep_dives_tool(proof_sketch: str) -> str:
            """Identify technical challenges and sub-problems in the proof strategy.

            Call this AFTER get_strategy_items_tool to extract detailed technical obstacles.

            Args:
                proof_sketch: Narrative proof sketch containing challenges

            Returns:
                JSON string of list[TechnicalDeepDive]
            """
            logger.debug(
                "Tool call: get_technical_deep_dives_tool(sketch_len=%d)",
                len(proof_sketch),
            )
            result = self._get_technical_deep_dives_impl(proof_sketch)
            logger.debug(
                "Tool result: get_technical_deep_dives_tool found %d technical challenges",
                len(result),
            )
            return [dive.model_dump() for dive in result].__str__()

        def assess_confidence_tool(
            strategy_completeness: str, technical_challenges: str, dependencies_available: str
        ) -> str:
            """Assess confidence score (High/Medium/Low) for the proof strategy.

            Args:
                strategy_completeness: How complete and coherent is the strategy?
                technical_challenges: How severe are the identified challenges?
                dependencies_available: Are all required dependencies available in framework?

            Returns:
                One of: "High", "Medium", "Low"
            """
            logger.debug(
                "Tool call: assess_confidence_tool(completeness=%s, challenges=%s, dependencies=%s)",
                strategy_completeness[:50],
                technical_challenges[:50],
                dependencies_available[:50],
            )
            # Simple heuristic - in practice this could be more sophisticated
            completeness_map = {"complete": 1, "mostly": 0.7, "partial": 0.3}
            challenges_map = {"minor": 1, "moderate": 0.6, "severe": 0.2}
            dependencies_map = {"all": 1, "most": 0.7, "some": 0.4, "few": 0.1}

            score = 0.0
            for key, value in completeness_map.items():
                if key in strategy_completeness.lower():
                    score += value * 0.4
                    break
            for key, value in challenges_map.items():
                if key in technical_challenges.lower():
                    score += value * 0.3
                    break
            for key, value in dependencies_map.items():
                if key in dependencies_available.lower():
                    score += value * 0.3
                    break

            if score >= 0.7:
                confidence = "High"
            elif score >= 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"

            logger.debug("Tool result: assess_confidence_tool returned %s (score=%.2f)", confidence, score)
            return confidence

        # Store tools
        self.tools = [
            get_strategy_items_tool,
            get_framework_dependencies_tool,
            get_technical_deep_dives_tool,
            assess_confidence_tool,
            get_label_data,
        ]

        # Create ReAct agent with orchestration instructions
        INSTRUCTIONS = """
You are orchestrating the generation of a complete SketchStrategy for a mathematical theorem.

Workflow (follow this order):
1. FIRST: Call get_strategy_items_tool with all four inputs to generate basic strategy
2. SECOND: Format the strategy summary and key steps into a narrative proof sketch
3. THIRD: Call get_framework_dependencies_tool with the formatted proof sketch
4. FOURTH: Call get_technical_deep_dives_tool with the formatted proof sketch
5. FIFTH: Call assess_confidence_tool based on what you learned from previous steps
6. FINALLY: Assemble all components into the final SketchStrategy object

Important guidelines:
- You MUST call ALL four tools in the correct order
- Use the strategy items output to formulate clear proof sketch prompts for tools 2-3
- The proof sketch should mention specific results, techniques, and challenges
- For confidence assessment, consider: completeness, technical difficulty, dependency availability
- The final SketchStrategy must include ALL fields from the signature

Output structure reminder:
- strategist: Agent name (use "Claude Sonnet 4.5 via SketchStrategist")
- method, summary, keySteps, strengths, weaknesses: From get_strategy_items_tool
- frameworkDependencies: From get_framework_dependencies_tool
- technicalDeepDives: From get_technical_deep_dives_tool
- confidenceScore: From assess_confidence_tool
"""
        signature = SketchStrategySignature.with_instructions(INSTRUCTIONS)
        self.agent = dspy.ReAct(signature, tools=self.tools, max_iters=max_iters)

    def forward(
        self,
        theorem_label: str,
        theorem_statement: str,
        framework_context: str = "",
        operator_notes: str = "",
    ) -> dspy.Prediction:
        """Generate complete SketchStrategy using orchestrated tool calls.

        This method coordinates multiple specialized tools through ReAct reasoning
        to build a comprehensive proof sketch strategy.

        Args:
            theorem_label: Framework label of the target theorem (e.g., "thm-main-convergence")
            theorem_statement: Formal statement with hypotheses for the theorem
            framework_context: Optional background - prior results, parameter regimes, available tools
            operator_notes: Optional instructions - preferred tools, constraints, known pitfalls

        Returns:
            dspy.Prediction containing:
                - strategy: Complete SketchStrategy object with all components
                - Includes: method, summary, steps, dependencies, deep dives, confidence

        Example:
            >>> strategist = SketchStrategist()
            >>> result = strategist(
            ...     theorem_label="thm-kl-convergence-euclidean",
            ...     theorem_statement='''
            ...         Under Axiom (Confining Potential) and Axiom (QSD Log-Concave),
            ...         the Euclidean Gas converges exponentially to unique QSD.
            ...     ''',
            ...     framework_context="LSI theory available: thm-lsi-target, lem-tensorization",
            ...     operator_notes="Prefer LSI-based approach over direct Lyapunov",
            ... )
            >>> strategy = result.strategy
            >>> print(strategy.method)
            'LSI + Grönwall Iteration'
            >>> print(len(strategy.frameworkDependencies.theorems))
            2
            >>> print(len(strategy.technicalDeepDives))
            3
            >>> print(strategy.confidenceScore)
            'High'
        """
        with dspy.context(lm=self.lm):
            return self.agent(
                theorem_label=theorem_label,
                theorem_statement=theorem_statement,
                framework_context=framework_context or "",
                operator_notes=operator_notes or "",
            )
