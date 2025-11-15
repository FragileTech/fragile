"""Factory functions for creating pre-configured proof sketcher agents.

This module provides convenient factory functions that instantiate complete
proof sketcher pipelines from ProofSketcherLMConfig objects. Use these functions
to simplify agent creation when working with Hydra/OmegaConf configuration.

Example:
    ```python
    from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
    from mathster.proof_sketcher.agent_factory import create_refine_pipeline

    # From Python config
    config = ProofSketcherLMConfig.cost_optimized()
    pipeline = create_refine_pipeline(lm_config=config, N=5, threshold=60)

    # From Hydra config
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="configs", config_name="config")
    def main(cfg: DictConfig):
        config = ProofSketcherLMConfig.from_dict(cfg.llm)
        pipeline = create_refine_pipeline(
            lm_config=config,
            N=cfg.pipeline.N,
            threshold=cfg.pipeline.threshold,
        )
        result = pipeline(**theorem_data)
    ```
"""

from __future__ import annotations

from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
from mathster.proof_sketcher.manual_refine_pipeline import (
    FeedbackConfig,
    LogVerbosity,
    ManualRefineSketchPipeline,
)
from mathster.proof_sketcher.sketch_pipeline import AgentSketchPipeline
from mathster.proof_sketcher.sketch_validator import SketchValidator
from mathster.proof_sketcher.sketcher import ProofSketchAgent


__all__ = [
    "create_drafting_agent",
    "create_validator",
    "create_sketch_pipeline",
    "create_refine_pipeline",
]


def create_drafting_agent(
    lm_config: ProofSketcherLMConfig,
) -> ProofSketchAgent:
    """Create a ProofSketchAgent with models from config.

    Args:
        lm_config: LLM configuration specifying models for different tiers

    Returns:
        Configured ProofSketchAgent instance

    Example:
        ```python
        config = ProofSketcherLMConfig.default()
        drafting_agent = create_drafting_agent(config)
        ```
    """
    lms = lm_config.to_dspy_lms()

    return ProofSketchAgent(
        strategist_1=lms["perspective_1"],  # Primary strategist
        strategist_2=lms["perspective_2"],  # Secondary strategist
        model=lms["fast"],  # Fast model for simple tasks
        stronger_model=lms["synthesis"],  # Synthesis model for complex reasoning
    )


def create_validator(
    lm_config: ProofSketcherLMConfig,
    project_root: str | None = None,
    gemini_prompt: str | None = None,
    codex_prompt: str | None = None,
) -> SketchValidator:
    """Create a SketchValidator with models from config.

    Args:
        lm_config: LLM configuration specifying models for different tiers
        project_root: Optional path to project root for prompt loading
        gemini_prompt: Optional custom prompt for Gemini reviewer
        codex_prompt: Optional custom prompt for Codex reviewer

    Returns:
        Configured SketchValidator instance

    Example:
        ```python
        config = ProofSketcherLMConfig.quality_optimized()
        validator = create_validator(config, project_root="/path/to/project")
        ```
    """
    lms = lm_config.to_dspy_lms()

    return SketchValidator(
        project_root=project_root,
        gemini_prompt=gemini_prompt,
        codex_prompt=codex_prompt,
        perspective_1_lm=lms["perspective_1"],  # Gemini reviewer
        perspective_2_lm=lms["perspective_2"],  # Codex reviewer
        synthesis_lm=lms["synthesis"],  # Consensus/synthesis
        fast_lm=lms["fast"],  # Metadata generation
    )


def create_sketch_pipeline(
    lm_config: ProofSketcherLMConfig,
    project_root: str | None = None,
    gemini_prompt: str | None = None,
    codex_prompt: str | None = None,
) -> AgentSketchPipeline:
    """Create an AgentSketchPipeline with models from config.

    This is a convenience wrapper around AgentSketchPipeline(lm_config=...).

    Args:
        lm_config: LLM configuration specifying models for different tiers
        project_root: Optional path to project root for prompt loading
        gemini_prompt: Optional custom prompt for Gemini reviewer
        codex_prompt: Optional custom prompt for Codex reviewer

    Returns:
        Configured AgentSketchPipeline instance

    Example:
        ```python
        config = ProofSketcherLMConfig.development()
        pipeline = create_sketch_pipeline(config)
        result = pipeline(
            title_hint="KL Convergence",
            theorem_label="thm-kl-conv",
            ...
        )
        ```
    """
    return AgentSketchPipeline(
        lm_config=lm_config,
        project_root=project_root,
        gemini_prompt=gemini_prompt,
        codex_prompt=codex_prompt,
    )


def create_refine_pipeline(
    lm_config: ProofSketcherLMConfig,
    N: int = 5,
    threshold: float = 60.0,
    fail_count: int = 5,
    verbosity: LogVerbosity | str = LogVerbosity.STANDARD,
    log_json_path: str | None = None,
    enable_iteration_feedback: bool = True,
    feedback_config: FeedbackConfig | None = None,
    project_root: str | None = None,
    gemini_prompt: str | None = None,
    codex_prompt: str | None = None,
) -> ManualRefineSketchPipeline:
    """Create a ManualRefineSketchPipeline with models from config.

    This is a convenience wrapper around ManualRefineSketchPipeline(lm_config=...).
    Provides complete refinement pipeline with iterative feedback.

    Args:
        lm_config: LLM configuration specifying models for different tiers
        N: Maximum number of refinement iterations
        threshold: Quality score threshold for early stopping (0-100)
        fail_count: Maximum consecutive failures before stopping
        verbosity: Logging verbosity level (minimal/standard/detailed/verbose/debug)
        log_json_path: Optional path to export refinement data as JSON
        enable_iteration_feedback: Enable feedback injection from best iteration
        feedback_config: Optional feedback configuration
        project_root: Optional path to project root for prompt loading
        gemini_prompt: Optional custom prompt for Gemini reviewer
        codex_prompt: Optional custom prompt for Codex reviewer

    Returns:
        Configured ManualRefineSketchPipeline instance

    Example:
        ```python
        from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
        from mathster.proof_sketcher.agent_factory import create_refine_pipeline

        config = ProofSketcherLMConfig.cost_optimized()
        pipeline = create_refine_pipeline(
            lm_config=config,
            N=5,
            threshold=60,
            verbosity="DETAILED",
        )

        result = pipeline(
            title_hint="KL Convergence Rate",
            theorem_label="thm-kl-conv",
            theorem_type="MainResult",
            theorem_statement="Under LSI, KL converges exponentially...",
            document_source="09_kl_convergence.md",
            creation_date="2025-01-12",
            proof_status="sketched",
        )

        print(f"Best score: {result.best_score:.2f}/100")
        print(f"Iterations: {result.total_iterations}")
        print(f"Reason: {result.stopped_reason}")
        ```
    """
    return ManualRefineSketchPipeline(
        lm_config=lm_config,
        N=N,
        threshold=threshold,
        fail_count=fail_count,
        verbosity=verbosity,
        log_json_path=log_json_path,
        enable_iteration_feedback=enable_iteration_feedback,
        feedback_config=feedback_config,
        project_root=project_root,
        gemini_prompt=gemini_prompt,
        codex_prompt=codex_prompt,
    )
