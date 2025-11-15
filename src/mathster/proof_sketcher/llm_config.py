"""LLM configuration for proof sketcher pipeline.

This module provides OmegaConf-compatible dataclasses for configuring language models
at different stages of the proof sketcher pipeline. Enables YAML-based configuration
with Hydra for fine-grained control over model selection.

Model Tiers:
    - Perspective 1: strategist_1 + Gemini reviewer (primary proof perspective)
    - Perspective 2: strategist_2 + Codex reviewer (secondary proof perspective)
    - Synthesis: All synthesis/consensus agents (cross-cutting analysis)
    - Fast: Simple extraction and formatting tasks

Example YAML:
    ```yaml
    perspective_1_model:
      provider: "openai"
      model: "gpt-4-turbo-preview"
      temperature: 0.7
      max_tokens: 4000

    perspective_2_model:
      provider: "anthropic"
      model: "claude-3-opus-20240229"
      temperature: 0.7
      max_tokens: 4000

    synthesis_model:
      provider: "openai"
      model: "gpt-4"
      temperature: 0.5
      max_tokens: 3000

    fast_model:
      provider: "openai"
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 2000
    ```

Usage:
    ```python
    from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
    from mathster.proof_sketcher.agent_factory import create_refine_pipeline

    # From Python
    config = ProofSketcherLMConfig.default()
    pipeline = create_refine_pipeline(lm_config=config)

    # From YAML with Hydra
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="configs", config_name="config")
    def main(cfg: DictConfig):
        config = ProofSketcherLMConfig.from_dict(cfg.llm)
        pipeline = create_refine_pipeline(lm_config=config)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dspy


@dataclass
class LLMModelConfig:
    """Configuration for a single language model.

    Attributes:
        provider: Model provider (e.g., "openai", "anthropic", "together")
        model: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
        temperature: Sampling temperature (0.0-1.0, default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4000)
        api_base: Optional API base URL override
        api_key: Optional API key override (use env vars when possible)
        additional_kwargs: Additional provider-specific parameters
    """

    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    api_base: str | None = None
    api_key: str | None = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dspy_lm(self) -> dspy.LM:
        """Convert configuration to dspy.LM instance.

        Returns:
            Configured dspy.LM instance ready for use in pipeline.

        Example:
            ```python
            config = LLMModelConfig(provider="openai", model="gpt-4")
            lm = config.to_dspy_lm()
            ```
        """
        kwargs = {
            "model": f"{self.provider}/{self.model}",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
        }

        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        return dspy.LM(**kwargs)

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"LLMModelConfig({self.provider}/{self.model}, T={self.temperature})"


@dataclass
class ProofSketcherLMConfig:
    """Complete LLM configuration for proof sketcher pipeline.

    Defines four model tiers optimized for different stages of proof generation:

    1. **Perspective 1 Model** (Primary proof perspective):
       - SketchStrategist (strategist_1): Classical techniques focus
       - SketchRefereeAgent (Gemini): Primary validation review
       - All 5 Gemini sub-agents (Completeness, LogicalFlow, Dependency, Tech, Overall)

    2. **Perspective 2 Model** (Secondary proof perspective):
       - SketchStrategist (strategist_2): Fragile Gas theory focus
       - SketchRefereeAgent (Codex): Secondary validation review
       - All 5 Codex sub-agents (Completeness, LogicalFlow, Dependency, Tech, Overall)

    3. **Synthesis Model** (Cross-cutting analysis):
       - StrategySynthesizer: Compare and merge dual strategies
       - AgentOverallAssessment (both reviewers): Synthesize 4 review components
       - Consensus Analysis: Compare reviewer agreement/disagreement
       - Action Items: Generate prioritized TODO list
       - Synthesis & Action Plan: Final decision and roadmap
       - DetailedProofAgent: Generate structured proof steps

    4. **Fast Model** (Simple extraction/formatting):
       - ProofStatementAgent: Formal/informal statement generation
       - CrossReferencesAgent: Label extraction
       - FutureWorkAgent: Gap categorization
       - AlternativeApproachesAgent: Record rejected ideas
       - ExpansionRoadmapAgent: Project planning
       - ValidationChecklistAgent: Boolean checklist
       - DependencyLedgerAgent: Dependency extraction
       - TechnicalDeepDiveAgent: Challenge identification
       - Metadata Generator: Report metadata
       - IterationFeedbackFormatter: Feedback extraction

    Attributes:
        perspective_1_model: Config for primary perspective (strategist_1 + Gemini)
        perspective_2_model: Config for secondary perspective (strategist_2 + Codex)
        synthesis_model: Config for synthesis/consensus agents
        fast_model: Config for simple extraction/formatting tasks
        claude_model_heavy: Claude model for complex tool calls (default: "sonnet")
        claude_model_fast: Claude model for simple tool calls (default: "haiku")
    """

    perspective_1_model: LLMModelConfig
    perspective_2_model: LLMModelConfig
    synthesis_model: LLMModelConfig
    fast_model: LLMModelConfig
    claude_model_heavy: str = "sonnet"
    claude_model_fast: str = "haiku"

    @classmethod
    def default(cls) -> ProofSketcherLMConfig:
        """Create default configuration with GPT-4 for all tiers.

        Returns:
            Default configuration suitable for high-quality proof generation.

        Example:
            ```python
            config = ProofSketcherLMConfig.default()
            ```
        """
        return cls(
            perspective_1_model=LLMModelConfig(
                provider="openai",
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=4000,
            ),
            perspective_2_model=LLMModelConfig(
                provider="openai",
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=4000,
            ),
            synthesis_model=LLMModelConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.5,
                max_tokens=3000,
            ),
            fast_model=LLMModelConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=2000,
            ),
            claude_model_heavy="sonnet",
            claude_model_fast="haiku",
        )

    @classmethod
    def cost_optimized(cls) -> ProofSketcherLMConfig:
        """Create cost-optimized configuration with tiered model selection.

        Uses stronger models only for perspectives and synthesis, with fast models
        for simple tasks. Estimated cost reduction: 40-50% vs. default.

        Returns:
            Cost-optimized configuration balancing quality and expense.

        Example:
            ```python
            config = ProofSketcherLMConfig.cost_optimized()
            ```
        """
        return cls(
            perspective_1_model=LLMModelConfig(
                provider="openai",
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=4000,
            ),
            perspective_2_model=LLMModelConfig(
                provider="anthropic",
                model="claude-3-opus-20240229",
                temperature=0.7,
                max_tokens=4000,
            ),
            synthesis_model=LLMModelConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.5,
                max_tokens=3000,
            ),
            fast_model=LLMModelConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=2000,
            ),
            claude_model_heavy="sonnet",
            claude_model_fast="haiku",
        )

    @classmethod
    def quality_optimized(cls) -> ProofSketcherLMConfig:
        """Create quality-optimized configuration with strongest models.

        Uses GPT-4 or Claude Opus for all reasoning tasks. Maximizes proof quality
        at higher cost. Suitable for critical theorems or publication-ready proofs.

        Returns:
            Quality-optimized configuration for maximum rigor.

        Example:
            ```python
            config = ProofSketcherLMConfig.quality_optimized()
            ```
        """
        return cls(
            perspective_1_model=LLMModelConfig(
                provider="openai",
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=4000,
            ),
            perspective_2_model=LLMModelConfig(
                provider="anthropic",
                model="claude-3-opus-20240229",
                temperature=0.7,
                max_tokens=4000,
            ),
            synthesis_model=LLMModelConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.5,
                max_tokens=3000,
            ),
            fast_model=LLMModelConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.7,
                max_tokens=3000,
            ),
            claude_model_heavy="sonnet",
            claude_model_fast="sonnet",
        )

    @classmethod
    def development(cls) -> ProofSketcherLMConfig:
        """Create development configuration with fast, cheap models.

        Uses GPT-3.5-turbo for all tasks. Suitable for testing, debugging, and
        rapid iteration. Estimated cost: ~10% of default configuration.

        Returns:
            Development configuration for fast iteration.

        Example:
            ```python
            config = ProofSketcherLMConfig.development()
            ```
        """
        return cls(
            perspective_1_model=LLMModelConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=3000,
            ),
            perspective_2_model=LLMModelConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=3000,
            ),
            synthesis_model=LLMModelConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=2000,
            ),
            fast_model=LLMModelConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=2000,
            ),
            claude_model_heavy="haiku",
            claude_model_fast="haiku",
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ProofSketcherLMConfig:
        """Create configuration from dictionary (e.g., from Hydra/OmegaConf).

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            Instantiated configuration object

        Example:
            ```python
            from omegaconf import OmegaConf

            yaml_str = \"\"\"
            perspective_1_model:
              provider: openai
              model: gpt-4
            \"\"\"
            cfg = OmegaConf.create(yaml_str)
            config = ProofSketcherLMConfig.from_dict(cfg)
            ```
        """
        return cls(
            perspective_1_model=LLMModelConfig(**config_dict["perspective_1_model"]),
            perspective_2_model=LLMModelConfig(**config_dict["perspective_2_model"]),
            synthesis_model=LLMModelConfig(**config_dict["synthesis_model"]),
            fast_model=LLMModelConfig(**config_dict["fast_model"]),
            claude_model_heavy=config_dict.get("claude_model_heavy", "sonnet"),
            claude_model_fast=config_dict.get("claude_model_fast", "haiku"),
        )

    def to_dspy_lms(self) -> dict[str, dspy.LM]:
        """Convert all model configs to dspy.LM instances.

        Returns:
            Dictionary mapping tier names to dspy.LM instances:
            - "perspective_1": Primary perspective LM
            - "perspective_2": Secondary perspective LM
            - "synthesis": Synthesis LM
            - "fast": Fast tasks LM

        Example:
            ```python
            config = ProofSketcherLMConfig.default()
            lms = config.to_dspy_lms()

            # Use in agents
            strategist_1 = SketchStrategist(lm=lms["perspective_1"])
            synthesizer = StrategySynthesizer(lm=lms["synthesis"])
            ```
        """
        return {
            "perspective_1": self.perspective_1_model.to_dspy_lm(),
            "perspective_2": self.perspective_2_model.to_dspy_lm(),
            "synthesis": self.synthesis_model.to_dspy_lm(),
            "fast": self.fast_model.to_dspy_lm(),
        }

    def validate(self) -> None:
        """Validate configuration has valid model specifications.

        Raises:
            ValueError: If any model configuration is invalid

        Example:
            ```python
            config = ProofSketcherLMConfig.default()
            config.validate()  # Raises if invalid
            ```
        """
        # Check all models have required fields
        for name, model in [
            ("perspective_1_model", self.perspective_1_model),
            ("perspective_2_model", self.perspective_2_model),
            ("synthesis_model", self.synthesis_model),
            ("fast_model", self.fast_model),
        ]:
            if not model.provider:
                raise ValueError(f"{name}.provider cannot be empty")
            if not model.model:
                raise ValueError(f"{name}.model cannot be empty")
            if not 0.0 <= model.temperature <= 2.0:
                raise ValueError(
                    f"{name}.temperature must be in [0.0, 2.0], got {model.temperature}"
                )
            if model.max_tokens <= 0:
                raise ValueError(
                    f"{name}.max_tokens must be positive, got {model.max_tokens}"
                )

        # Check Claude model names
        valid_claude_models = ["haiku", "sonnet", "opus"]
        if self.claude_model_heavy not in valid_claude_models:
            raise ValueError(
                f"claude_model_heavy must be one of {valid_claude_models}, "
                f"got {self.claude_model_heavy}"
            )
        if self.claude_model_fast not in valid_claude_models:
            raise ValueError(
                f"claude_model_fast must be one of {valid_claude_models}, "
                f"got {self.claude_model_fast}"
            )

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return (
            f"ProofSketcherLMConfig(\n"
            f"  perspective_1: {self.perspective_1_model},\n"
            f"  perspective_2: {self.perspective_2_model},\n"
            f"  synthesis: {self.synthesis_model},\n"
            f"  fast: {self.fast_model}\n"
            f")"
        )
