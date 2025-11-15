#!/usr/bin/env python3
"""Example script demonstrating Hydra-based proof sketcher configuration.

This script shows how to use the LLM configuration system with Hydra and OmegaConf
for flexible, YAML-based model selection across the proof sketcher pipeline.

Usage:
    # Use default configuration
    python examples/hydra_proof_sketcher.py

    # Override LLM config
    python examples/hydra_proof_sketcher.py llm=cost_optimized

    # Override pipeline settings
    python examples/hydra_proof_sketcher.py pipeline=quick

    # Override specific parameters
    python examples/hydra_proof_sketcher.py llm.perspective_1_model.model=gpt-4-turbo pipeline.N=3

    # Combine overrides
    python examples/hydra_proof_sketcher.py llm=development pipeline=quick

For more information on Hydra configuration, see:
https://hydra.cc/docs/intro/
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from mathster.proof_sketcher.agent_factory import create_refine_pipeline
from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run proof sketcher with Hydra configuration.

    Args:
        cfg: Hydra configuration object loaded from YAML files
    """
    # Log configuration for debugging
    logger.info("=" * 80)
    logger.info("PROOF SKETCHER WITH HYDRA CONFIGURATION")
    logger.info("=" * 80)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Instantiate LLM configuration from Hydra config
    lm_config = ProofSketcherLMConfig.from_dict(cfg.llm)

    # Validate configuration
    lm_config.validate()
    logger.info("LLM Configuration validated successfully")
    logger.info("Model Tier Summary:")
    logger.info("  - Perspective 1: %s", lm_config.perspective_1_model)
    logger.info("  - Perspective 2: %s", lm_config.perspective_2_model)
    logger.info("  - Synthesis: %s", lm_config.synthesis_model)
    logger.info("  - Fast: %s", lm_config.fast_model)

    # Create pipeline from configuration
    pipeline = create_refine_pipeline(
        lm_config=lm_config,
        N=cfg.pipeline.N,
        threshold=cfg.pipeline.threshold,
        fail_count=cfg.pipeline.fail_count,
        verbosity=cfg.pipeline.verbosity,
        log_json_path=cfg.pipeline.get("log_json_path"),
        enable_iteration_feedback=cfg.pipeline.enable_iteration_feedback,
    )

    logger.info("Pipeline created successfully")
    logger.info("Refinement Settings:")
    logger.info("  - Max iterations: %d", cfg.pipeline.N)
    logger.info("  - Threshold: %.1f", cfg.pipeline.threshold)
    logger.info("  - Fail count: %d", cfg.pipeline.fail_count)
    logger.info("  - Verbosity: %s", cfg.pipeline.verbosity)
    logger.info("  - Iteration feedback: %s", cfg.pipeline.enable_iteration_feedback)

    # Example theorem data (replace with actual data)
    theorem_data = {
        "title_hint": "KL Convergence Under LSI",
        "theorem_label": "thm-kl-convergence-lsi",
        "theorem_type": "MainResult",
        "theorem_statement": (
            "Under the Log-Sobolev Inequality (LSI) with constant λ, "
            "the Kullback-Leibler divergence between the empirical distribution "
            "of the Euclidean Gas and the target quasi-stationary distribution "
            "converges exponentially fast with rate proportional to λ."
        ),
        "document_source": "docs/source/1_euclidean_gas/09_kl_convergence.md",
        "creation_date": "2025-01-12",
        "proof_status": "sketched",
        "framework_context": (
            "This theorem builds on:\n"
            "- thm-lsi-euclidean (Log-Sobolev Inequality for Euclidean Gas)\n"
            "- thm-qsd-existence (Existence of Quasi-Stationary Distribution)\n"
            "- lem-kl-grönwall (Grönwall's Inequality for KL Divergence)\n"
            "The proof technique follows the classical LSI → Grönwall → exponential rate pipeline."
        ),
        "operator_notes": (
            "Focus on:\n"
            "1. Verifying LSI constant bounds\n"
            "2. Setting up the entropy production inequality\n"
            "3. Applying Grönwall's inequality correctly\n"
            "4. Extracting explicit convergence rate"
        ),
    }

    # Run refinement pipeline
    logger.info("=" * 80)
    logger.info("STARTING PROOF SKETCH REFINEMENT")
    logger.info("=" * 80)

    result = pipeline(**theorem_data)

    # Display results
    logger.info("=" * 80)
    logger.info("REFINEMENT COMPLETE")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info("  - Best score: %.2f/100", result.best_score)
    logger.info("  - Total iterations: %d", result.total_iterations)
    logger.info("  - Best iteration: #%d", result.best_iteration_num)
    logger.info("  - Score progression: %s", result.scores)
    logger.info("  - Early stopped: %s", result.early_stopped)
    logger.info("  - Threshold met: %s", result.threshold_met)
    logger.info("  - Stopped reason: %s", result.stopped_reason)
    logger.info("  - Total time: %.2fs", result.total_time)
    logger.info("  - Avg time/iteration: %.2fs", result.average_time_per_iteration)

    # Access best result
    best_result = result.best_result
    logger.info("\nBest Proof Sketch:")
    logger.info("  - Theorem: %s", best_result.sketch.statement.formal[:100] + "...")
    logger.info("  - Strategy 1 quality: Gemini score=%d/5", best_result.scores.gemini_overall_score)
    logger.info("  - Strategy 2 quality: Codex score=%d/5", best_result.scores.codex_overall_score)
    logger.info("  - Final decision: %s", best_result.validation_report.synthesisAndActionPlan.finalDecision)

    # Optionally export results
    if cfg.pipeline.get("log_json_path"):
        logger.info("\nRefinement log exported to: %s", cfg.pipeline.log_json_path)

    logger.info("=" * 80)
    logger.info("PROOF SKETCHER COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
