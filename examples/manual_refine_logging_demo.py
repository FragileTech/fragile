#!/usr/bin/env python3
"""Demonstration of ManualRefineSketchPipeline with different logging verbosity levels.

This script shows how to use the enhanced custom logging capabilities of the
ManualRefineSketchPipeline module with configurable verbosity levels.

Usage:
    # Run with standard logging (default)
    python examples/manual_refine_logging_demo.py

    # Run with detailed logging
    python examples/manual_refine_logging_demo.py --verbosity detailed

    # Run with verbose logging and JSON export
    python examples/manual_refine_logging_demo.py --verbosity verbose --json-log refinement_log.json

    # Run with minimal logging (only start/end summary)
    python examples/manual_refine_logging_demo.py --verbosity minimal
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import sys

from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.manual_refine_pipeline import (
    ManualRefineSketchPipeline,
    LogVerbosity,
)
from mathster.proof_sketcher.sketch_pipeline import AgentSketchPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


def demo_minimal_verbosity():
    """Demonstrate minimal verbosity: only start/end summary."""
    print("\n" + "=" * 80)
    print("DEMO: MINIMAL VERBOSITY")
    print("Shows only: pipeline start, final summary")
    print("=" * 80)

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025", temperature=0.1)

    agent = ManualRefineSketchPipeline(
        pipeline=AgentSketchPipeline(),
        N=3,
        threshold=60,
        verbosity=LogVerbosity.MINIMAL,
    )

    sample_data = {
        "title_hint": "Test Theorem",
        "theorem_label": "thm-test",
        "theorem_type": "Theorem",
        "theorem_statement": "Test statement for minimal logging demo.",
        "document_source": "test.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
    }

    result = agent(**sample_data)
    print(f"\n[OUTPUT] Best score: {result.best_score:.2f}/100")
    print(f"[OUTPUT] Total iterations: {result.total_iterations}")


def demo_standard_verbosity():
    """Demonstrate standard verbosity: per-iteration summaries."""
    print("\n" + "=" * 80)
    print("DEMO: STANDARD VERBOSITY (Default)")
    print("Shows: pipeline start, per-iteration summaries, final summary")
    print("=" * 80)

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025", temperature=0.1)

    agent = ManualRefineSketchPipeline(
        pipeline=AgentSketchPipeline(),
        N=3,
        threshold=60,
        verbosity=LogVerbosity.STANDARD,
    )

    sample_data = {
        "title_hint": "Test Theorem",
        "theorem_label": "thm-test",
        "theorem_type": "Theorem",
        "theorem_statement": "Test statement for standard logging demo.",
        "document_source": "test.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
    }

    result = agent(**sample_data)
    print(f"\n[OUTPUT] Score progression: {result.scores}")


def demo_detailed_verbosity():
    """Demonstrate detailed verbosity: adds score breakdowns."""
    print("\n" + "=" * 80)
    print("DEMO: DETAILED VERBOSITY")
    print("Shows: standard output + detailed score component breakdowns")
    print("=" * 80)

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025", temperature=0.1)

    agent = ManualRefineSketchPipeline(
        pipeline=AgentSketchPipeline(),
        N=3,
        threshold=60,
        verbosity=LogVerbosity.DETAILED,
    )

    sample_data = {
        "title_hint": "Test Theorem",
        "theorem_label": "thm-test",
        "theorem_type": "Theorem",
        "theorem_statement": "Test statement for detailed logging demo.",
        "document_source": "test.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
    }

    result = agent(**sample_data)
    print(f"\n[OUTPUT] Score variance: {result.score_variance:.2f}")


def demo_verbose_verbosity():
    """Demonstrate verbose verbosity: adds iteration comparisons and convergence."""
    print("\n" + "=" * 80)
    print("DEMO: VERBOSE VERBOSITY")
    print("Shows: detailed output + iteration comparisons + convergence analysis")
    print("=" * 80)

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025", temperature=0.1)

    agent = ManualRefineSketchPipeline(
        pipeline=AgentSketchPipeline(),
        N=3,
        threshold=60,
        verbosity=LogVerbosity.VERBOSE,
    )

    sample_data = {
        "title_hint": "Test Theorem",
        "theorem_label": "thm-test",
        "theorem_type": "Theorem",
        "theorem_statement": "Test statement for verbose logging demo.",
        "document_source": "test.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
    }

    result = agent(**sample_data)
    print(f"\n[OUTPUT] All iterations tracked: {len(result.all_iterations)}")


def demo_json_export():
    """Demonstrate JSON export functionality."""
    print("\n" + "=" * 80)
    print("DEMO: JSON EXPORT")
    print("Shows: Export refinement metrics to JSON file")
    print("=" * 80)

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025", temperature=0.1)

    json_path = "/tmp/refinement_log_demo.json"
    agent = ManualRefineSketchPipeline(
        pipeline=AgentSketchPipeline(),
        N=3,
        threshold=60,
        verbosity=LogVerbosity.STANDARD,
        log_json_path=json_path,
    )

    sample_data = {
        "title_hint": "Test Theorem",
        "theorem_label": "thm-test",
        "theorem_type": "Theorem",
        "theorem_statement": "Test statement for JSON export demo.",
        "document_source": "test.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
    }

    result = agent(**sample_data)

    print(f"\n[OUTPUT] JSON log written to: {json_path}")
    with open(json_path) as f:
        log_data = json.load(f)
    print(f"[OUTPUT] JSON contains {len(log_data['iterations'])} iterations")
    print(f"[OUTPUT] JSON keys: {list(log_data.keys())}")


def demo_all_features():
    """Demonstrate all features together."""
    print("\n" + "=" * 80)
    print("DEMO: ALL FEATURES COMBINED")
    print("Shows: Verbose logging + JSON export + custom parameters")
    print("=" * 80)

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025", temperature=0.1)

    json_path = "/tmp/refinement_full_demo.json"
    agent = ManualRefineSketchPipeline(
        pipeline=AgentSketchPipeline(),
        N=3,  # Few iterations for demo
        threshold=70,  # Higher threshold
        fail_count=2,  # Lower fail count
        verbosity=LogVerbosity.VERBOSE,
        log_json_path=json_path,
    )

    sample_data = {
        "title_hint": "KL Convergence Theorem",
        "theorem_label": "thm-kl-convergence",
        "theorem_type": "Theorem",
        "theorem_statement": (
            "The swarm law μ_t converges exponentially fast in KL divergence to "
            "the unique quasi-stationary distribution π."
        ),
        "document_source": "docs/source/1_euclidean_gas/09_kl_convergence.md",
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "proof_status": "Sketch",
        "framework_context": "LSI theory, Bakry-Émery criterion, Grönwall's lemma available.",
        "operator_notes": "Prefer LSI-based approach. Focus on explicit constants.",
    }

    result = agent(**sample_data)

    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    print(f"Best score: {result.best_score:.2f}/100 (iteration {result.best_iteration_num})")
    print(f"Total iterations: {result.total_iterations}")
    print(f"Total time: {result.total_time:.1f}s")
    print(f"Avg time/iter: {result.average_time_per_iteration:.1f}s")
    print(f"Score improvement: {result.score_improvement:+.2f}")
    print(f"Score variance: {result.score_variance:.2f}")
    print(f"Threshold met: {result.threshold_met}")
    print(f"Stopped: {result.stopped_reason}")

    print("\nPer-iteration breakdown:")
    for it in result.all_iterations:
        print(
            f"  Iteration {it.iteration_num}: score={it.score:.2f}, "
            f"time={it.elapsed_time:.1f}s, "
            f"is_best={it.is_best}, "
            f"improvement={it.improvement:+.2f}"
        )


def main():
    """Run demonstrations based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate ManualRefineSketchPipeline logging features"
    )
    parser.add_argument(
        "--demo",
        choices=["minimal", "standard", "detailed", "verbose", "json", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--verbosity",
        choices=["minimal", "standard", "detailed", "verbose", "debug"],
        help="Override verbosity for single-demo mode",
    )
    parser.add_argument(
        "--json-log",
        help="Path to export JSON log (for single-demo mode)",
    )

    args = parser.parse_args()

    if args.demo == "minimal":
        demo_minimal_verbosity()
    elif args.demo == "standard":
        demo_standard_verbosity()
    elif args.demo == "detailed":
        demo_detailed_verbosity()
    elif args.demo == "verbose":
        demo_verbose_verbosity()
    elif args.demo == "json":
        demo_json_export()
    elif args.demo == "all":
        demo_all_features()


if __name__ == "__main__":
    main()
