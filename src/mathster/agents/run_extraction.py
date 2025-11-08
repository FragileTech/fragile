#!/usr/bin/env python3

"""
Convenience script to run all directive extraction agents on a single document.

Example
-------
python -m mathster.agents.run_extraction \
  --doc docs/source/1_euclidean_gas/03_cloning.md \
  --lm openai/gpt-4o-mini \
  --passes 4
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Sequence

from mathster.agents import (
    extract_algorithms,
    extract_assumptions,
    extract_axioms,
    extract_conjectures,
    extract_definitions,
    extract_proofs,
    extract_remarks,
    extract_theorems,
)


logger = logging.getLogger(__name__)


AgentRunner = Callable[..., None]


AGENT_CONFIG: dict[str, dict[str, object]] = {
    "theorems": {
        "runner": extract_theorems.run_agent,
        "threshold": 0.95,
        "max_tokens": 16000,
    },
    "definitions": {
        "runner": extract_definitions.run_agent,
        "threshold": 0.90,
        "max_tokens": 12000,
    },
    "proofs": {
        "runner": extract_proofs.run_agent,
        "threshold": 0.90,
        "max_tokens": 16000,
    },
    "assumptions": {
        "runner": extract_assumptions.run_agent,
        "threshold": 0.90,
        "max_tokens": 16000,
    },
    "axioms": {
        "runner": extract_axioms.run_agent,
        "threshold": 0.90,
        "max_tokens": 16000,
    },
    "remarks": {
        "runner": extract_remarks.run_agent,
        "threshold": 0.90,
        "max_tokens": 16000,
    },
    "conjectures": {
        "runner": extract_conjectures.run_agent,
        "threshold": 0.90,
        "max_tokens": 16000,
    },
    "algorithms": {
        "runner": extract_algorithms.run_agent,
        "threshold": 0.90,
        "max_tokens": 16000,
    },
}


def run_agents_in_sequence(
    doc_path: str,
    agent_names: Sequence[str],
    *,
    lm_spec: str,
    passes: int,
    threshold_override: float | None,
    max_tokens_override: int | None,
) -> None:
    """
    Execute each requested agent sequentially.
    """

    for name in agent_names:
        config = AGENT_CONFIG[name]
        runner: AgentRunner = config["runner"]  # type: ignore[assignment]

        threshold = (
            threshold_override
            if threshold_override is not None
            else float(config.get("threshold", 0.9))
        )
        max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else int(config.get("max_tokens", 16000))
        )

        logger.info(
            "Running %s agent (threshold=%.3f, passes=%s, max_tokens=%s)...",
            name,
            threshold,
            passes,
            max_tokens,
        )
        runner(
            document_path=doc_path,
            lm_spec=lm_spec,
            passes=passes,
            threshold=threshold,
            max_tokens=max_tokens,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Mathster extraction agents on a single document."
    )
    parser.add_argument("--doc", required=True, help="Path to the source markdown document.")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENT_CONFIG.keys()),
        default=list(AGENT_CONFIG.keys()),
        help="Subset of agents to run (defaults to all).",
    )
    parser.add_argument(
        "--lm",
        default="gemini/gemini-flash-lite-latest",
        help="LM spec to use for every agent.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=5,
        help="Number of DSPy Refine passes per agent.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override reward threshold for all agents (defaults to per-agent value).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens per agent (defaults to per-agent value).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def main(argv: Sequence[str] | None = None) -> None:
    import flogging

    flogging.setup(level="INFO")
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    run_agents_in_sequence(
        doc_path=args.doc,
        agent_names=args.agents,
        lm_spec=args.lm,
        passes=args.passes,
        threshold_override=args.threshold,
        max_tokens_override=args.max_tokens,
    )


if __name__ == "__main__":
    main()
