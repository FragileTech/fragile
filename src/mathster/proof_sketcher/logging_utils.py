#!/usr/bin/env python3
"""Logging utilities for proof sketcher workflow observability."""

from __future__ import annotations

from contextlib import contextmanager
import functools
import logging
import time
from typing import Any, Callable, TypeVar

import dspy


__all__ = [
    "agent_execution_context",
    "format_agent_context",
    "log_agent_execution",
    "log_dspy_prediction",
]

F = TypeVar("F", bound=Callable[..., Any])


def log_agent_execution(
    agent_name: str | None = None,
    level: int = logging.INFO,
    log_inputs: bool = False,
    log_outputs: bool = False,
) -> Callable[[F], F]:
    """Decorator to log agent/module execution with timing.

    Args:
        agent_name: Name to use in logs (defaults to function name)
        level: Logging level (INFO or DEBUG)
        log_inputs: Whether to log input arguments
        log_outputs: Whether to log output results

    Usage:
        @log_agent_execution("MyAgent")
        def forward(self, **kwargs):
            ...
    """

    def decorator(func: F) -> F:
        logger = logging.getLogger(func.__module__)
        resolved_name = agent_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            # Log start
            if log_inputs:
                logger.log(level, f"▶ {resolved_name} starting with inputs: {_safe_repr(kwargs)}")
            else:
                logger.log(level, f"▶ {resolved_name} starting")

            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time

                # Log success
                if log_outputs:
                    logger.log(
                        level,
                        f"✓ {resolved_name} completed in {elapsed:.2f}s with output: {_safe_repr(result)}",
                    )
                else:
                    logger.log(level, f"✓ {resolved_name} completed in {elapsed:.2f}s")

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"✗ {resolved_name} failed after {elapsed:.2f}s: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise

        return wrapper  # type: ignore

    return decorator


@contextmanager
def agent_execution_context(
    agent_name: str,
    extra: dict[str, Any] | None = None,
    level: int = logging.INFO,
):
    """Context manager for timing and logging agent execution blocks.

    Args:
        agent_name: Name of the agent/module
        extra: Additional context to log
        level: Logging level

    Usage:
        with agent_execution_context("CompletenessAgent", {"theorem": label}):
            result = self.agent(**kwargs)
    """
    logger = logging.getLogger(__name__)
    start_time = time.perf_counter()

    context_msg = f"▶ {agent_name} starting"
    if extra:
        context_msg += f" | {format_agent_context(extra)}"
    logger.log(level, context_msg)

    try:
        yield
        elapsed = time.perf_counter() - start_time
        logger.log(level, f"✓ {agent_name} completed in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"✗ {agent_name} failed after {elapsed:.2f}s: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise


def log_dspy_prediction(
    logger: logging.Logger,
    prediction: dspy.Prediction,
    agent_name: str,
    level: int = logging.DEBUG,
):
    """Log key fields from a DSPy prediction.

    Args:
        logger: Logger instance to use
        prediction: DSPy prediction object
        agent_name: Name of agent that produced prediction
        level: Logging level
    """
    output_fields = [k for k in dir(prediction) if not k.startswith("_")]
    logger.log(level, f"{agent_name} prediction fields: {output_fields}")


def format_agent_context(context: dict[str, Any]) -> str:
    """Format context dictionary for structured logging.

    Args:
        context: Dictionary of context fields

    Returns:
        Formatted string like "key1=value1, key2=value2"
    """
    return ", ".join(f"{k}={_safe_repr(v)}" for k, v in context.items())


def _safe_repr(obj: Any, max_len: int = 100) -> str:
    """Safe representation of object, truncating if too long."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            return r[: max_len - 3] + "..."
        return r
    except Exception:
        return f"<{type(obj).__name__}>"
