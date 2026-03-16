"""Small benchmark harness for Dreamer control experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch


def _parse_task_spec(spec: str) -> tuple[str, str]:
    """Parse ``domain:task`` task specs."""
    domain, sep, task = spec.partition(":")
    if not sep or not domain or not task:
        msg = f"Invalid task spec {spec!r}; expected domain:task"
        raise ValueError(msg)
    return domain, task


def _parse_train_log_summary(log_path: Path) -> dict[str, float | None]:
    """Extract the final training/eval summary from a Dreamer log."""
    rew20 = None
    eval_reward = None
    eval_std = None
    eval_len = None
    line_rew20 = re.compile(r"rew_20=([-+0-9.]+)")
    line_eval = re.compile(r"EVAL  reward=([-+0-9.]+) \+/- ([-+0-9.]+)  len=([0-9]+)")
    if not log_path.exists():
        return {
            "final_rew20": None,
            "final_eval_reward": None,
            "final_eval_std": None,
            "final_eval_len": None,
        }
    for line in log_path.read_text().splitlines():
        rew20_match = line_rew20.search(line)
        if rew20_match:
            rew20 = float(rew20_match.group(1))
        eval_match = line_eval.search(line)
        if eval_match:
            eval_reward = float(eval_match.group(1))
            eval_std = float(eval_match.group(2))
            eval_len = float(eval_match.group(3))
    return {
        "final_rew20": rew20,
        "final_eval_reward": eval_reward,
        "final_eval_std": eval_std,
        "final_eval_len": eval_len,
    }


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the latest epoch checkpoint in a run directory, if any."""
    epoch_ckpts = sorted(run_dir.glob("epoch_*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    best_path = run_dir / "best.pt"
    if best_path.exists():
        return best_path
    return None


def _load_checkpoint_metrics(path: Path | None) -> dict[str, Any]:
    """Load checkpoint metrics safely."""
    if path is None or not path.exists():
        return {}
    state = torch.load(path, map_location="cpu")
    metrics = state.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    """Summarize a Dreamer training run from checkpoints and logs."""
    best_metrics = _load_checkpoint_metrics(run_dir / "best.pt")
    final_metrics = _load_checkpoint_metrics(_find_latest_checkpoint(run_dir))
    log_summary = _parse_train_log_summary(run_dir / "train.log")
    return {
        "run_dir": str(run_dir),
        "best_eval_reward": best_metrics.get("eval/reward_mean"),
        "best_eval_std": best_metrics.get("eval/reward_std"),
        "final_eval_reward": log_summary["final_eval_reward"],
        "final_eval_std": log_summary["final_eval_std"],
        "final_eval_len": log_summary["final_eval_len"],
        "final_rew20": log_summary["final_rew20"],
        "final_exact_covector_norm": final_metrics.get("critic/exact_covector_norm_mean"),
        "final_on_policy_exact_covector_norm": final_metrics.get(
            "critic/on_policy/exact_covector_norm_mean",
        ),
        "final_exact_increment_abs_err": final_metrics.get("critic/exact_increment_abs_err"),
        "final_on_policy_calibration_ratio": final_metrics.get("critic/on_policy/calibration_ratio"),
        "final_return_trust": final_metrics.get("actor/return_trust_used"),
        "final_return_gate": final_metrics.get("actor/return_gate"),
        "final_exact_control_gate": final_metrics.get("actor/exact_control_gate"),
        "final_hodge_conservative_exact": final_metrics.get(
            "actor/policy_hodge_conservative_exact_mean",
        ),
        "final_force_rel_err": final_metrics.get("actor/policy_force_rel_err_mean"),
    }


def _build_command(
    *,
    domain: str,
    task: str,
    seed: int,
    device: str,
    total_epochs: int,
    checkpoint_dir: Path,
    extra_args: list[str],
) -> list[str]:
    """Build the trainer subprocess command."""
    return [
        sys.executable,
        "-m",
        "fragile.learning.rl.train_dreamer",
        "--domain",
        domain,
        "--task",
        task,
        "--seed",
        str(seed),
        "--device",
        device,
        "--total_epochs",
        str(total_epochs),
        "--checkpoint_dir",
        str(checkpoint_dir),
        *extra_args,
    ]


def _run_one(
    *,
    repo_root: Path,
    domain: str,
    task: str,
    seed: int,
    device: str,
    total_epochs: int,
    output_root: Path,
    extra_args: list[str],
) -> dict[str, Any]:
    """Run one Dreamer benchmark job and return its summary."""
    run_dir = output_root / f"{domain}_{task}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "benchmark.log"
    cmd = _build_command(
        domain=domain,
        task=task,
        seed=seed,
        device=device,
        total_epochs=total_epochs,
        checkpoint_dir=run_dir,
        extra_args=extra_args,
    )
    with log_path.open("w") as log_file:
        subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    summary = _summarize_run(run_dir)
    summary["domain"] = domain
    summary["task"] = task
    summary["seed"] = seed
    return summary


def _write_summaries(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    """Write benchmark summaries to JSON and CSV."""
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    json_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    if not summaries:
        csv_path.write_text("")
        return
    fieldnames = list(summaries[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Dreamer control tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["cartpole:balance", "cartpole:swingup"],
        help="Task specs of the form domain:task.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to benchmark.",
    )
    parser.add_argument("--device", default="cpu", help="Trainer device.")
    parser.add_argument("--total-epochs", type=int, default=20, help="Epochs per run.")
    parser.add_argument(
        "--output-root",
        default="outputs/dreamer/control_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--trainer-arg",
        action="append",
        default=[],
        help="Extra arg to forward to the trainer. Repeat for multiple args.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    summaries: list[dict[str, Any]] = []
    for task_spec in args.tasks:
        domain, task = _parse_task_spec(task_spec)
        for seed in args.seeds:
            summary = _run_one(
                repo_root=repo_root,
                domain=domain,
                task=task,
                seed=seed,
                device=args.device,
                total_epochs=args.total_epochs,
                output_root=output_root,
                extra_args=args.trainer_arg,
            )
            summaries.append(summary)
            print(
                f"{domain}:{task} seed={seed} "
                f"best_eval={summary['best_eval_reward']} final_eval={summary['final_eval_reward']}",
            )
    _write_summaries(output_root, summaries)


if __name__ == "__main__":
    main()
