"""Measure population rates using independent whole native updates and rooted draws."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 200:
        msg = "At least 200 bootstrap replicates required"
        raise ValueError(msg)
    report = read(args.dataset / "report.json")
    index = read(args.dataset / "archive-index.json")
    if index.get("status") != "complete":
        msg = "A complete native dataset is required"
        raise ValueError(msg)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(202610049999)
    groups = defaultdict(list)
    records = []
    entries = []
    for case in report["cases"]:
        native = read(args.dataset / case["native_operands"])
        rooted = read(args.dataset / case["root_reference"])
        actual = np.asarray([r["output_tests"] for r in native["raw"]], dtype=float)
        roots = np.asarray([r["output_tests"] for r in rooted["rows"]], dtype=float)
        if actual.shape != (case["native_replicas"], 7):
            msg = "Native seed denominator mismatch"
            raise ValueError(msg)
        if roots.shape != (case["root_reference_draws"], 7):
            msg = "Root draw denominator mismatch"
            raise ValueError(msg)
        if rooted["truncation_probability"] != 0:
            msg = "Capacity-selected root laws are not admissible"
            raise ValueError(msg)
        am, av = actual.mean(axis=0), actual.var(axis=0, ddof=1)
        rm, rv = roots.mean(axis=0), roots.var(axis=0, ddof=1)
        # Resample complete seven-observable vectors; preserve within-update dependence.
        ai = rng.integers(len(actual), size=(args.bootstrap, len(actual)))
        ri = rng.integers(len(roots), size=(args.bootstrap, len(roots)))
        ac, rc = actual[ai], roots[ri]
        bam, bav, brm = ac.mean(axis=1), ac.var(axis=1, ddof=1), rc.mean(axis=1)
        bmse = ((ac - brm[:, None, :]) ** 2).mean(axis=1)
        mse = ((actual - rm) ** 2).mean(axis=0)
        raw = {
            "case_id": case["id"],
            "bootstrap_seed": 202610049999,
            "resampling_unit": "Independent whole native update or independent rooted draw",
            "native_bootstrap_indices": ai.tolist(),
            "root_bootstrap_indices": ri.tolist(),
            "bootstrap_native_means": bam.tolist(),
            "bootstrap_native_variances": bav.tolist(),
            "bootstrap_root_means": brm.tolist(),
            "bootstrap_mse": bmse.tolist(),
        }
        path = args.output / (case["id"] + "-bootstrap.json.gz")
        payload = json.dumps(raw, separators=(",", ":"), allow_nan=False).encode()
        with path.open("wb") as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
                zipped.write(payload)
        entries.append({
            "path": path.name,
            "sha256": sha(path),
            "decoded_sha256": hashlib.sha256(payload).hexdigest(),
            "decoded_bytes": len(payload),
        })
        values = {
            "case_id": case["id"],
            "N": case["N"],
            "d": case["d"],
            "profile": case["profile"],
            "native_replicas": len(actual),
            "independent_root_draws": len(roots),
            "native_mean": am.tolist(),
            "native_variance": av.tolist(),
            "reference_mean": rm.tolist(),
            "reference_mean_variance": (rv / len(roots)).tolist(),
            "squared_observed_mean_difference": ((am - rm) ** 2).tolist(),
            "mean_square_error_against_estimated_reference": mse.tolist(),
            "debiased_mse_diagnostic": (mse - rv / len(roots)).tolist(),
            "N_times_native_variance": (case["N"] * av).tolist(),
            "raw_bootstrap": path.name,
        }
        records.append(values)
        groups[case["profile"], case["d"]].append((values, bav, bmse))
    slopes = []
    for (profile, dimension), cases in sorted(groups.items()):
        cases.sort(key=lambda c: c[0]["N"])
        ns = np.asarray([c[0]["N"] for c in cases], dtype=float)
        if ns.tolist() != [8.0, 32.0, 128.0]:
            msg = "The three population sizes are required"
            raise ValueError(msg)
        lx = np.log(ns)
        weights = (lx - lx.mean()) / ((lx - lx.mean()) ** 2).sum()
        for j, name in enumerate(report["test_order"]):
            for metric, key, bootstrap_field, prediction in [
                ("variance", "native_variance", 1, -1.0),
                ("mse", "mean_square_error_against_estimated_reference", 2, -1.0),
            ]:
                y = np.asarray([c[0][key][j] for c in cases])
                boot = np.stack([c[bootstrap_field][:, j] for c in cases], axis=1)
                eligible = np.all(boot > 0, axis=1)
                row = {
                    "profile": profile,
                    "d": dimension,
                    "test_index": j,
                    "test": name,
                    "metric": metric,
                    "N": ns.tolist(),
                    "observed": y.tolist(),
                    "theoretical_upper_rate_exponent": prediction,
                    "valid_bootstrap_draws": int(eligible.sum()),
                    "scope": "Descriptive empirical log slope with pointwise percentile bootstrap interval. An upper O(1/N) estimate need not have slope exactly -1. Root-reference Monte Carlo error can dominate MSE; its variance is retained separately. No analytic theorem pass is inferred from slope.",
                }
                if np.all(y > 0) and eligible.sum() >= args.bootstrap * 0.9:
                    distribution = np.log(boot[eligible]) @ weights
                    row.update({
                        "status": "measured",
                        "slope": float(np.log(y) @ weights),
                        "bootstrap_95_percent_interval": np.quantile(
                            distribution, [0.025, 0.975]
                        ).tolist(),
                        "endpoint_ratio_N128_over_N8": float(y[-1] / y[0]),
                    })
                else:
                    row["status"] = "deterministic_zero_or_insufficient_positive_bootstrap"
                slopes.append(row)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    profiles = sorted({p for p, _ in groups})
    for ax, profile in zip(axes.flat, profiles, strict=False):
        for d in [1, 2, 4]:
            cases = sorted(groups[profile, d], key=lambda c: c[0]["N"])
            ns = np.asarray([c[0]["N"] for c in cases])
            y = np.asarray([c[0]["native_variance"][0] for c in cases])
            ax.loglog(ns, y, "o-", label=f"d={d}")
            ax.loglog(ns, y[0] * ns[0] / ns, ":", alpha=0.4)
        ax.set_title(profile)
        ax.set_xlabel("Swarm size N")
        ax.set_ylabel("Variance of mean sin(x₁)")
        ax.legend()
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.85,
        "64 independent complete native updates per point.\nDotted lines: 1/N anchored at N=8.\nShared collision rotations retained.\nFull marked, probability-normalized observable.",
        fontsize=11,
    )
    fig.savefig(args.output / "native-population-variance.png", dpi=180)
    fig.savefig(args.output / "native-population-variance.pdf")
    plt.close(fig)
    measured = [s for s in slopes if s["status"] == "measured"]
    summary = {
        "native_cases": len(records),
        "new_native_updates": 0,
        "bootstrap_replicates": args.bootstrap,
        "rate_rows": len(slopes),
        "measured_rate_rows": len(measured),
        "rate_rows_with_decreasing_endpoint": sum(
            s["endpoint_ratio_N128_over_N8"] < 1 for s in measured
        ),
        "variance_rate_rows": sum(s["metric"] == "variance" for s in measured),
        "variance_rows_with_decreasing_endpoint": sum(
            s["metric"] == "variance" and s["endpoint_ratio_N128_over_N8"] < 1 for s in measured
        ),
    }
    result = {
        "summary": summary,
        "cases": records,
        "slopes": slopes,
        "input_index_sha256": sha(args.dataset / "archive-index.json"),
        "input_report_sha256": sha(args.dataset / "report.json"),
        "runner_sha256": sha(Path(__file__)),
        "scope": "Bootstrap intervals describe independent finite experiments; rigorous bound comparisons are stored in the Chapter9 runner, separately from these measured rates.",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    # Read every compressed payload back, checking SHA and decoded content.
    for entry in entries:
        path = args.output / entry["path"]
        if sha(path) != entry["sha256"]:
            msg = "Compressed artifact checksum mismatch"
            raise ValueError(msg)
        payload = gzip.decompress(path.read_bytes())
        if hashlib.sha256(payload).hexdigest() != entry["decoded_sha256"]:
            msg = "Decoded artifact checksum mismatch"
            raise ValueError(msg)
        json.loads(payload)
    (args.output / "verification.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "deep_verified": True,
                "artifacts": entries,
                "summary": {
                    "artifacts": len(entries),
                    "decoded_bytes": sum(e["decoded_bytes"] for e in entries),
                },
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Native population scaling",
        "",
        "Each point uses independent complete native updates. Root-reference sampling uncertainty is retained. These are measured rates, not additional theorem pass labels.",
        "",
        "| Profile | d | Observable | Metric | N=128 / N=8 | Log slope | Pointwise bootstrap 95% interval |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in measured:
        lo, hi = row["bootstrap_95_percent_interval"]
        test = row["test"].replace("|", "\\|")
        lines.append(
            f"| {row['profile']} | {row['d']} | {test} | {row['metric']} | {row['endpoint_ratio_N128_over_N8']:.5g} | {row['slope']:.3f} | [{lo:.3f}, {hi:.3f}] |"
        )
    (args.output / "rates.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
