"""Exact pooled native one-dimensional alive-law transport, without tail quantization.

Each pooled atom has its original 1/(R*N) alive-submass before normalization.
Independent whole trajectories are the resampling units. All54 fresh laws have
point curves; the six d2,N32 laws also have 200 bootstrap intervals by default.
"""

import argparse
from collections import defaultdict
import json
from operator import itemgetter
from pathlib import Path

from chapter11_completion_ledger import load_archive, packed, sha
import numpy as np


def uniform_transport(x, y):
    if len(x) == 0 or len(y) == 0:
        return None
    x, y = np.sort(x), np.sort(y)
    edges = np.unique(
        np.concatenate((np.arange(len(x) + 1) / len(x), np.arange(len(y) + 1) / len(y)))
    )
    midpoint = (edges[:-1] + edges[1:]) / 2
    a = np.minimum((midpoint * len(x)).astype(int), len(x) - 1)
    b = np.minimum((midpoint * len(y)).astype(int), len(y) - 1)
    return float(np.sum(np.diff(edges) * (x[a] - y[b]) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    root = args.dataset.resolve()
    index = json.loads((root / "archive-index.json").read_text())
    if index["status"] != "complete":
        message = "Complete immutable native operand dataset required"
        raise ValueError(message)
    groups = defaultdict(list)
    for entry in index["entries"]:
        if entry["tag"].startswith("native-"):
            row = load_archive(root, entry)
            if row["group"].startswith("native-trajectories-v1/"):
                groups[row["group"]].append(row)
    rng = np.random.default_rng(11190317)
    curves = []
    manifest = []
    for key in sorted(groups):
        if not key.endswith("/left"):
            continue
        right_key = key[:-4] + "right"
        left = sorted(groups[key], key=itemgetter("seed"))
        right = sorted(groups[right_key], key=itemgetter("seed"))
        if len(left) != 24 or len(right) != 24:
            message = "Exactly24 independent trajectories per fresh initial law required"
            raise ValueError(message)
        n, d, h = left[0]["N"], left[0]["d"], left[0]["h"]
        if any(r["N"] != n or r["d"] != d or r["h"] != h for r in left + right):
            message = "Population/dimension/physical timestep mismatch"
            raise ValueError(message)

        def coordinates(rows):
            result = []
            for row in rows:
                coords = [np.array(row["initial"]["projected_coordinates"], dtype=float)] + [
                    np.array(o["observation"]["projected_coordinates"], dtype=float)
                    for o in row["observations"]
                ]
                if len(coords) < 33:
                    if row["terminal"]["mass"] != 0:
                        message = "Only an extinct trajectory admits cemetery continuation"
                        raise ValueError(message)
                    coords += [np.array([], dtype=float)] * (33 - len(coords))
                if len(coords) != 33:
                    message = "Fresh deterministic design is exactly32updates"
                    raise ValueError(message)
                result.append(coords)
            return result

        a, b = coordinates(left), coordinates(right)
        point = []
        for time in range(33):
            x = np.concatenate([r[time] for r in a])
            y = np.concatenate([r[time] for r in b])
            point.append({
                "time": time * h,
                "left_alive_submass": len(x) / (24 * n),
                "right_alive_submass": len(y) / (24 * n),
                "left_alive_atoms": len(x),
                "right_alive_atoms": len(y),
                "exact_normalized_projected_W2_squared": uniform_transport(x, y),
            })
        ia = ib = None
        draws = None
        intervals = None
        if d == 2 and n == 32:
            ia = rng.integers(0, len(left), (args.bootstrap, len(left)))
            ib = rng.integers(0, len(right), (args.bootstrap, len(right)))
            draws = np.empty((args.bootstrap, 33))
            for replicate in range(args.bootstrap):
                for time in range(33):
                    x = np.concatenate([a[i][time] for i in ia[replicate]])
                    y = np.concatenate([b[i][time] for i in ib[replicate]])
                    value = uniform_transport(x, y)
                    draws[replicate, time] = np.nan if value is None else value
            finite = np.isfinite(draws).all(1)
            intervals = (
                np.quantile(draws[finite], [0.025, 0.975], axis=0).T.tolist()
                if sum(finite)
                else None
            )
        values = np.array([p["exact_normalized_projected_W2_squared"] for p in point])
        fit = np.flatnonzero((values > 1e-14) & (np.arange(33) > 0))
        rate = (
            -float(
                np.polyfit(np.array([p["time"] for p in point])[fit], np.log(values[fit]), 1)[0]
            )
            if len(fit) >= 4
            else None
        )
        record = {
            "left_group": key,
            "right_group": right_key,
            "N": n,
            "d": d,
            "physical_times": [p["time"] for p in point],
            "point_curve": point,
            "pointwise_whole_trajectory_bootstrap_95_intervals": intervals,
            "bootstrap_replicates": args.bootstrap if intervals else 0,
            "endpoint_decreased": bool(values[-1] < values[0]),
            "physical_time_log_linear_decay_rate": rate,
            "rate_fit_times": [p["time"] for i, p in enumerate(point) if i in fit],
            "theorem_prediction": None,
            "scope": "Exact first-coordinate monotone transport of pooled alive submeasure realizations normalized by their own masses. Whole complete native trajectories, not walkers, are independent. This finite-horizon Monte Carlo law contrast retains its sampling floor; it does not establish full ambient W2, continuous-law KL, native QSD or an LSI.",
        }
        saved = {
            "result": record,
            "left_seeds": [r["seed"] for r in left],
            "right_seeds": [r["seed"] for r in right],
            "left_raw_projected_alive_coordinates_by_trajectory_time": [
                [x.tolist() for x in r] for r in a
            ],
            "right_raw_projected_alive_coordinates_by_trajectory_time": [
                [x.tolist() for x in r] for r in b
            ],
            "left_whole_trajectory_resampling_indices": ia.tolist() if ia is not None else None,
            "right_whole_trajectory_resampling_indices": ib.tolist() if ib is not None else None,
            "bootstrap_exact_transport_curves": [
                [None if not np.isfinite(x) else float(x) for x in r] for r in draws
            ]
            if draws is not None
            else None,
        }
        artifact = packed(args.output / f"pooled-transport-{len(curves):03}.json.gz", saved)
        manifest.append(artifact)
        record["archive"] = artifact
        curves.append(record)
    result = {
        "chapter": 11,
        "fresh_native_laws": len(curves),
        "bootstrap_laws": sum(c["bootstrap_replicates"] > 0 for c in curves),
        "source_index_SHA256": sha(root / "archive-index.json"),
        "helper_source_SHA256": sha(args.output / "source.py"),
        "curves": curves,
        "archive_manifest": manifest,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            "point_curves": len(curves),
            "bootstrap_curves": result["bootstrap_laws"],
            "endpoints_decreased": sum(c["endpoint_decreased"] for c in curves),
        })
    )


if __name__ == "__main__":
    main()
