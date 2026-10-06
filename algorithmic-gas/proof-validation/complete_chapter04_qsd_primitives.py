"""Complete finite-N primitive assembly and actual native conditional-survival audit.

Uses the repository's existing CGD.8--18 evaluator; no pair-error trace supplies
a QSD, eigenfunction, or population-uniform mixing rate. Gaussian tails stay full.
"""

import argparse
from dataclasses import asdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys

from audit_local_structural_kinetics import AllStepsReader, checked_payload, load_skipper, SKIPPER
from complete_chapter04_estimates import comparison, require, sha, write_json
import numpy as np
from read_native_cbor import FirstStepReader
from scipy.special import gammainc, log_ndtr


class ReferenceStagesReader(AllStepsReader):
    """Keep complete native steps with just the proof's consumed stage fields."""

    def value(self, depth=0, keep=True):
        if not keep:
            return super().value(depth, keep=False)
        if self.position < len(self.data) and self.data[self.position] >> 5 == 5:
            require(depth <= 64, "CBOR native nesting limit exceeded")
            _, _, payload = self.header()
            count = payload if isinstance(payload, int) else int.from_bytes(payload, "big")
            result = {}
            for _ in range(count):
                key = self.value(depth + 1)
                retain = key not in {
                    "influences",
                    "graph",
                    "before",
                    "donor_fitness",
                    "noise",
                    "elite_injection",
                    "rewards",
                    "observations",
                    "final_population",
                    "report",
                    "generations",
                }
                if key in {"fields", "validity"} and "stage" in result:
                    retain = result["stage"] in {"literal_clone", "B1_input", "B1", "terminal"}
                if key == "values" and "field" in result:
                    retain = result["field"] in {"executed_noise", "potential_force"}
                item = self.value(depth + 1, keep=retain)
                if retain:
                    result[key] = item
            return result
        return FirstStepReader.value(self, depth, keep)


def source_evaluator(repository):
    path = repository / "proof-validation/reference/qsd_certificate.py"
    spec = importlib.util.spec_from_file_location("chapter04_existing_qsd_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def logadd(a, b):
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def analytic_primitive(config, n, d, row):
    kinetic, transform = config["kinetic"], config["clone_transform"]
    integrator = kinetic["integrator"]
    require(
        integrator["kind"] == "baoab" and kinetic["boundary_schedule"] == "end_of_step",
        "Kernel must have the actual terminal-only BAOAB stages",
    )
    require(
        config["qft"]["curl"] is None
        and config["qft"]["graph_viscosity"] is None
        and not config["qft"]["innovation_shifts"],
        "Additional providers are outside CGD regime",
    )
    viscosity = config["qft"]["viscosity"]
    require(viscosity["row_normalized"] == row, "Recorded normalization differs")
    for noise in (kinetic["noise"], transform["jitter"]):
        require(
            noise["innovation"] == "gaussian"
            and noise["geometry"]["kind"] == "isotropic"
            and noise["geometry"]["scale"]["values"] == [1.0],
            "Unit isotropic Gaussian provider required",
        )
    require(
        config["boundary"]["kind"] == "absorbing_box", "This is the killed terminal-box regime"
    )
    boundary = config["boundary"]["domain"]
    ld = boundary["upper"][0]
    require(
        boundary["upper"] == [ld] * d and boundary["lower"] == [-ld] * d, "Symmetric cube required"
    )
    for name in ("distance_donors", "cloning_donors"):
        provider = config[name]
        require(
            provider["law"] == "independent"
            and provider["history_window"] == 0
            and provider["count"] == 1,
            "Current independent canonical companions required",
        )
    require(
        transform["collision_rotation"] == "haar"
        and config["clone_decision"]["revival_from_companion"],
        "Canonical component rotations/revival required",
    )
    h, gamma = integrator["dt"], integrator["friction"]
    t, c = h / 2, math.exp(-gamma * h)
    q2 = -math.expm1(-2 * gamma * h) / (2 * gamma) if gamma else h
    q, s = math.sqrt(q2), kinetic["position_diffusion"] * math.sqrt(h)
    sj, cap = transform["jitter_amplitude"], kinetic["velocity_cap"]
    vc, rd = (1 + 2 * abs(transform["restitution"])) * cap, math.sqrt(d) * ld
    nu, rho = viscosity["coefficient"], viscosity["bandwidth"]
    require(q > 0 and s > 0 and (sj > 0 or n == 1), "Positive primitive noises required")
    lf, bf = 1.0, 0.0  # Configured harmonic source is checked at every actual B1/B2 stage.
    kappa = 1 - t**2 * lf - (2 if row else 1) * t * nu
    require(kappa > 0, "Global B2 coercivity predicate fails")
    j0 = sj
    logp0 = math.log(float(gammainc(d / 2, 0.5)))

    def stages(j):
        b0 = rd + j
        b1 = (1 + 2 * t * nu) * vc + t * (bf + lf * b0)
        return b0, b1, b0 + t * b1

    _, b1, _ = stages(j0)
    a = ld + j0 + t * (1 + c) * b1
    sigma = math.hypot(t * q, s)
    lu, ll = float(log_ndtr((ld - a) / sigma)), float(log_ndtr((-ld - a) / sigma))
    coordinate_log_floor = lu + math.log(-math.expm1(ll - lu))
    loga = logp0 + d * coordinate_log_floor
    radius = math.sqrt(2 * d * (math.log(12 * d * n) - 2 * loga))
    j, g = sj * radius, radius
    ell = math.exp(-0.5) / rho
    cx = 16 * nu * vc * (rd + j) / rho**2 if row and n > 2 else 0 if row else 4 * nu * vc * ell
    beta, kb = t**2 * (lf + cx), 1 - (2 if row else 1) * t * nu
    require(beta < 1 and kb > 0, "Two-update derivative/determinant predicates fail")
    _, b1g, bxg = stages(j)
    z0 = c * b1g + q * g
    r2 = bxg + t * z0
    hout, rout = (1 + 2 * t * nu) * z0 + t * (bf + lf * r2), r2 + s * g

    def lower_density(r, velocity):
        _, first_velocity, first_position = stages(j0)
        z = (velocity + t * (bf + lf * first_position)) / kappa * (1 if row else math.sqrt(n))
        second_position = first_position + t * z
        logd = math.log(1 + t**2 * lf + 2 * t * nu)
        if nu and not (row and n == 2):
            extra = math.log((8 if row else 4) * t**2 * nu * ell * z)
            if row:
                extra += 2 * second_position**2 / rho**2
            logd = logadd(logd, extra)
        m = n * d
        return (
            n * logp0
            - m * math.log(2 * math.pi * q * s)
            - m * (0.5 * math.log(n) + logd)
            - n * (z + c * first_velocity) ** 2 / (2 * q2)
            - n * (math.sqrt(d) * r + first_position + t * z) ** 2 / (2 * s**2)
        )

    m = n * d
    target_volume = m * math.log(ld) + n * (
        d / 2 * math.log(math.pi) + d * math.log(cap / 2) - math.lgamma(1 + d / 2)
    )
    logeps = min(-math.log(2), lower_density(ld / 2, cap) + target_volume)
    logm_density = (
        n * math.log(4 * n**2)
        - m * math.log(2 * math.pi * min(s, sj) * q)
        - m * (math.log1p(-beta) + math.log(kb))
    )
    logl = lower_density(rout, hout)
    logmf = min(0.0, logl + 2 * loga - math.log(2) - logm_density)
    logdelta = logeps + logmf
    return {
        "h": h,
        "t": t,
        "c": c,
        "q": q,
        "s": s,
        "sigma_h": sigma,
        "V_c": vc,
        "R_D": rd,
        "J_0": j0,
        "A": a,
        "B_1": b1,
        "log_p0": logp0,
        "log_coordinate_floor": coordinate_log_floor,
        "coercivity_margin": kappa,
        "drift_margin": 1 - beta,
        "beta_F": beta,
        "log_survival_lower_bound": loga,
        "jitter_event_radius": j,
        "noise_event_radius": g,
        "output_position_radius": rout,
        "precap_velocity_radius": hout,
        "log_minorization_lower_bound": logeps,
        "log_two_step_density_upper_bound": logm_density,
        "log_output_density_lower_bound": logl,
        "log_eigenfunction_min_lower_bound": logmf,
        "log_doob_minorization_lower_bound": logdelta,
        "log_tail_probability_upper_bound": math.log(6 * d * n) - radius**2 / (2 * d),
        "phase_diameter_squared": 4 * d * ld**2 + 4 * cap**2,
        "force_lipschitz": lf,
        "force_at_origin": bf,
    }


def stage(step, name, key):
    return next(s["fields"][key]["values"] for s in step["stages"] if s["stage"] == name)


def evaluation(step, stage_name, key):
    return next(
        f["values"]
        for f in step["field_evaluations"]
        if f["stage"] == stage_name and f["field"] == key
    )


def viscous_position_jacobian(positions, velocities, viscosity, bandwidth, row):
    """Exact Gaussian-provider derivative, with recipient velocity held fixed."""
    x, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    n, d = x.shape
    require(v.shape == x.shape and n >= 2, "Dense Jacobian dimension mismatch")
    displacement = x[:, None, :] - x[None, :, :]
    relative_velocity = v[None, :, :] - v[:, None, :]
    weights = np.exp(-np.sum(displacement**2, axis=2) / (2 * bandwidth**2))
    np.fill_diagonal(weights, 0)
    if row:
        weights /= np.sum(weights, axis=1, keepdims=True)
        mean_velocity = np.sum(weights[:, :, None] * relative_velocity, axis=1)
        relative_velocity -= mean_velocity[:, None, :]
    else:
        weights /= n
    blocks = (
        viscosity
        / bandwidth**2
        * weights[:, :, None, None]
        * relative_velocity[:, :, :, None]
        * displacement[:, :, None, :]
    )
    diagonals = -np.sum(blocks, axis=1)
    blocks[np.arange(n), np.arange(n)] = diagonals
    return blocks.transpose(0, 2, 1, 3).reshape(n * d, n * d)


def conditional_survival(step, primitive, n, d):
    x, literal = stage(step, "B1_input", "positions"), stage(step, "literal_clone", "positions")
    v, out = stage(step, "B1", "velocities"), stage(step, "terminal", "positions")
    ou, diffusion = (
        evaluation(step, "O", "executed_noise"),
        evaluation(step, "position_diffusion", "executed_noise"),
    )
    for values in (x, literal, v, out, ou, diffusion):
        require(
            len(values) == n * d and all(map(math.isfinite, values)),
            "Native stage dimension/finiteness failed",
        )
    h, t, c, sigma = primitive["h"], primitive["t"], primitive["c"], primitive["sigma_h"]
    rows = []
    for i in range(n):
        indices = range(i * d, (i + 1) * d)
        mean = [x[a] + t * (1 + c) * v[a] for a in indices]
        jitter = math.sqrt(math.fsum((x[a] - literal[a]) ** 2 for a in indices))
        eligible = jitter <= primitive["J_0"]
        logprob = 0.0
        for mu in mean:
            lo, hi = float(log_ndtr((-2 - mu) / sigma)), float(log_ndtr((2 - mu) / sigma))
            logprob += hi + math.log(-math.expm1(lo - hi))
        identity = max(
            abs(
                out[a]
                - (
                    x[a]
                    + t * (1 + c) * v[a]
                    + t * primitive["q"] * ou[a]
                    + primitive["s"] * diffusion[a]
                )
            )
            for a in indices
        )
        # On the realized row-jitter event the conditional Gaussian probability
        # is >= a_F/p0; the unconditional a_F follows by event integration.
        logbound = primitive["log_survival_lower_bound"] - primitive["log_p0"]
        rows.append({
            "row": i,
            "jitter_norm": jitter,
            "eligible_jitter_event": eligible,
            "mean": mean,
            "mean_envelope": primitive["A"],
            "log_conditional_alive_probability": logprob,
            "log_event_conditional_floor": logbound,
            "survival_floor_residual": logbound - logprob if eligible else None,
            "position_noise_identity_absolute_residual": identity,
            "new_native_steps": 0,
            "h": h,
        })
    return rows


def run(dataset, output, repository, skipper=None):
    require(not output.exists(), "Use a fresh immutable derived directory")
    output.mkdir(parents=True)
    if skipper:
        load_skipper(skipper)
    require(SKIPPER, "Pass the already tested native CBOR skipper shared library")
    evaluator, evaluator_path = source_evaluator(repository)
    report = json.loads((dataset / "chapter04-report.json").read_text())
    entries = {
        e["path"]: e
        for e in (
            json.loads(s)
            for s in (dataset / "archive-journal.jsonl").read_text().splitlines()
            if s
        )
    }
    comparisons, references, verified = [], [], []
    for reference in report["references"]:
        n, d, row, config = (
            reference["N"],
            reference["d"],
            reference["row_normalized"],
            reference["native_config"],
        )
        mode = "row" if row else "count"
        primitive = analytic_primitive(config, n, d, row)
        kinetic, clone, viscosity = (
            config["kinetic"],
            config["clone_transform"],
            config["qft"]["viscosity"],
        )
        certificate = asdict(
            evaluator.viscous_qsd_primitive_certificate(
                n_walkers=n,
                dimension=d,
                timestep=primitive["h"],
                viscosity=viscosity["coefficient"],
                bandwidth=viscosity["bandwidth"],
                force_lipschitz=1.0,
                force_at_origin=0.0,
                terminal_half_width=2.0,
                clone_jitter=clone["jitter_amplitude"],
                restitution=clone["restitution"],
                velocity_cap=kinetic["velocity_cap"],
                ou_coefficient=primitive["c"],
                ou_amplitude=primitive["q"],
                position_amplitude=primitive["s"],
                row_normalized=row,
            )
        )
        require(
            not certificate["limitations"], "Existing same-kernel certificate fails its premises"
        )
        for name, observed in certificate.items():
            if name in primitive:
                comparisons.append(
                    comparison(
                        f"{mode}:primitive-formula:{name}",
                        [
                            "def-w2-finite-population-qsd-regime",
                            "cor-w2-reference-fitness-degeneracy",
                        ],
                        observed,
                        primitive[name],
                        "equal",
                        kind="independent_scalar_formula_reconstruction",
                        hypotheses="Actual configured N200d3 harmonic force and native count/row viscosity; binary64 logarithmic evaluations, not interval enclosures or observed QSD eigenfunction values",
                    )
                )
        comparisons.append(
            comparison(
                f"{mode}:discarded-event-budget",
                ["def-w2-finite-population-qsd-regime"],
                primitive["log_tail_probability_upper_bound"],
                2 * primitive["log_survival_lower_bound"] - math.log(2),
                kind="analytic_bound_evaluation",
                hypotheses="Full unbounded standard Gaussian draws; coordinate and latent-event union bounds, no truncation",
            )
        )
        ledger, steps, max_force_residual = [], 0, 0.0
        for source in reference["raw_archives"]:
            relative = source["archive"]
            payload = checked_payload(dataset, entries[relative])
            decoder = ReferenceStagesReader(payload)
            archive = decoder.value()
            require(decoder.position == len(payload), "Trailing native archive bytes")
            require(
                len(archive["steps"]) == source["steps"][1] - source["steps"][0],
                "Incomplete native reference chunk",
            )
            verified.append({
                "path": str(dataset / relative),
                "sha256": entries[relative]["sha256"],
                "steps": len(archive["steps"]),
                "decoded_bytes": len(payload),
            })
            for step in archive["steps"]:
                positions = stage(step, "B1_input", "positions")
                if steps == 0:
                    velocity = stage(step, "B1_input", "velocities")
                    xarray = np.asarray(positions).reshape(n, d)
                    varray = np.asarray(velocity).reshape(n, d)
                    radius = primitive["R_D"] + primitive["jitter_event_radius"]
                    vc = primitive["V_c"]
                    require(
                        np.max(np.linalg.norm(xarray, axis=1)) <= radius
                        and np.max(np.linalg.norm(varray, axis=1)) <= vc,
                        "Actual first-drift probe outside declared event/collision velocity envelope",
                    )
                    derivative = viscous_position_jacobian(
                        xarray, varray, viscosity["coefficient"], viscosity["bandwidth"], row
                    )
                    matrix = np.eye(n * d) + primitive["t"] ** 2 * (derivative - np.eye(n * d))
                    sign, logdet = np.linalg.slogdet(matrix)
                    require(
                        sign > 0 and math.isfinite(logdet),
                        "Native first-drift probe has nonpositive determinant",
                    )
                    sharper = primitive["t"] ** 2 * (
                        1
                        + (
                            (
                                3
                                * math.sqrt(3)
                                / 2
                                * viscosity["coefficient"]
                                * vc
                                * radius
                                / viscosity["bandwidth"] ** 2
                            )
                            if row
                            else 2
                            * viscosity["coefficient"]
                            * vc
                            * math.exp(-0.5)
                            / viscosity["bandwidth"]
                        )
                    )
                    comparisons.append(
                        comparison(
                            f"{mode}:actual-first-drift-determinant",
                            ["cor-w2-sharp-first-drift-margin"],
                            float(logdet),
                            n * d * math.log1p(-sharper),
                            "lower",
                            kind="actual_native_stage_analytic_Jacobian",
                            sharper_beta=sharper,
                            hypotheses="Prepared B1 native stage within proved radius, frozen collision velocities bounded by Vc. Exact provider Jacobian; binary64 LU determinant, not interval enclosure. Count uses normalized Hilbert norm, row uses maximum-row norm.",
                        )
                    )
                force = evaluation(step, "B1", "potential_force")
                max_force_residual = max(
                    max_force_residual,
                    *(abs(a + b) for a, b in zip(positions, force, strict=True)),
                )
                rows = conditional_survival(step, primitive, n, d)
                ledger.append({"source_archive": relative, "epoch": step["epoch"], "rows": rows})
                steps += 1
        require(steps == reference["steps"], "Reference trajectory coverage incomplete")
        eligible = [r for s in ledger for r in s["rows"] if r["eligible_jitter_event"]]
        comparisons.extend([
            comparison(
                f"{mode}:actual-harmonic-force",
                ["def-w2-finite-population-qsd-regime"],
                max_force_residual,
                0,
                "equal",
                individual_checks=steps * n * d,
            ),
            comparison(
                f"{mode}:actual-conditional-tagged-survival",
                ["def-w2-finite-population-qsd-regime"],
                max(r["survival_floor_residual"] for r in eligible),
                0,
                individual_checks=len(eligible),
                hypotheses="Recorded preparation/collision/B1 stages; only actual row-jitter event eligibility used. Gaussian conditional final-position probability, not observed survival count or conditioned swarm law. All ineligible rows retained.",
            ),
            comparison(
                f"{mode}:actual-final-position-two-noise-identity",
                ["def-w2-finite-population-qsd-regime"],
                max(
                    r["position_noise_identity_absolute_residual"]
                    for s in ledger
                    for r in s["rows"]
                ),
                0,
                "equal",
                individual_checks=steps * n,
                hypotheses="Actual continuous native trajectory, both unbounded independent OU and final-position innovations; every row/time checked without SEM",
            ),
        ])
        path = output / f"{mode}-conditional-survival-ledger.json.gz"
        path.write_bytes(gzip.compress(json.dumps(ledger, allow_nan=False).encode(), mtime=0))
        logdelta, logmf = (
            primitive["log_doob_minorization_lower_bound"],
            primitive["log_eigenfunction_min_lower_bound"],
        )
        # delta <= 1/2, so delta/h <= -log(1-delta)/h <= 2delta/h.
        # Store these log bounds; exp(logdelta) underflows and rho rounds to 1.
        require(logdelta <= -math.log(2), "Log-rate enclosure requires delta<=1/2")
        rates = {
            "log_TV_rate_lower": logdelta - math.log(primitive["h"]),
            "log_TV_rate_upper": logdelta - math.log(primitive["h"]) + math.log(2),
            "log_W2_rate_lower": logdelta - math.log(primitive["h"]) - math.log(2),
            "log_W2_rate_upper": logdelta - math.log(primitive["h"]),
            "log_single_initial_TV_prefactor": -logmf,
            "log_initial_sensitive_TV_prefactor": -2 * logmf,
            "log_first_nontrivial_single_initial_step_threshold_upper": math.log(-logmf)
            - logdelta,
            "finite_N": n,
            "N_uniform_mixing_rate": False,
            "physical_alive_error_diameter_N_independent": True,
            "stationary_QSD_observable_comparison": "Not measured: no supplied physical QSD law or eigenfunction; native trajectory cannot identify it from pair cost",
            "finite_horizon_TV_upper": 1.0,
            "finite_horizon_W2_squared_upper": primitive["phase_diameter_squared"],
            "finite_horizon_steps": steps,
        }
        references.append({
            "mode": mode,
            "N": n,
            "d": d,
            "native_config": config,
            "seed": reference["seed"],
            "primitive": primitive,
            "existing_evaluator_certificate": certificate,
            "rates": rates,
            "eligible_tagged_rows": len(eligible),
            "all_tagged_rows": steps * n,
            "conditional_survival_ledger": str(path),
            "ledger_sha256": sha(path),
            "ledger_decoded_bytes": len(gzip.decompress(path.read_bytes())),
        })
        print(
            f"Completed {mode} reference: {steps} steps, {len(eligible)} qualified tagged rows, log delta={logdelta:.6g}",
            flush=True,
        )
    result = {
        "chapter": 4,
        "new_native_steps": 0,
        "source_report": str(dataset / "chapter04-report.json"),
        "source_report_sha256": sha(dataset / "chapter04-report.json"),
        "helper_sha256": sha(Path(__file__)),
        "existing_evaluator_path": str(evaluator_path),
        "existing_evaluator_sha256": sha(evaluator_path),
        "source_formal_path": "docs/source/2_fractal_gas/convergence_program/18_coupled_gas_discharge.md",
        "source_formal_sha256": sha(
            repository
            / "docs/source/2_fractal_gas/convergence_program/18_coupled_gas_discharge.md"
        ),
        "source_reader_sha256": sha(
            Path(__file__).with_name("audit_local_structural_kinetics.py")
        ),
        "skipper_sha256": sha(skipper),
        "native_archive_verification": verified,
        "comparisons": comparisons,
        "references": references,
        "summary": {
            "references": len(references),
            "comparisons": len(comparisons),
            "failed": sum(not c["passed"] for c in comparisons),
            "all_tagged_rows": sum(r["all_tagged_rows"] for r in references),
            "qualified_tagged_rows": sum(r["eligible_tagged_rows"] for r in references),
        },
        "scope": "Same-kernel full primitive constant evaluation plus actual-stage Gaussian conditional survival/noise identities. Finite-N universal rate is an analytic consequence of the proved hypotheses; its astronomical certificate does not resolve stationary-law decay experimentally at 128 steps. No raw or earlier derived artifact overwritten.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cbor-skipper", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.repository, args.cbor_skipper)
    print(json.dumps(result["summary"], indent=2))
    raise SystemExit(bool(result["summary"]["failed"]))


if __name__ == "__main__":
    main()
