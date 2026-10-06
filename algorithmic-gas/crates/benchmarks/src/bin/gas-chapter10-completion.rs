//! Saved Chapter 10 density-law, matrix, normalization and convergence experiments.
use algorithmic_gas_benchmarks::{
    convergence_chapter10_completion::*,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: gas-chapter10-completion EMPTY_OUTPUT".into());
    }
    let mut store = ArchiveStore::new(&args[1])?;
    let mut cases = vec![];
    let mut check_count = 0;
    let mut failed = 0;
    let mut save = |tag: &str,
                    row: Value,
                    store: &mut ArchiveStore|
     -> Result<(), Box<dyn std::error::Error>> {
        let checks = row["checks"]
            .as_array()
            .ok_or("missing comparison operands")?;
        check_count += checks.len();
        failed += checks.iter().filter(|r| r["passed"] != true).count();
        let archive = store.save_json(tag, &row)?;
        cases.push(json!({"id":tag,"archive":archive,"comparisons":checks.len(),"failed":checks.iter().filter(|r|r["passed"]!=true).count(),"constants":row["constants"],"parameters":row["parameters"],"observed_endpoint_H_rate":row["observed_endpoint_H_rate"],"endpoint_entropy_decreased":row["endpoint_entropy_decreased"]}));
        Ok(())
    };
    for k in [0.25, 1., 4.] {
        for theta in [0.25, 1., 4.] {
            for gamma in [0.25, 1., 4.] {
                for (zone, initial) in [
                    ("well", [0.1, 0.2]),
                    ("saddle", [0.5, -0.3]),
                    ("tail", [4., 2.]),
                ] {
                    let tag = format!("gaussian-k{k}-temperature{theta}-friction{gamma}-{zone}");
                    let trajectory = gaussian_trajectory(k, theta, gamma, initial, 0.002, 10_000)?;
                    save(&tag, trajectory, &mut store)?;
                }
            }
        }
    }
    // Global nonquadratic wells, slow zones and tail tilts: original Gibbs density is unbounded.
    for amplitude in [0., 0.2, 1., 3.] {
        for theta in [0.5, 1., 2.] {
            for gamma in [0.5, 2.] {
                for (zone, tilt) in [
                    ("well", [0.1, 0.2, 0.1]),
                    ("saddle", [0.6, -0.3, 0.2]),
                    ("tail", [-0.6, 0.7, -0.3]),
                ] {
                    let tag = format!(
                        "nonquadratic-barrier{amplitude}-temperature{theta}-friction{gamma}-{zone}"
                    );
                    let shift = if zone == "tail" { 4. / theta } else { 0. };
                    let mut fine = density_integrals_shifted(
                        1.,
                        amplitude,
                        2. * std::f64::consts::PI,
                        theta,
                        gamma,
                        tilt,
                        shift,
                        2048,
                    );
                    let coarse = density_integrals_shifted(
                        1.,
                        amplitude,
                        2. * std::f64::consts::PI,
                        theta,
                        gamma,
                        tilt,
                        shift,
                        1024,
                    );
                    let mut refinement = vec![];
                    for field in ["H", "Ix", "Iv", "Ixv", "Phi", "Phi_dot"] {
                        equality(
                            &mut refinement,
                            "quadrature_refinement",
                            field,
                            fine[field].as_f64().unwrap(),
                            coarse[field].as_f64().unwrap(),
                            5e-7,
                        );
                    }
                    let mut checks: Vec<Comparison> =
                        serde_json::from_value(fine["checks"].clone())?;
                    checks.extend(refinement);
                    fine["checks"] = serde_json::to_value(checks)?;
                    fine["coarse_grid_operands"] = coarse;
                    fine["dimension_and_population_scope"] = json!({"d":[1,2,4,8],"N":[1,8,32,128],"coordinate_LSI":fine["constants"]["lsi"],"product_LSI":"same constant by coordinate and particle tensorization; extensive entropy/Fisher divided by N*d"});
                    save(&tag, fine, &mut store)?;
                }
            }
        }
    }
    save(
        "common-target-jump",
        common_target_jump_suite()?,
        &mut store,
    )?;
    save(
        "full-killed-discrete-identities",
        killed_and_discrete_suite()?,
        &mut store,
    )?;
    let mut algebra = vec![];
    let mut operands = vec![];
    for diffusion in [0.01, 0.1, 1., 4.] {
        for gamma in [0.1, 1., 4.] {
            for m in [0., 1., 20., 400.] {
                for lsi in [0.1, 1., 100., 1e12] {
                    let c = kinetic_constants(diffusion, gamma, m, lsi)?;
                    equality(&mut algebra, "constants", "g_min", c.g_min, c.eta, 0.);
                    equality(&mut algebra, "constants", "g_max", c.g_max, 3. * c.eta, 0.);
                    check(
                        &mut algebra,
                        "constants",
                        "positive_determinant",
                        0.,
                        c.a * c.c - c.b * c.b,
                        0.,
                    );
                    check(
                        &mut algebra,
                        "constants",
                        "eta_Hessian_Young_bound",
                        c.eta * (2. * m + c.l_m * c.l_m),
                        diffusion / 2.,
                        1e-13,
                    );
                    check(
                        &mut algebra,
                        "constants",
                        "eta_le_Dhalf",
                        c.eta,
                        diffusion / 2.,
                        1e-13,
                    );
                    for ratio in [0_f64, 0.01, 0.1, 1., 10., 100.] {
                        let ix = ratio;
                        let iv = 1.;
                        let sqrt = ix.sqrt();
                        check(
                            &mut algebra,
                            "constants",
                            "Young_cross",
                            2. * c.eta * c.l_m * sqrt,
                            c.eta * ix + c.eta * c.l_m * c.l_m * iv,
                            1e-12,
                        );
                        let raw = -2. * c.eta * ix
                            - (diffusion + 4. * c.eta * gamma - 2. * c.eta * m) * iv
                            + 2. * c.eta * c.l_m * sqrt;
                        check(
                            &mut algebra,
                            "constants",
                            "matrix_dissipation_envelope",
                            raw,
                            -c.eta * ix - diffusion / 2. * iv,
                            1e-12,
                        );
                    }
                    for omega in [0., 0.1, 1.] {
                        for aj in [0., 1., 1.1, 2.] {
                            let margin = c.eta - 3. * c.eta * omega * (aj - 1_f64).max(0.);
                            if margin > 0. {
                                check(
                                    &mut algebra,
                                    "common_target_jump",
                                    "jump_rate_margin",
                                    margin / (lsi / 2. + c.g_max),
                                    c.rate,
                                    1e-12,
                                );
                            }
                        }
                    }
                    operands.push(json!({"constants":c}));
                }
            }
        }
    }
    // The scalar forcing and defect recurrences are integrated, not endpoint substitutions.
    let mut recurrences = vec![];
    for a in [0.1, 1., 4.] {
        for ci in [0.5, 2.] {
            for b in [0., 0.2, 2.] {
                for e in [0., 0.1] {
                    let r = a / (2. * ci);
                    let floor = (b * b / (2. * a) + e) / r;
                    let phi0 = 3.;
                    for step in 0..=100 {
                        let t = step as f64 / 10.;
                        let solution = (-r * t).exp() * phi0 + floor * (-(-r * t).exp_m1());
                        check(
                            &mut algebra,
                            "forcing_floor",
                            "Young_forcing",
                            b * (solution / ci).sqrt(),
                            a / 2. * solution / ci + b * b / (2. * a),
                            1e-12,
                        );
                        recurrences.push(json!({"kind":"forcing","a":a,"C_I":ci,"B":b,"E":e,"rate":r,"time":t,"Phi":solution,"floor":floor}));
                    }
                }
            }
        }
    }
    for tau in [0.005_f64, 0.01, 0.02, 0.04] {
        for order in [1_f64, 2.] {
            let r = 0.7;
            let k = 0.2;
            let b = 0.1;
            let q = (-r * tau).exp() + k * tau.powf(order + 1.);
            let floor = b * tau.powf(order + 1.) / (1. - q);
            let mut actual = 3.;
            for step in 0_i32..=200 {
                let envelope = q.powi(step) * 3. + floor * (1. - q.powi(step));
                equality(
                    &mut algebra,
                    "discrete_defect",
                    "geometric_recurrence",
                    actual,
                    envelope,
                    2e-12,
                );
                recurrences.push(json!({"kind":"numerical_defect","tau":tau,"p":order,"r":r,"K":k,"B":b,"q":q,"step":step,"Phi":actual,"floor":floor,"scaled_floor":floor/tau.powf(order)}));
                actual = q * actual + b * tau.powf(order + 1.);
            }
        }
    }
    save(
        "all-constants-scalar-recurrences",
        json!({"operands":operands,"recurrences":recurrences,"checks":algebra}),
        &mut store,
    )?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = root.join("crates/benchmarks/src/convergence_chapter10_completion.rs");
    let book = root.join("docs/source/2_fractal_gas/convergence_program/10_kl_hypocoercive.md");
    let report = json!({"schema_version":1,"chapter":10,"cases":cases,"comparisons":check_count,"comparisons_failed":failed,"Gaussian_trajectories":81,"Gaussian_RK4_steps":810000,"nonquadratic_density_cases":72,"native_updates":0,"scope":"Identified continuous kinetic Gaussian density evolution; nonquadratic Gibbs-relative mixed-gradient generator identities; specified full killed jump and discrete QSD kernel; exact scalar forcing/defect recurrences. No empirical atomic KL or native QSD/LSI claim.","provenance":{"module_sha256":sha256_file(&source)?,"chapter_sha256":sha256_file(&book)?,"executable_sha256":sha256_file(&std::env::current_exe()?)?}});
    fs::write(
        store.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    store.finish(if failed == 0 { "complete" } else { "failed" })?;
    let verification = store.verify(true)?;
    fs::write(
        store.root().join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!(
        "{}",
        json!({"comparisons":check_count,"failed":failed,"cases":cases.len(),"archive_verification":verification})
    );
    if failed > 0 {
        return Err("finite Chapter10 comparison failed; operands retained".into());
    }
    Ok(())
}
