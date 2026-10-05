//! Continuous-reference operands at the exact force and thermostat coefficients of native runs.
use algorithmic_gas_benchmarks::{
    convergence_chapter10_completion::*,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::json;
use std::{fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("gas-chapter10-landscape-matched EMPTY_OUTPUT".into());
    }
    let mut store = ArchiveStore::new(&args[0])?;
    store.save_json("execution-source", &json!({
        "runner":include_str!("gas-chapter10-landscape-matched.rs"),
        "module":include_str!("../convergence_chapter10_completion.rs"),
        "chapter":include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/10_kl_hypocoercive.md"),
        "inventory":include_str!("../../../../proof-validation/chapter10_inventory.json"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,
        "scope":"Exact Rastrigin force U=sum[x²+10(1-cos(2pi x))] and native velocity diffusion D=1/2, temperature theta=D/gamma. Continuous conservative reference only; native cap/cloning/position diffusion/killing are not assigned this Gibbs law."}))?;
    let mut cases = vec![];
    let mut comparisons = 0usize;
    let mut failures = 0usize;
    for gamma in [1., 2.] {
        let theta = 0.5 / gamma;
        for (zone, center, tilt) in [
            ("well", 0., [0.1, 0.2, 0.1]),
            ("between-wells", 0.5, [0.6, -0.3, 0.2]),
            ("tail", 4., [-0.6, 0.7, -0.3]),
        ] {
            let tag =
                format!("nonquadratic-native-force-temperature{theta}-friction{gamma}-{zone}");
            let mut fine = density_integrals_shifted(
                2.,
                10.,
                2. * std::f64::consts::PI,
                theta,
                gamma,
                tilt,
                2. * center / theta,
                4096,
            );
            let coarse = density_integrals_shifted(
                2.,
                10.,
                2. * std::f64::consts::PI,
                theta,
                gamma,
                tilt,
                2. * center / theta,
                2048,
            );
            let mut checks: Vec<Comparison> = serde_json::from_value(fine["checks"].clone())?;
            for field in ["H", "Ix", "Iv", "Ixv", "Phi", "Phi_dot"] {
                equality(
                    &mut checks,
                    "quadrature_refinement",
                    field,
                    fine[field]
                        .as_f64()
                        .ok_or("nonfinite fine density operand")?,
                    coarse[field]
                        .as_f64()
                        .ok_or("nonfinite coarse density operand")?,
                    5e-7,
                );
            }
            comparisons += checks.len();
            failures += checks.iter().filter(|c| !c.passed).count();
            fine["checks"] = serde_json::to_value(&checks)?;
            fine["coarse_grid_operands"] = coarse;
            fine["dimension_and_population_scope"] = json!({"d":[1,2,4,8],"N":[1,8,32,128],"coordinate_LSI":fine["constants"]["lsi"],"product_LSI":"same coordinate constant by tensorization; normalize entropy/Fisher by N or N*d"});
            fine["envelope_center"] = json!(center);
            fine["profile_scope"] = json!(
                "Positive Gibbs-relative mixed-gradient density with the declared Gaussian-envelope center; between-wells tilt need not be a density concentrated at the saddle. Native saddle preparations are retained separately in complete native trajectories."
            );
            let archive = store.save_json(&tag, &fine)?;
            cases.push(json!({"id":tag,"archive":archive,"parameters":fine["parameters"],"constants":fine["constants"],"Phi":fine["Phi"],"Phi_dot":fine["Phi_dot"],"checks":checks.len(),"failures":checks.iter().filter(|c|!c.passed).count()}));
        }
    }
    let report = json!({"chapter":10,"cases":cases,"comparisons":comparisons,"comparisons_failed":failures,"native_updates":0,"source_scope":"Force and velocity-thermostat coefficients matched to actual native ensembles; continuous conservative reference law and full native law retain distinct generators."});
    store.save_json("completion-report", &report)?;
    store.finish("complete")?;
    let verification = store.verify(true)?;
    fs::write(
        PathBuf::from(&args[0]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        PathBuf::from(&args[0]).join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!(
        "{}",
        json!({"cases":cases.len(),"comparisons":comparisons,"failed":failures})
    );
    if failures > 0 {
        return Err("landscape-matched reference comparison failed; raw operands retained".into());
    }
    Ok(())
}
