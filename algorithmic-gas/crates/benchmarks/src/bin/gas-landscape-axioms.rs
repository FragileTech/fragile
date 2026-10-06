//! Independent native-gradient checks of the global nonlinear landscape axiom.
use algorithmic_gas::{ExecutionContext, Precision, TensorBatch, compute::BackendKind};
use algorithmic_gas_benchmarks::{
    convergence_experiments::{ArchiveStore, sha256_file},
    convergence_landscape_axioms::{
        native_rastrigin_gradient, rastrigin_gradient_budget, rastrigin_segment_energy,
        rastrigin_short_gradient_budget,
    },
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) || args.get(1).is_some_and(|a| a != "short") {
        return Err("gas-landscape-axioms EMPTY_OUTPUT_DIRECTORY [short]".into());
    }
    let mut store = ArchiveStore::new(&args[0])?;
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let mut sources = vec![];
    for name in [
        "docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md",
        "crates/benchmarks/src/convergence_landscape_axioms.rs",
        "crates/benchmarks/src/bin/gas-landscape-axioms.rs",
        "crates/benchmarks/src/benchmark.rs",
        "crates/benchmarks/src/classics.rs",
    ] {
        let p = workspace.join(name);
        sources.push(json!({"path":name,"sha256":sha256_file(&p)?,"text":fs::read_to_string(p)?}));
    }
    let provenance = store.save_json(
        "axioms/provenance",
        &json!({
            "sources":sources,"executable_sha256":sha256_file(&std::env::current_exe()?)?,
            "command":std::env::args().collect::<Vec<_>>()
        }),
    )?;
    let panels = 8192;
    let mut cx =
        futures_lite::future::block_on(ExecutionContext::new(BackendKind::Cpu, Precision::F64))?;
    let mut cases: Vec<Value> = vec![];
    let mut comparisons = 0;
    let mut failures = 0;
    for d in [1, 2, 4, 8] {
        let budget = if args.len() == 2 {
            rastrigin_short_gradient_budget(d)?
        } else {
            rastrigin_gradient_budget(d)?
        };
        for factor in [1., 1.5, 2.] {
            for midpoint in [-2., 0., 0.5, 10.] {
                for direction in 0..3 {
                    let mut u: Vec<f64> = (0..d)
                        .map(|j| match direction {
                            0 => {
                                if j == 0 {
                                    1.
                                } else {
                                    0.
                                }
                            }
                            1 => 1.,
                            _ => {
                                if j % 2 == 0 {
                                    1.
                                } else {
                                    -0.5
                                }
                            }
                        })
                        .collect();
                    let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
                    u.iter_mut().for_each(|v| *v /= norm);
                    let length = factor * budget.minimum_segment_length;
                    let x: Vec<_> = u
                        .iter()
                        .enumerate()
                        .map(|(j, v)| midpoint + 0.13 * j as f64 - length * v / 2.)
                        .collect();
                    let y: Vec<_> = x.iter().zip(&u).map(|(a, v)| a + length * v).collect();
                    let points: Vec<_> = (0..=panels)
                        .flat_map(|i| {
                            let t = i as f64 / panels as f64;
                            x.iter().zip(&y).map(move |(a, b)| a + t * (b - a))
                        })
                        .collect();
                    let input = TensorBatch::vectors(panels + 1, d, points)?;
                    let native =
                        futures_lite::future::block_on(native_rastrigin_gradient(&mut cx, &input))?;
                    let gradients: Vec<Vec<f64>> = native
                        .values()
                        .chunks_exact(d)
                        .map(|g| g.to_vec())
                        .collect();
                    let mut integral = 0.;
                    for (i, gradient) in gradients.iter().enumerate() {
                        let energy = gradient.iter().map(|v| v * v).sum::<f64>();
                        let weight = if i == 0 || i == panels {
                            1.
                        } else if i % 2 == 0 {
                            2.
                        } else {
                            4.
                        };
                        integral += weight * energy;
                    }
                    integral /= 3. * panels as f64;
                    let (exact, quadrature_error) = rastrigin_segment_energy(&x, &y, panels)?;
                    let tolerance = 2e-10 * exact.abs().max(1.);
                    let actual_length = x
                        .iter()
                        .zip(&y)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let checks = [
                        json!({"name":"native_gradient_vs_exact_segment_integral","observed":(integral-exact).abs(),"bound":quadrature_error+tolerance,"passed":(integral-exact).abs()<=quadrature_error+tolerance}),
                        json!({"name":"exact_gradient_energy_lower_bound","observed":budget.gradient_energy_lower,"bound":exact+tolerance,"passed":budget.gradient_energy_lower<=exact+tolerance}),
                        json!({"name":"quadrature_lower_confirms_nondeception","observed":budget.gradient_energy_lower,"bound":integral-quadrature_error+tolerance,"passed":budget.gradient_energy_lower<=integral-quadrature_error+tolerance}),
                        json!({"name":"actual_segment_length_premise","observed":budget.minimum_segment_length,"bound":actual_length+tolerance,"passed":budget.minimum_segment_length<=actual_length+tolerance}),
                    ];
                    comparisons += checks.len();
                    failures += checks.iter().filter(|c| c["passed"] != true).count();
                    let name = format!("d{d}_length{factor}_mid{midpoint}_dir{direction}");
                    let inputs=store.save_json(&format!("axioms/{name}"),&json!({
                        "x":x,"y":y,"dimension":d,"parameters":budget,"quadrature_panels":panels,
                        "native_gradients":gradients,"measured_segment_integral":integral,
                        "independent_exact_segment_integral":exact,"simpson_error_upper":quadrature_error,
                        "checks":checks
                    }))?;
                    cases.push(json!({"case":name,"d":d,"budget":budget,"measured":integral,"exact":exact,"quadrature_error_upper":quadrature_error,"checks":checks,"inputs_archive":inputs}));
                }
            }
        }
    }
    let report = json!({"cases":cases,"provenance_archive":provenance,
        "source_label":"cor-eg-nonquadratic-gradient-certificate","scope":"Global analytic non-deception under a declared bounded perturbation of an SPD force. Native gradients test exact trigonometric segment integrals and rigorous Simpson error envelopes. No source-pressure or whole-law mixing consequence is inferred.",
        "summary":{"cases":cases.len(),"comparisons":comparisons,"comparisons_failed":failures,"native_gradient_evaluations":cases.len()*(panels+1),"native_updates":0}});
    fs::write(
        store.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    store.save_json("axioms/report", &report)?;
    store.finish(if failures == 0 {
        "complete"
    } else {
        "discrepancy"
    })?;
    let verified = store.verify(true)?;
    fs::write(
        store.root().join("verification.json"),
        serde_json::to_vec_pretty(&verified)?,
    )?;
    println!("{}", report["summary"]);
    if failures > 0 {
        return Err("Nonlinear axiom comparisons failed; evidence saved".into());
    }
    Ok(())
}
