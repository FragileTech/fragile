//! Conditional full-step fluctuations from frozen actual-engine inputs.
use algorithmic_gas::{RecordingConfig, Result, mean_field::step_diagnostics};
use algorithmic_gas_benchmarks::lecture_meanfield;
use serde_json::json;

fn moments(xs: &[f64]) -> serde_json::Value {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.);
    json!({"mean":mean,"variance":variance,"mean_standard_error":(variance/n).sqrt()})
}

async fn run(seed_start: u64) -> Result<serde_json::Value> {
    let mut observations = vec![];
    for case in ["canonical", "boundary_stress"] {
        for n in [16, 32, 64, 128, 256] {
            let params = if case == "boundary_stress" {
                json!({"walkers":n,"box":0.6,"position_diffusion":0.7})
            } else {
                json!({"walkers":n})
            };
            let base = lecture_meanfield::config("III-03", &params, 91_000)?;
            let mut source = base.build::<f64>().await?;
            for step in 1..=64 {
                if [1, 4, 16, 64].contains(&step) {
                    let frozen = source.population().clone();
                    let mut cosines = vec![];
                    let mut alive = vec![];
                    let mut component_sharing = vec![];
                    let mut revivals = 0;
                    let mut extinction = 0;
                    let mut sizes = std::collections::BTreeMap::<usize, usize>::new();
                    for seed in seed_start..seed_start + 128 {
                        let config = lecture_meanfield::config("III-03", &params, seed)?;
                        let mut gas = config.build::<f64>().await?;
                        gas.replace_population(frozen.clone()).await?;
                        gas.start_recording(RecordingConfig::default())?;
                        match gas.step().await {
                            Ok(_) => {}
                            // Extinction commits no step: there is nothing to diagnose.
                            Err(algorithmic_gas::GasError::Extinction) => {
                                extinction += 1;
                                gas.stop_recording();
                                continue;
                            }
                            Err(error) => return Err(error),
                        }
                        let archive = gas.stop_recording().unwrap();
                        let diag = step_diagnostics(&archive.steps[0])?;
                        revivals += diag.revivals;
                        let sharing = diag
                            .component_sizes
                            .iter()
                            .map(|s| s * (s - 1))
                            .sum::<usize>() as f64
                            / (n * (n - 1)) as f64;
                        component_sharing.push(sharing);
                        for size in diag.component_sizes {
                            *sizes.entry(size).or_default() += 1;
                        }
                        let positions = gas.population().observations.field("positions")?;
                        cosines.push(
                            positions
                                .values()
                                .chunks(2)
                                .map(|row| row[0].cos())
                                .sum::<f64>()
                                / n as f64,
                        );
                        alive.push(diag.alive_after as f64 / n as f64);
                    }
                    observations.push(json!({
                        "case":case,"N":n,"observation_update":step,
                        "source_seed":91_000,"source_completed_updates":step-1,
                        "probe_seed_start":seed_start,"probe_replicas":128,
                        "probe_engine_step":0,"configuration":base,
                        "input_population":frozen,
                        "cosine":moments(&cosines),"alive_fraction":moments(&alive),
                        "shared_component_probability":moments(&component_sharing),
                        "component_histogram":sizes,"revivals":revivals,"extinctions":extinction
                    }));
                }
                source.step().await?;
            }
            eprintln!("{case} N={n}: frozen-input comparisons complete");
        }
    }
    Ok(json!({"kind":"actual_rust_conditional_full_step","observations":observations}))
}

fn main() -> Result<()> {
    let seed_start = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("integer seed start"))
        .unwrap_or(92_000);
    let report = futures_lite::future::block_on(run(seed_start))?;
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}
