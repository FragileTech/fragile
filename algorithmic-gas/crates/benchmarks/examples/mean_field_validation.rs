//! Reproducible actual-engine population study. JSON goes to stdout.
use algorithmic_gas::{
    RecordingConfig, Result, TensorBatch,
    mean_field::{population_observables, step_diagnostics},
    physics::field_evolution::{WeakFieldObservable, collision_field_balance},
};
use algorithmic_gas_benchmarks::lecture_meanfield;
use serde_json::{Value, json};

async fn run_case(case: &str, seed_start: u64, replicas: usize, sizes: &[usize]) -> Result<Value> {
    let mut configurations = vec![];
    let mut observations = vec![];
    let mut trajectories = vec![];
    for &n in sizes {
        let p = if case == "boundary_stress" {
            json!({"walkers":n,"box":0.6,"position_diffusion":0.7})
        } else {
            json!({"walkers":n})
        };
        configurations.push(lecture_meanfield::config("III-03", &p, seed_start)?);
        for seed in seed_start..seed_start + replicas as u64 {
            let c = lecture_meanfield::config("III-03", &p, seed)?;
            let mut gas = c.build::<f64>().await?;
            if case == "shared_initial" || case == "shifted_initial" {
                let mut pop = gas.population().clone();
                let mut x = pop.observations.field("positions")?.values().to_vec();
                let shared = x[0];
                for row in x.chunks_mut(2) {
                    row[0] = if case == "shared_initial" {
                        shared
                    } else {
                        1.2 + 0.2 * row[0]
                    };
                }
                pop.observations
                    .fields
                    .insert("positions".into(), TensorBatch::vectors(n, 2, x)?);
                gas.replace_population(pop).await?;
            }
            let initial = population_observables(gas.population())?;
            observations.push(json!({"N":n,"seed":seed,"step":0,"observables":initial}));
            let mut clone_residual = 0.;
            let mut clone_variance = 0.;
            let mut stress_residual = 0.;
            let mut stress_variance = 0.;
            let mut position_residual = 0.;
            let mut position_variance = 0.;
            let mut max_momentum: f64 = 0.;
            let mut max_energy: f64 = 0.;
            let mut revivals = 0;
            let mut clones = 0;
            let mut deaths = 0;
            let mut large_components = 0;
            let mut extinct = false;
            for k in 1..=64 {
                gas.start_recording(RecordingConfig {
                    max_steps: 1,
                    max_bytes: 64 * 1024 * 1024,
                    ..Default::default()
                })?;
                match gas.step().await {
                    Ok(_) => {}
                    Err(algorithmic_gas::GasError::Extinction) => {
                        extinct = true;
                        gas.stop_recording();
                        break;
                    }
                    Err(e) => return Err(e),
                }
                let archive = gas.stop_recording().unwrap();
                let step = &archive.steps[0];
                let diag = step_diagnostics(step)?;
                clone_residual += diag.clones as f64 - diag.expected_clones;
                clone_variance += diag.clone_count_variance;
                revivals += diag.revivals;
                clones += diag.clones;
                deaths += n - diag.alive_after;
                large_components += diag.component_sizes.iter().filter(|&&s| s > 2).count();
                let balance = collision_field_balance(
                    &c.gas,
                    step,
                    &WeakFieldObservable::Stress {
                        k: vec![0., 0.],
                        a: 0,
                        b: 0,
                    },
                )?;
                stress_residual += balance.martingale_increment[0];
                stress_variance += balance.martingale_covariance[0];
                max_momentum = max_momentum.max(balance.maximum_momentum_conservation_residual);
                max_energy = max_energy.max(balance.maximum_relative_energy_residual);
                let prior = step.stages.iter().find(|s| s.stage == "B2").unwrap();
                let post = step
                    .stages
                    .iter()
                    .find(|s| s.stage == "position_diffusion")
                    .unwrap();
                let variance = c.gas.kinetic.position_diffusion.powi(2) * 0.04;
                for (x, y) in prior.fields["positions"]
                    .values
                    .chunks(2)
                    .zip(post.fields["positions"].values.chunks(2))
                {
                    let expected = (-variance / 2.).exp() * x[0].cos();
                    position_residual += (y[0].cos() - expected) / n as f64;
                    position_variance += (0.5 * (1. + (-2. * variance).exp() * (2. * x[0]).cos())
                        - expected * expected)
                        / (n * n) as f64;
                }
                if [1, 4, 16, 64].contains(&k) {
                    observations.push(json!({"N":n,"seed":seed,"step":k,"observables":population_observables(gas.population())?,"graph":diag}));
                }
            }
            trajectories.push(json!({"N":n,"seed":seed,"extinct":extinct,"clone_residual":clone_residual,"clone_variance":clone_variance,
                "stress_residual":stress_residual,"stress_variance":stress_variance,"position_residual":position_residual,"position_variance":position_variance,
                "maximum_momentum_residual":max_momentum,"maximum_energy_residual":max_energy,"clones":clones,"revivals":revivals,"terminal_dead_slots":deaths,"components_larger_than_pairs":large_components}));
        }
        eprintln!("{case}: N={n}, {replicas} trajectories complete");
    }
    Ok(
        json!({"case":case,"seed_start":seed_start,"replicas_per_N":replicas,"configurations":configurations,"observations":observations,"trajectories":trajectories}),
    )
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let replicas = args.first().and_then(|x| x.parse().ok()).unwrap_or(128);
    let seed_start = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(0);
    let case = args.get(2).map(String::as_str).unwrap_or("canonical");
    let sizes = args
        .get(3)
        .map(|s| {
            s.split(',')
                .map(|n| n.parse().expect("population size"))
                .collect()
        })
        .unwrap_or_else(|| {
            if case == "canonical" {
                vec![16, 32, 64, 128, 256]
            } else {
                vec![16, 64, 256]
            }
        });
    let report = futures_lite::future::block_on(run_case(case, seed_start, replicas, &sizes))?;
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}
