// Run the Einstein-Hilbert gas natively and export its history.
use algorithmic_gas::Precision;
use algorithmic_gas_benchmarks::eh_gas::{EhRunConfig, run_configured};
use std::fs;

const USAGE: &str = "Usage: gas-eh [--config FILE.json] [--steps N] [--walkers N] [--dimensions N] \
[--seed N] [--precision f32|f64] [--init-spread X] [--record-every N] [--no-graph] \
[--output FILE.cbor|FILE.json] [--print-config]";

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut config = EhRunConfig::default();
    let mut output: Option<String> = None;
    let mut print_config = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--config" => config = serde_json::from_str(&fs::read_to_string(value()?)?)?,
            "--steps" => config.steps = value()?.parse()?,
            "--walkers" => config.walkers = value()?.parse()?,
            "--dimensions" => config.dimensions = value()?.parse()?,
            "--seed" => config.seed = value()?.parse()?,
            "--init-spread" => config.init_spread = value()?.parse()?,
            "--record-every" => config.record_every = value()?.parse()?,
            "--precision" => {
                config.precision = match value()?.as_str() {
                    "f32" => Precision::F32,
                    "f64" => Precision::F64,
                    other => return Err(format!("unknown precision {other}").into()),
                }
            }
            "--no-graph" => config.record_graph = false,
            "--output" => output = Some(value()?),
            "--print-config" => print_config = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}").into()),
        }
    }
    if print_config {
        println!("{}", serde_json::to_string_pretty(&config)?);
        return Ok(());
    }
    let history = futures_lite::future::block_on(run_configured(&config))?;
    println!("{}", serde_json::to_string_pretty(&history.summary)?);
    if let Some(path) = output {
        if path.ends_with(".json") {
            fs::write(&path, serde_json::to_vec(&history)?)?;
        } else {
            let mut bytes = Vec::new();
            ciborium::ser::into_writer(&history, &mut bytes)?;
            fs::write(&path, bytes)?;
        }
        eprintln!("wrote {} frames to {path}", history.frames.len());
    }
    Ok(())
}
