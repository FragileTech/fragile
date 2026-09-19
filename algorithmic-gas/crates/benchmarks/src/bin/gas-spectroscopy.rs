// Spectroscopy runner: prints the same JSON the browser receives.
use algorithmic_gas_benchmarks::spectroscopy::defaults;
use std::{env, fs};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let output = match args.iter().position(|a| a == "--output") {
        Some(i) => {
            let path = args.get(i + 1).ok_or("--output path required")?.clone();
            args.drain(i..=i + 1);
            Some(path)
        }
        None => None,
    };
    let command = args.first().ok_or(
        "Usage: gas-spectroscopy defaults | variants | run REQUEST.json | analyze EVIDENCE.cbor \
         [ANALYSIS.json] | archive REQUEST.json ARCHIVE.cbor... [--output PATH] [--evidence PATH]",
    )?;
    let document = match command.as_str() {
        "defaults" => defaults()?,
        "variants" | "run" | "analyze" | "archive" => {
            return Err(
                "gas-spectroscopy run, analyze, archive and variants are unavailable".into(),
            );
        }
        _ => return Err(format!("unknown command {command}").into()),
    };
    let text = serde_json::to_string_pretty(&document)?;
    match output {
        Some(path) => fs::write(path, text)?,
        None => println!("{text}"),
    }
    Ok(())
}
