//! Native measured-vs-predicted chapter 3 proposal experiment matrix.
use algorithmic_gas_benchmarks::convergence_cloning_empirical::validate_empirical;
use std::{fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let strict = take_flag(&mut args, "--strict");
    let compact = take_flag(&mut args, "--compact");
    let samples = take_value(&mut args, "--samples")?
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(1024);
    let output = take_value(&mut args, "--output")?
        .unwrap_or_else(|| "outputs/convergence/chapter03-empirical.json".into());
    if !args.is_empty() {
        return Err(
            "Usage: gas-cloning-validation [--samples N] [--output FILE] [--compact] [--strict]"
                .into(),
        );
    }
    let report = futures_lite::future::block_on(validate_empirical(samples, compact))?;
    if let Some(parent) = Path::new(&output).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    if strict && report["summary"]["comparisons_violated"] != 0 {
        return Err(
            "chapter 3 empirical bound/equality discrepancies; inspect saved report".into(),
        );
    }
    Ok(())
}
fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(i) = args.iter().position(|v| v == flag) {
        args.remove(i);
        true
    } else {
        false
    }
}
fn take_value(
    args: &mut Vec<String>,
    flag: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if let Some(i) = args.iter().position(|v| v == flag) {
        if i + 1 == args.len() {
            return Err(format!("{flag} requires a value").into());
        }
        let value = args.remove(i + 1);
        args.remove(i);
        Ok(Some(value))
    } else {
        Ok(None)
    }
}
