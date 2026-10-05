//! Individual formula validation for convergence chapters one through three.
use algorithmic_gas_benchmarks::convergence_estimates::{
    required_estimates_complete, validate_chapters,
};
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let strict = take_flag(&mut args, "--strict");
    let complete = take_flag(&mut args, "--require-complete");
    let samples = take_value(&mut args, "--samples")?
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1024);
    let output = take_value(&mut args, "--output")?;
    let chapters = take_value(&mut args, "--chapters")?
        .unwrap_or_else(|| "1,2,3".to_owned())
        .split(',')
        .map(str::parse::<u8>)
        .collect::<Result<Vec<_>, _>>()?;
    if !args.is_empty() {
        return Err("Usage: gas-estimates [--chapters 1,2,3] [--samples N] [--output REPORT.json] [--strict] [--require-complete]".into());
    }
    let report = futures_lite::future::block_on(validate_chapters(samples, &chapters))?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        if let Some(parent) = Path::new(&path)
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, encoded)?;
    } else {
        io::stdout().write_all(&encoded)?;
        io::stdout().write_all(b"\n")?;
    }
    if strict
        && (report["summary"]["phase_errors"].as_u64().unwrap_or(1) > 0
            || report["summary"]["comparisons_failed"]
                .as_u64()
                .unwrap_or(1)
                > 0
            || report["summary"]["hypotheses_failed"].as_u64().unwrap_or(1) > 0
            || report["coverage"]["unbound_evidence"]
                .as_array()
                .is_none_or(|values| !values.is_empty()))
    {
        return Err("an estimate comparison, hypothesis, or exact source binding failed; inspect the saved evidence".into());
    }
    if complete
        && (report["summary"]["phase_errors"] != 0
            || !required_estimates_complete(&report["coverage"]))
    {
        return Err("source estimates still lack individual executed evidence; inspect the complete expression catalog".into());
    }
    Ok(())
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|value| value == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}
fn take_value(
    args: &mut Vec<String>,
    flag: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if let Some(index) = args.iter().position(|value| value == flag) {
        if index + 1 == args.len() {
            return Err(format!("{flag} needs a value").into());
        }
        let value = args.remove(index + 1);
        args.remove(index);
        Ok(Some(value))
    } else {
        Ok(None)
    }
}
