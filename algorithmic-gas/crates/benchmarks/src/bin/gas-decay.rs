//! Standalone native complete-update alive-error and parameter-calibration runner.
use algorithmic_gas_benchmarks::convergence_decay::{
    DecayValidationConfig, DecayValidationReport, refresh_decay_summary, validate_decay,
};
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|value| value == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let output = if let Some(index) = args.iter().position(|value| value == "--output") {
        if index + 1 >= args.len() {
            return Err("--output requires a path".into());
        }
        let path = args.remove(index + 1);
        args.remove(index);
        Some(path)
    } else {
        None
    };
    let strict = take_flag(&mut args, "--strict");
    let require_decrease = take_flag(&mut args, "--require-decrease");
    let value = match args.first().map(String::as_str).unwrap_or("run") {
        "config" if args.len() == 1 => serde_json::to_value(DecayValidationConfig::default())?,
        "run" if args.len() <= 2 => {
            let config = args.get(1).map(|path| -> Result<DecayValidationConfig, Box<dyn std::error::Error>> {
                Ok(serde_json::from_slice(&fs::read(path)?)?)
            }).transpose()?.unwrap_or_default();
            serde_json::to_value(futures_lite::future::block_on(validate_decay(&config))?)?
        }
        "analyze" if args.len() == 2 => {
            let mut report: DecayValidationReport = serde_json::from_slice(&fs::read(&args[1])?)?;
            refresh_decay_summary(&mut report)?;
            serde_json::to_value(report)?
        }
        _ => return Err("Usage: gas-decay config | run [CONFIG.json] | analyze REPORT.json [--output REPORT.json] [--strict] [--require-decrease]".into()),
    };
    let encoded = serde_json::to_vec_pretty(&value)?;
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
    if let Some(cases) = value["cases"].as_array() {
        if strict
            && cases.iter().any(|case| {
                case["checks"]
                    .as_array()
                    .is_some_and(|checks| checks.iter().any(|check| check["passed"] == false))
                    || case["statistical_checks"].as_array().is_some_and(|checks| {
                        checks.iter().any(|check| check["status"] == "violated")
                    })
            })
        {
            return Err("prediction discrepancies recorded; inspect the saved decay report".into());
        }
        let pairs = value["config"]["seeds"].as_array().map_or(0, Vec::len) as u64;
        if require_decrease
            && cases.iter().any(|case| {
                case["endpoint_decline"]["empirical_mean_decreased"] != true
                    || case["endpoint_decline"]["complete_seed_pairs"].as_u64() != Some(pairs)
            })
        {
            return Err("not every complete paired alive-error endpoint mean decreased; inspect the saved report and its uncertainty".into());
        }
    }
    Ok(())
}
