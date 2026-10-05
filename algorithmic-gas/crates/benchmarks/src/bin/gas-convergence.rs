//! Native chapter 1--3 inventory and simulation runner.
use algorithmic_gas_benchmarks::convergence_validation::{
    ValidationConfig, inventory, run_validation,
};
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let output = if let Some(index) = args.iter().position(|x| x == "--output") {
        if index + 1 >= args.len() {
            return Err("--output requires a path".into());
        }
        let path = args.remove(index + 1);
        args.remove(index);
        Some(path)
    } else {
        None
    };
    let strict = if let Some(index) = args.iter().position(|x| x == "--strict") {
        args.remove(index);
        true
    } else {
        false
    };
    let value = match args.first().map(String::as_str).unwrap_or("run") {
        "config" if args.len() == 1 => serde_json::to_value(ValidationConfig::default())?,
        "inventory" if args.len() == 1 => serde_json::to_value(inventory()?)?,
        "run" if args.len() <= 2 => {
            let config = args.get(1).map(|path| -> std::result::Result<ValidationConfig,Box<dyn std::error::Error>> { Ok(serde_json::from_slice(&fs::read(path)?)?) }).transpose()?.unwrap_or_default();
            futures_lite::future::block_on(run_validation(&config))?
        }
        _ => return Err("Usage: gas-convergence config | inventory | run [CONFIG.json] [--output REPORT.json] [--strict]".into()),
    };
    let encoded = serde_json::to_vec_pretty(&value)?;
    if let Some(path) = output {
        if let Some(parent) = Path::new(&path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, encoded)?;
    } else {
        io::stdout().write_all(&encoded)?;
        io::stdout().write_all(b"\n")?;
    }
    if strict
        && (value["summary"]["exact_checks_failed"]
            .as_u64()
            .unwrap_or(0)
            > 0
            || value["summary"]["run_errors"].as_u64().unwrap_or(0) > 0
            || value["summary"]["phase_errors"].as_u64().unwrap_or(0) > 0
            || value["summary"]["kinetic_statistical_checks_violated"]
                .as_u64()
                .unwrap_or(0)
                > 0
            || value["summary"]["cloning_statistical_checks_violated"]
                .as_u64()
                .unwrap_or(0)
                > 0
            || value["summary"]["auxiliary_statistical_checks_violated"]
                .as_u64()
                .unwrap_or(0)
                > 0)
    {
        return Err("validation discrepancies recorded; inspect report".into());
    }
    Ok(())
}
