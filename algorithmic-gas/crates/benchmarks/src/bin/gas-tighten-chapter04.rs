use algorithmic_gas_benchmarks::convergence_tightening_chapter04::analyze_dataset;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: gas-tighten-chapter04 DATASET_DIRECTORY FRESH_OUTPUT_DIRECTORY".into());
    }
    let r = analyze_dataset(Path::new(&args[0]), Path::new(&args[1]))?;
    if r["summary"]["comparisons_failed"].as_u64().unwrap() != 0 {
        return Err("Numerical tightening comparisons failed; inspect saved report".into());
    }
    Ok(())
}
