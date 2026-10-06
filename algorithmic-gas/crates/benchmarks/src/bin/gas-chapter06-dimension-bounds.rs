//! Recompute dimension-explicit analytic bounds from retained native manifests.
use algorithmic_gas_benchmarks::{
    convergence_barrier_dimension::validate_retained_dimension_bounds,
    convergence_experiments::{ArchiveStore, ExperimentConfig, sha256_file},
    convergence_experiments_chapter06::run_dimension_probes,
};
use serde_json::json;
use std::{fs, path::PathBuf};

fn native_source_snapshot(
    path: &std::path::Path,
    output: &mut Vec<serde_json::Value>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut paths = fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            native_source_snapshot(&path, output)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(
                json!({"path":path,"sha256":sha256_file(&path)?,"text":fs::read_to_string(&path)?}),
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 || !["reanalyze", "probe"].contains(&args[0].as_str()) {
        return Err("gas-chapter06-dimension-bounds reanalyze INPUT_DATASET OUTPUT.json | probe CONFIG.json EMPTY_OUTPUT_DIR".into());
    }
    if args[0] == "probe" {
        let config: ExperimentConfig = serde_json::from_reader(fs::File::open(&args[1])?)?;
        config.validate()?;
        let mut store = ArchiveStore::new(&args[2])?;
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .to_path_buf();
        let mut native_sources = vec![];
        native_source_snapshot(
            &workspace.join("crates/algorithmic-gas/src"),
            &mut native_sources,
        )?;
        store.save_json("native-source-provenance",&json!({"core_sources":native_sources,"benchmark_library":include_str!("../lib.rs"),"cargo_lock":fs::read_to_string(workspace.join("Cargo.lock"))?,"runtime":"Native Rust f64 selected-cloning kernel; source snapshot retained with complete operator/noise records."}))?;
        store.save_json("dimension-probe-source",&json!({"config":config,"module":include_str!("../convergence_barrier_dimension.rs"),"native_wrapper":include_str!("../convergence_experiments_chapter06.rs"),"chapter":include_str!("../../../../docs/source/2_fractal_gas/convergence_program/06_convergence.md"),"scope":"Fresh selected-cloning native kernel; own independent seeds and complete raw records."}))?;
        let report = futures_lite::future::block_on(run_dimension_probes(&config, &mut store))?;
        store.save_json("dimension-probe-report", &report)?;
        store.finish("complete")?;
        fs::write(
            PathBuf::from(&args[2]).join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", report["summary"]);
        return Ok(());
    }
    let output = PathBuf::from(&args[2]);
    if output.exists() {
        return Err("derived output must be new; existing experiment audits are immutable".into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let report = validate_retained_dimension_bounds(PathBuf::from(&args[1]).as_path())?;
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", report["summary"]);
    Ok(())
}
