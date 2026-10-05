//! Run certified dimension/curvature kinetic probes with immutable native data.
use algorithmic_gas_benchmarks::{
    convergence_experiments::{ArchiveStore, ExperimentConfig, sha256_file},
    convergence_experiments_chapter05::dimension,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
type Error = Box<dyn std::error::Error>;
fn snapshot(directory: &Path, workspace: &Path, out: &mut Vec<Value>) -> Result<(), Error> {
    let mut files = fs::read_dir(directory)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    for file in files {
        if file.is_dir() {
            snapshot(&file, workspace, out)?;
        } else if file.extension().is_some_and(|e| e == "rs") {
            out.push(json!({"path":file.strip_prefix(workspace)?.to_string_lossy(),"sha256":sha256_file(&file)?,"text":fs::read_to_string(file)?}));
        }
    }
    Ok(())
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "sector" {
        let omega = args[1].parse::<f64>()?;
        let mut certificate = dimension::sector_curvature_profile(0.04, 1., omega)?;
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md",
        );
        certificate["source_path"] = json!(source);
        certificate["source_sha256"] = json!(sha256_file(&source)?);
        println!("{}", serde_json::to_string_pretty(&certificate)?);
        return Ok(());
    }
    if args.len() == 1 && args[0] == "constants" {
        let mut table = vec![];
        for omega in [0.5, 1., 2.] {
            for d in [1, 2, 4, 8] {
                table.push(json!({"curvature":omega,"d":d,"gaussian":dimension::constants(d,0.04,1.,1.,2.,omega,4096)?,"sector":dimension::sector_constants(0.04,1.,omega)?}));
            }
        }
        println!("{}", serde_json::to_string_pretty(&table)?);
        return Ok(());
    }
    if args.is_empty() || args.len() > 2 {
        return Err(
            "Usage: gas-kinetic-dimension constants | EMPTY_OUTPUT_DIRECTORY [CONFIG_JSON]".into(),
        );
    }
    let cfg: ExperimentConfig = if args.len() == 2 {
        serde_json::from_str(&fs::read_to_string(&args[1])?)?
    } else {
        ExperimentConfig {
            samples: 64,
            steps: 16,
            seed: 20261004,
            archive_chunk_steps: 16,
            ..Default::default()
        }
    };
    let mut store = ArchiveStore::new(&args[0])?;
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let mut implementation = vec![];
    for path in [
        "algorithmic-gas/crates/algorithmic-gas/src",
        "algorithmic-gas/crates/benchmarks/src",
    ] {
        snapshot(&workspace.join(path), workspace, &mut implementation)?;
    }
    let mut sources = vec![];
    for path in [
        "docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md",
        "docs/source/2_fractal_gas/convergence_program/18a_keystone_uniform_coupled.md",
    ] {
        let file = workspace.join(path);
        sources.push(
            json!({"path":path,"sha256":sha256_file(&file)?,"text":fs::read_to_string(file)?}),
        );
    }
    let provenance=store.save_json("dimension/provenance",&json!({"implementation":implementation,"source_snapshots":sources,"configuration":cfg,"command":std::env::args().collect::<Vec<_>>(),"executable_sha256":sha256_file(&std::env::current_exe()?)?,"Cargo_lock":fs::read_to_string(workspace.join("algorithmic-gas/Cargo.lock"))?}))?;
    let mut report = match futures_lite::future::block_on(dimension::run(&cfg, &mut store)) {
        Ok(r) => r,
        Err(e) => {
            store.finish("incomplete")?;
            return Err(e.into());
        }
    };
    report["provenance_archive"] = json!(provenance);
    fs::write(
        store.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    store.save_json("dimension/report", &report)?;
    let failures = report["summary"]["comparisons_failed"].as_u64().unwrap();
    store.finish(if failures == 0 {
        "completed"
    } else {
        "discrepancy"
    })?;
    fs::write(
        store.root().join("verification.json"),
        serde_json::to_vec_pretty(&store.verify(true)?)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    if failures > 0 {
        return Err("Dimension experiment discrepancies retained with raw native records".into());
    }
    Ok(())
}
