//! Deterministic Gaussian cubature of native quadratic refinement observables.
use algorithmic_gas_benchmarks::{
    convergence_experiments::{sha256_file, ArchiveStore},
    convergence_experiments_chapter05::run_cubature,
};
use serde_json::{json, Value};
use std::{fs, path::Path};

type Error = Box<dyn std::error::Error>;

fn snapshots(directory: &Path, workspace: &Path, records: &mut Vec<Value>) -> Result<(), Error> {
    let mut paths = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            snapshots(&path, workspace, records)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            records.push(json!({"path":path.strip_prefix(workspace)?.to_string_lossy(),"sha256":sha256_file(&path)?,"text":fs::read_to_string(&path)?}));
        }
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("Usage: gas-kinetic-cubature EMPTY_OUTPUT_DIRECTORY".into());
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let mut store = ArchiveStore::new(&args[0])?;
    let mut code = vec![];
    for directory in [
        "crates/algorithmic-gas/src",
        "crates/benchmarks/src",
    ] {
        snapshots(&workspace.join(directory), workspace, &mut code)?;
    }
    let source_path =
        workspace.join("docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md");
    let inventory = workspace.join("proof-validation/chapter05_inventory.json");
    let provenance = store.save_json("cubature/provenance", &json!({"command":std::env::args().collect::<Vec<_>>(),"executable_sha256":sha256_file(&std::env::current_exe()?)?,"implementation":code,"Cargo_lock":fs::read_to_string(workspace.join("Cargo.lock"))?,"chapter_source":{"path":source_path.strip_prefix(workspace)?.to_string_lossy(),"sha256":sha256_file(&source_path)?,"text":fs::read_to_string(&source_path)?,"inventory":serde_json::from_str::<Value>(&fs::read_to_string(inventory)?)?},"method":"96-node degree-two Gaussian cubature, not Monte Carlo sampling"}))?;
    let mut report = match futures_lite::future::block_on(run_cubature(&mut store)) {
        Ok(report) => report,
        Err(error) => {
            store.finish("incomplete")?;
            return Err(error.into());
        }
    };
    report["provenance_archive"] = json!(provenance);
    fs::write(
        store.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    store.save_json("cubature/report", &report)?;
    let failed = report["summary"]["comparisons_failed"].as_u64().unwrap();
    store.finish(if failed == 0 {
        "completed"
    } else {
        "discrepancy"
    })?;
    let verification = store.verify(true)?;
    fs::write(
        store.root().join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    if failed > 0 {
        return Err("Cubature discrepancies retained with the complete raw data".into());
    }
    Ok(())
}
