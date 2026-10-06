//! Run fresh native experiments or verify/re-analyze their immutable raw data.
use algorithmic_gas_benchmarks::{
    convergence_cloning::analyze_cloning_step,
    convergence_experiments::{ArchiveStore, ExperimentConfig, sha256_file},
    convergence_experiments_chapter04, convergence_experiments_chapter05,
    convergence_experiments_chapter06,
    convergence_stored::attach_coverage,
};
use serde_json::{Value, json};
use std::{
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
type Error = Box<dyn std::error::Error>;
fn value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, Error> {
    let Some(i) = args.iter().position(|s| s == flag) else {
        return Ok(None);
    };
    if i + 1 >= args.len() {
        return Err(format!("{flag} needs a value").into());
    }
    let result = args.remove(i + 1);
    args.remove(i);
    Ok(Some(result))
}
fn flag(args: &mut Vec<String>, name: &str) -> bool {
    if let Some(i) = args.iter().position(|s| s == name) {
        args.remove(i);
        true
    } else {
        false
    }
}
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}
fn native_sources(directory: &Path, workspace: &Path, code: &mut Vec<Value>) -> Result<(), Error> {
    let mut paths = fs::read_dir(directory)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            native_sources(&path, workspace, code)?;
        } else if path.extension().is_some_and(|x| x == "rs") {
            code.push(json!({"path":path.strip_prefix(workspace)?.to_string_lossy(),"sha256":sha256_file(&path)?,"text":fs::read_to_string(&path)?}));
        }
    }
    Ok(())
}
fn read_json(path: impl AsRef<Path>) -> Result<Value, Error> {
    Ok(serde_json::from_reader(BufReader::new(fs::File::open(
        path,
    )?))?)
}
fn write_json(path: impl AsRef<Path>, data: &Value) -> Result<(), Error> {
    let path = path.as_ref();
    let temporary = path.with_extension("json.partial");
    fs::write(&temporary, serde_json::to_vec_pretty(data)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
fn counts(report: &Value) -> (usize, usize) {
    let rows = report["comparisons"].as_array();
    (
        rows.map_or(0, Vec::len),
        rows.into_iter()
            .flatten()
            .filter(|r| r["passed"] == false || r["status"] == "violated")
            .count(),
    )
}
fn main() -> Result<(), Error> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let command = if args.is_empty() {
        "help".into()
    } else {
        args.remove(0)
    };
    if command == "help" || command == "--help" {
        println!(
            "gas-proof-experiments config\ngas-proof-experiments run [CONFIG.json] --chapter 4|5|6|all --output EMPTY_DIR [--strict]\ngas-proof-experiments verify DATASET [--deep]\ngas-proof-experiments reanalyze DATASET --output EMPTY_DERIVED_DIR\nRaw native records and resumable checkpoints use gzip CBOR; SHA256 index commits after every chunk. Reanalysis recomputes native conditional cloning checks without advancing the engine."
        );
        return Ok(());
    }
    if command == "config" {
        println!(
            "{}",
            serde_json::to_string_pretty(&ExperimentConfig::default())?
        );
        return Ok(());
    }
    if command == "verify" {
        let deep = flag(&mut args, "--deep");
        if args.len() != 1 {
            return Err("verify needs DATASET [--deep]".into());
        }
        let result = ArchiveStore::open(&args[0])?.verify(deep)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    if command == "reanalyze" {
        let output =
            value(&mut args, "--output")?.ok_or("reanalyze requires a fresh --output directory")?;
        if args.len() != 1 {
            return Err("reanalyze needs DATASET --output EMPTY_DERIVED_DIR".into());
        }
        let source = ArchiveStore::open(&args[0])?;
        let mut target = ArchiveStore::new(output)?;
        let mut steps = 0usize;
        let mut checks = 0usize;
        let mut failed = 0usize;
        let mut derived = vec![];
        for entry in source
            .entries()
            .iter()
            .filter(|e| e.kind == "native_run_archive")
        {
            let archive = source.load_archive(&entry.path)?;
            let mut rows = vec![];
            for (index, step) in archive.steps.iter().enumerate() {
                steps += 1;
                if step.stages.iter().any(|s| s.stage == "pre_clone") {
                    let report = analyze_cloning_step(&archive, index)?;
                    checks += report.checks.len();
                    failed += report.checks.iter().filter(|c| !c.passed).count();
                    if let Some(m) = &report.moments {
                        checks += m.checks.len();
                        failed += m.checks.iter().filter(|c| !c.passed).count();
                    }
                    rows.push(report);
                }
            }
            let path=target.save_json(&entry.tag,&json!({"source_archive":entry.path,"source_sha256":entry.sha256,"cloning_steps":rows}))?;
            derived.push(path);
        }
        let report = json!({"source_dataset":args[0],"new_engine_steps":0,"native_steps_decoded":steps,"conditional_checks":checks,"failed":failed,"derived_archives":derived,"scope":"Recomputed native conditional cloning identities and moments. Kinetic/SDE and survivor ensemble analyses retain their separate saved experiment reports; this command does not replace those predictions."});
        write_json(target.root().join("report.json"), &report)?;
        target.finish(if failed == 0 {
            "completed"
        } else {
            "discrepancy"
        })?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "source_dataset": args[0],
                "report": target.root().join("report.json"),
                "new_engine_steps": 0,
                "native_steps_decoded": steps,
                "conditional_checks": checks,
                "failed": failed,
                "derived_archives": derived.len(),
            }))?
        );
        if failed > 0 {
            return Err("saved-data cloning discrepancies retained".into());
        }
        return Ok(());
    }
    if command != "run" {
        return Err("unknown command; use help".into());
    }
    let chapter = value(&mut args, "--chapter")?.unwrap_or_else(|| "all".into());
    let chapters: &[u64] = match chapter.as_str() {
        "4" => &[4],
        "5" => &[5],
        "6" => &[6],
        "all" => &[4, 5, 6],
        _ => return Err("--chapter must be 4, 5, 6 or all".into()),
    };
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let output = value(&mut args, "--output")?
        .unwrap_or_else(|| format!("outputs/convergence/chapters04-06-experiments/run-{stamp}"));
    let strict = flag(&mut args, "--strict");
    if args.len() > 1 || args.iter().any(|s| s.starts_with('-')) {
        return Err("run takes one optional CONFIG.json plus --chapter/--output/--strict".into());
    }
    let config: ExperimentConfig = if let Some(path) = args.first() {
        serde_json::from_value(read_json(path)?)?
    } else {
        ExperimentConfig::default()
    };
    config.validate()?;
    let mut store = ArchiveStore::new(output)?;
    write_json(
        store.root().join("config.json"),
        &serde_json::to_value(&config)?,
    )?;
    let workspace = root();
    let mut snapshots = vec![];
    let mut inventories = vec![];
    for &number in chapters {
        let inventory = read_json(workspace.join(format!(
            "proof-validation/chapter{number:02}_inventory.json"
        )))?;
        let relative = inventory["source"].as_str().ok_or("missing source")?;
        let path = workspace.join(relative);
        let source = fs::read_to_string(&path)?;
        if inventory["source_sha256"] != sha256_file(&path)? {
            return Err(format!("Chapter {number} inventory is stale; regenerate proof-validation/generate_stored_inventories.py").into());
        }
        let snapshot=store.save_json(&format!("chapter{number:02}-source"),&json!({"source":relative,"sha256":sha256_file(&path)?,"text":source,"inventory":inventory}))?;
        snapshots.push(snapshot);
        inventories.push((inventory, source));
    }
    let mut code = vec![];
    native_sources(
        &workspace.join("crates/algorithmic-gas/src"),
        &workspace,
        &mut code,
    )?;
    native_sources(
        &workspace.join("crates/benchmarks/src"),
        &workspace,
        &mut code,
    )?;
    for relative in [
        "Cargo.lock",
        "crates/benchmarks/src/convergence_experiments.rs",
        "crates/benchmarks/src/convergence_experiments_chapter04.rs",
        "crates/benchmarks/src/convergence_experiments_chapter05.rs",
        "crates/benchmarks/src/convergence_experiments_chapter06.rs",
        "crates/benchmarks/src/bin/gas-proof-experiments.rs",
    ] {
        let path = workspace.join(relative);
        if path.exists() {
            code.push(json!({"path":relative,"sha256":sha256_file(&path)?,"text":fs::read_to_string(path)?}));
        }
    }
    let provenance = json!({"schema_version":1,"started_unix_seconds":stamp,"command":std::env::args().collect::<Vec<_>>(),"executable_sha256":sha256_file(&std::env::current_exe()?)?,"config":config,"source_snapshots":snapshots,"implementation":code,"numerical_scope":"Independent native replicate samples; normalized permutation-invariant errors. Finite empirical comparisons do not certify unstated global QSD/minorization hypotheses."});
    let provenance_path = store.save_json("provenance", &provenance)?;
    write_json(
        store.root().join("provenance.json"),
        &json!({"archive":provenance_path,"source_snapshots":snapshots}),
    )?;
    let mut results = vec![];
    let mut total = 0usize;
    let mut failures = 0usize;
    for (&number, (inventory, source)) in chapters.iter().zip(inventories.iter()) {
        eprintln!(
            "Running native chapter {number}; retaining every completed raw chunk in {}",
            store.root().display()
        );
        let result = futures_lite::future::block_on(async {
            match number {
                4 => convergence_experiments_chapter04::run(&config, &mut store).await,
                5 => convergence_experiments_chapter05::run(&config, &mut store).await,
                6 => convergence_experiments_chapter06::run(&config, &mut store).await,
                _ => unreachable!(),
            }
        });
        let mut report = match result {
            Ok(r) => r,
            Err(e) => {
                store.finish("incomplete")?;
                write_json(
                    store.root().join("execution-error.json"),
                    &json!({"chapter":number,"error":e.to_string(),"completed_chapters":results,"raw_data_retained":true}),
                )?;
                return Err(e.into());
            }
        };
        if let Err(e) = attach_coverage(&mut report, inventory, source) {
            store.finish("incomplete")?;
            write_json(
                store.root().join(format!("chapter{number:02}-report.json")),
                &report,
            )?;
            return Err(e.into());
        }
        let (n, f) = counts(&report);
        total += n;
        failures += f;
        write_json(
            store.root().join(format!("chapter{number:02}-report.json")),
            &report,
        )?;
        store.save_json(&format!("chapter{number:02}-report"), &report)?;
        eprintln!("Chapter {number}: {n} comparisons, {f} discrepancies");
        results.push(report);
    }
    let complete = results.iter().all(|r| r["coverage"]["complete"] == true);
    let summary = json!({"comparisons":total,"comparisons_failed":failures,"complete_required_expressions":complete,"raw_archives":store.entries().len()});
    write_json(
        store.root().join("report.json"),
        &json!({"schema_version":1,"config":config,"chapters":results,"summary":summary,"raw_data_index":"archive-index.json"}),
    )?;
    store.finish(if failures == 0 {
        "completed"
    } else {
        "discrepancy"
    })?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if strict && failures > 0 {
        return Err("experimental discrepancies retained alongside raw data".into());
    }
    Ok(())
}
