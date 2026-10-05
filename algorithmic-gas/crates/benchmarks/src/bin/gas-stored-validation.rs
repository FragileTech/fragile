//! Re-analyze chapters 4–6 using saved experiments, with no new engine steps.
use algorithmic_gas_benchmarks::{
    convergence_stored::attach_coverage, convergence_stored_chapter04,
    convergence_stored_chapter05, convergence_stored_chapter06,
};
use serde_json::{Value, json};
use std::{fs, io::BufReader, path::PathBuf};

fn take_value(
    args: &mut Vec<String>,
    flag: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        if index + 1 >= args.len() {
            return Err(format!("{flag} requires a value").into());
        }
        let value = args.remove(index + 1);
        args.remove(index);
        Ok(Some(value))
    } else {
        Ok(None)
    }
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let chapter = take_value(&mut args, "--chapter")?.unwrap_or_else(|| "all".into());
    let output = take_value(&mut args, "--output")?
        .unwrap_or_else(|| "outputs/convergence/chapters04-06-stored.json".into());
    let strict = take_flag(&mut args, "--strict");
    let require_complete = take_flag(&mut args, "--require-complete");
    let chapters: &[u64] = match chapter.as_str() {
        "4" => &[4],
        "5" => &[5],
        "6" => &[6],
        "all" => &[4, 5, 6],
        _ => return Err("--chapter must be 4, 5, 6 or all".into()),
    };
    if args.is_empty() {
        args = vec![
            "outputs/convergence/chapters01-03-extended.json".into(),
            "outputs/convergence/decay-independent.json".into(),
            "outputs/convergence/decay-shared.json".into(),
            "outputs/convergence/chapter03-empirical.json".into(),
        ];
    }
    if args.iter().any(|arg| arg.starts_with('-')) {
        return Err("unknown argument; use --chapter, --output, --strict, --require-complete and saved report paths".into());
    }
    let mut reports = vec![];
    for path in &args {
        eprintln!("Reading retained experiments: {path}");
        let value: Value = serde_json::from_reader(BufReader::new(fs::File::open(path)?))?;
        reports.push((path.clone(), value));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("workspace root missing")?
        .to_path_buf();
    let mut results = vec![];
    for &number in chapters {
        eprintln!("Analyzing chapter {number} from retained data");
        let mut result = match number {
            4 => convergence_stored_chapter04::validate(&reports)?,
            5 => convergence_stored_chapter05::validate(&reports)?,
            6 => convergence_stored_chapter06::validate(&reports)?,
            _ => unreachable!(),
        };
        let inventory: Value =
            serde_json::from_reader(BufReader::new(fs::File::open(root.join(format!(
                "algorithmic-gas/proof-validation/chapter{number:02}_inventory.json"
            )))?))?;
        let source = fs::read_to_string(
            root.join(
                inventory["source"]
                    .as_str()
                    .ok_or("inventory source missing")?,
            ),
        )?;
        attach_coverage(&mut result, &inventory, &source)?;
        results.push(result);
    }
    let comparisons = results
        .iter()
        .map(|r| r["comparisons"].as_array().map_or(0, Vec::len))
        .sum::<usize>();
    let failed = results
        .iter()
        .flat_map(|r| r["comparisons"].as_array().into_iter().flatten())
        .filter(|comparison| comparison["passed"] == false || comparison["status"] == "violated")
        .count();
    let unbound = results
        .iter()
        .map(|r| {
            r["coverage"]["unbound_evidence"]
                .as_array()
                .map_or(0, Vec::len)
        })
        .sum::<usize>();
    let complete = results.iter().all(|r| r["coverage"]["complete"] == true);
    let value = json!({"schema_version":1,"input_reports":args,"new_engine_steps":0,
        "chapters":results,"summary":{"comparisons":comparisons,"comparisons_failed":failed,
        "unbound_evidence":unbound,"complete_required_expressions":complete},
        "scope":"Re-analysis of existing native records, with theorem-specific applicability gates and exact source coverage. No fresh trajectories or universal theorem certificate."});
    let output = PathBuf::from(output);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&value)?)?;
    println!("{}", serde_json::to_string_pretty(&value["summary"])?);
    if strict && (failed > 0 || unbound > 0) {
        return Err("stored-data discrepancies or source binding errors retained in report".into());
    }
    if require_complete && !complete {
        return Err("stored data does not cover every required expression; inspect the saved coverage and gaps".into());
    }
    Ok(())
}
