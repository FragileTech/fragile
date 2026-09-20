// Spectroscopy runner: prints the same JSON the browser receives.
use algorithmic_gas::{RunArchive, physics::spectroscopy::AnalysisConfig};
use algorithmic_gas_benchmarks::spectroscopy::{
    SpectroscopyEvidence, SpectroscopyRequest, SpectroscopySession, analyze_archive,
    analyze_evidence, defaults,
};
use serde_json::Value;
use std::{
    env, fs,
    io::{self, Read},
};
type Fallible<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const USAGE: &str = "Usage: gas-spectroscopy defaults | variants | run REQUEST.json | analyze \
                     EVIDENCE.cbor [ANALYSIS.json] | archive REQUEST.json ARCHIVE.cbor... \
                     [--output PATH] [--evidence PATH]";
fn read(path: &str) -> Fallible<String> {
    if path == "-" {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text)?;
        return Ok(text);
    }
    Ok(fs::read_to_string(path)?)
}
fn option(args: &mut Vec<String>, name: &str) -> Fallible<Option<String>> {
    match args.iter().position(|a| a == name) {
        Some(i) => {
            let path = args
                .get(i + 1)
                .ok_or(format!("{name} path required"))?
                .clone();
            args.drain(i..=i + 1);
            Ok(Some(path))
        }
        None => Ok(None),
    }
}
/// JSON is the human-readable archive format of the engine and CBOR its state
/// format; the extension says which one this file is.
fn archive(path: &str) -> Fallible<RunArchive<f64>> {
    if path.ends_with(".json") {
        return Ok(RunArchive::from_json(&read(path)?)?);
    }
    Ok(RunArchive::from_bytes(&fs::read(path)?)?)
}
async fn execute(request: SpectroscopyRequest) -> algorithmic_gas::Result<(Value, Vec<u8>)> {
    let mut session = SpectroscopySession::create(request).await?;
    while !session.done() {
        session.advance(64).await?;
    }
    let analysis = session.request().spectroscopy.analysis.clone();
    let report = session.analyze(&analysis)?;
    Ok((
        serde_json::to_value(report)
            .map_err(|e| algorithmic_gas::GasError::Configuration(e.to_string()))?,
        session.evidence().to_bytes()?,
    ))
}
fn main() -> Fallible<()> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let output = option(&mut args, "--output")?;
    let evidence_path = option(&mut args, "--evidence")?;
    let command = args.first().ok_or(USAGE)?;
    let document = match command.as_str() {
        "defaults" => defaults()?,
        "variants" => defaults()?["variants"].take(),
        "run" => {
            let path = args.get(1).ok_or("request path required")?;
            let request: SpectroscopyRequest = serde_json::from_str(&read(path)?)?;
            let (report, evidence) = futures_lite::future::block_on(execute(request))?;
            if let Some(path) = &evidence_path {
                fs::write(path, evidence)?;
            }
            report
        }
        "analyze" => {
            let path = args.get(1).ok_or("evidence path required")?;
            let evidence = SpectroscopyEvidence::from_bytes(&fs::read(path)?)?;
            let analysis: AnalysisConfig = match args.get(2) {
                Some(path) => serde_json::from_str(&read(path)?)?,
                None => evidence.request.spectroscopy.analysis.clone(),
            };
            serde_json::to_value(analyze_evidence(&evidence, &analysis)?)?
        }
        "archive" => {
            let path = args.get(1).ok_or("request path required")?;
            let request: SpectroscopyRequest = serde_json::from_str(&read(path)?)?;
            let archives = args
                .get(2..)
                .filter(|rest| !rest.is_empty())
                .ok_or("at least one archive path required")?
                .iter()
                .map(|path| archive(path))
                .collect::<Fallible<Vec<_>>>()?;
            serde_json::to_value(analyze_archive(&request.spectroscopy, &archives)?)?
        }
        _ => return Err(format!("unknown command {command}; {USAGE}").into()),
    };
    let text = serde_json::to_string_pretty(&document)?;
    match output {
        Some(path) => fs::write(path, text)?,
        None => println!("{text}"),
    }
    Ok(())
}
