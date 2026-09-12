// Execute the registered scientific experiment shared with the lecture worker.
use algorithmic_gas::lecture::{LectureRequest, catalog, stress_cases};
use algorithmic_gas_benchmarks::lecture::{ExperimentEvidence, LectureSession, analyze_evidence};
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, Read, Write},
};
fn read(path: &str) -> std::result::Result<String, Box<dyn std::error::Error>> {
    if path == "-" {
        let mut s = String::new();
        io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else {
        Ok(fs::read_to_string(path)?)
    }
}
async fn execute(request: LectureRequest) -> algorithmic_gas::Result<Value> {
    let mut session = LectureSession::create(request).await?;
    while !session.done() {
        session.advance(8).await?;
    }
    Ok(json!({"snapshot":session.snapshot()?,"evidence":session.evidence()}))
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let output = if let Some(i) = args.iter().position(|s| s == "--output") {
        if i + 1 >= args.len() {
            return Err("--output path required".into());
        }
        let p = args.remove(i + 1);
        args.remove(i);
        Some(p)
    } else {
        None
    };
    let command=args.first().ok_or("Usage: gas-lecture catalog | run REQUEST.json | analyze EVIDENCE.json | stress ID [START] [COUNT] | ID [SEED] [STEPS]")?;
    let result = match command.as_str() {
        "catalog" => json!(catalog()),
        "run" => {
            let r: LectureRequest =
                serde_json::from_str(&read(args.get(1).ok_or("request path required")?)?)?;
            futures_lite::future::block_on(execute(r))?
        }
        "analyze" => {
            let v: Value =
                serde_json::from_str(&read(args.get(1).ok_or("evidence path required")?)?)?;
            let e: ExperimentEvidence =
                serde_json::from_value(v.get("evidence").unwrap_or(&v).clone())?;
            json!(futures_lite::future::block_on(analyze_evidence(&e))?)
        }
        "stress" => {
            let id = args.get(1).ok_or("stress ID required")?;
            let cases = stress_cases(id)?;
            let start = args
                .get(2)
                .map(|s| s.parse::<usize>())
                .transpose()?
                .unwrap_or(0);
            let count = args
                .get(3)
                .map(|s| s.parse::<usize>())
                .transpose()?
                .unwrap_or(cases.len());
            let mut results = vec![];
            for (i, r) in cases.into_iter().enumerate().skip(start).take(count) {
                let record = match futures_lite::future::block_on(execute(r.clone())) {
                    Ok(v) => {
                        json!({"case":i,"request":r,"status":"completed","result":v["snapshot"]["result"]})
                    }
                    Err(e) => json!({"case":i,"request":r,"status":"error","error":e.to_string()}),
                };
                eprintln!("{} case {}: {}", id, i, record["status"]);
                results.push(record);
            }
            json!({"experiment":id,"cases":results})
        }
        id => {
            let seed = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(7);
            let steps = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(96);
            futures_lite::future::block_on(execute(LectureRequest {
                id: id.into(),
                seed,
                parameters: json!({}),
                steps,
            }))?
        }
    };
    let encoded = serde_json::to_vec(&result)?;
    if let Some(path) = output {
        fs::write(path, encoded)?;
    } else {
        io::stdout().write_all(&encoded)?;
        io::stdout().write_all(b"\n")?;
    }
    if result
        .get("cases")
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|r| r["status"] == "error"))
    {
        return Err("Some stress cases failed; inspect the result artifact".into());
    }
    Ok(())
}
