//! Export source-bound primitive JSON structural rate registers.
use algorithmic_gas_benchmarks::{
    convergence_experiments::sha256_file,
    convergence_structural_mixing::{
        KernelHypotheses, KineticParameters, LandscapeProfiles, SelectionParameters, evaluate,
        rastrigin_source_example,
    },
};
use serde_json::json;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("Usage: gas-structural-mixing [OUTPUT_JSON] [PRIMITIVES_JSON]".into());
    }
    let source=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md");
    let cases = if args.len() == 2 {
        let inputs: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let kinetic: KineticParameters = serde_json::from_value(inputs["kinetic"].clone())?;
        let selection: SelectionParameters = serde_json::from_value(inputs["selection"].clone())?;
        let profiles: LandscapeProfiles = serde_json::from_value(inputs["profiles"].clone())?;
        let hypotheses: KernelHypotheses = serde_json::from_value(inputs["hypotheses"].clone())?;
        vec![
            json!({"inputs":inputs,"computed_register":evaluate(&kinetic,&selection,&profiles,&hypotheses)?}),
        ]
    } else {
        [1, 2, 4, 8]
            .into_iter()
            .map(rastrigin_source_example)
            .collect::<algorithmic_gas::Result<Vec<_>>>()?
    };
    let text = std::fs::read_to_string(&source)?;
    let mut quotes = Vec::new();
    for label in [
        "def-slcc-regime",
        "lem-slcc-base-mixing",
        "lem-slcc-selection-perturbation",
        "thm-slcc-active-contraction",
        "def-slcw-regime",
        "lem-slcw-normalization",
        "lem-slcw-selection-perturbation",
        "thm-slcw-active-contraction",
        "cor-slcw-rastrigin-uniform-law",
    ] {
        let start = text
            .find(&format!(":label: {label}\n"))
            .ok_or("Source label missing")?;
        let end = start
            + text[start..]
                .find("\n:::")
                .ok_or("Source statement terminator missing")?;
        quotes.push(json!({"label":label,"source_quotes":[&text[start..end]]}));
    }
    let report = json!({"source_path":source,"source_sha256":sha256_file(&source)?,"source_register":quotes,"cases":cases,"native_steps":0,"scope":"Primitive source formulas evaluated with explicit actual-force and full-kernel gates; not fitted transition matrices, interval certification or new trajectory evidence"});
    let output = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = args.first() {
        std::fs::write(path, output)?;
    } else {
        println!("{}", String::from_utf8(output)?);
    }
    Ok(())
}
