//! Exact Chapter 9 fixed-parameter constant ledger (no simulated kernel substitution).
use algorithmic_gas_benchmarks::{
    convergence_chapter09_completion::*,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
fn native_comparisons(
    path: &str,
    output: &mut ArchiveStore,
) -> Result<Value, Box<dyn std::error::Error>> {
    let input = ArchiveStore::open(path)?;
    if input.status() != "complete" {
        return Err("Chapter8 native input dataset must be complete".into());
    }
    let data: Value = serde_json::from_reader(fs::File::open(input.root().join("report.json"))?)?;
    let cases = data["cases"]
        .as_array()
        .ok_or("missing actual native cases")?;
    let failure = 0.01 / (cases.len() * 7 * 3) as f64;
    let mut rows = vec![];
    let mut failed = 0;
    let mut count = 0;
    for case in cases {
        let n = case["N"].as_u64().ok_or("N")? as usize;
        let alive = case["entering_alive"].as_u64().ok_or("alive")? as usize;
        let cfg: algorithmic_gas::GasConfig =
            serde_json::from_value(case["actual_configuration"]["gas"].clone())?;
        let constants = constants_from_native(&cfg, alive as f64 / n as f64)?;
        let raw = input.load_json(
            case["native_operands"]
                .as_str()
                .ok_or("native operands path")?,
        )?;
        let reference =
            input.load_json(case["root_reference"].as_str().ok_or("reference path")?)?;
        let actual: Vec<[f64; 7]> = raw["raw"]
            .as_array()
            .ok_or("actual rows")?
            .iter()
            .map(|r| serde_json::from_value(r["output_tests"].clone()))
            .collect::<std::result::Result<_, _>>()?;
        let root: Vec<[f64; 7]> = reference["rows"]
            .as_array()
            .ok_or("rooted rows")?
            .iter()
            .map(|r| serde_json::from_value(r["output_tests"].clone()))
            .collect::<std::result::Result<_, _>>()?;
        if actual.len() != case["native_replicas"].as_u64().ok_or("replicas")? as usize
            || root.len() != case["root_reference_draws"].as_u64().ok_or("root draws")? as usize
            || actual.len() < 2
            || root.len() < 2
        {
            return Err("independent-replica denominators must match complete raw data".into());
        }
        if case["reference_truncation_probability"] != 0.
            || reference["truncation_probability"] != 0.
        {
            return Err("no truncated/capacity-selected reference law permitted".into());
        }
        let (mut am, mut av) = ([0.; 7], [0.; 7]);
        let (mut rm, mut rv) = ([0.; 7], [0.; 7]);
        for j in 0..7 {
            am[j] = actual.iter().map(|r| r[j]).sum::<f64>() / actual.len() as f64;
            rm[j] = root.iter().map(|r| r[j]).sum::<f64>() / root.len() as f64;
            av[j] = actual.iter().map(|r| (r[j] - am[j]).powi(2)).sum::<f64>()
                / (actual.len() - 1) as f64;
            rv[j] =
                root.iter().map(|r| (r[j] - rm[j]).powi(2)).sum::<f64>() / (root.len() - 1) as f64;
        }
        let e = ((8. / failure).ln() / (2. * actual.len() as f64)).sqrt();
        let r = ((8. / failure).ln() / (2. * root.len() as f64)).sqrt();
        let vb = constants.variance_upper(n, 1.)?;
        let bb = constants.bias_upper(n, 1.)?;
        let mb = constants.mean_square_upper(n, 1.)?;
        let mut checks = vec![];
        for j in 0..7 {
            let bias = (am[j] - rm[j]).abs();
            let mse =
                actual.iter().map(|a| (a[j] - rm[j]).powi(2)).sum::<f64>() / actual.len() as f64;
            for (name, observed, bound, universal, allowance, label) in [
                (
                    "conditional-variance",
                    av[j],
                    &vb,
                    1.,
                    5. * e + 1. / (actual.len() - 1) as f64,
                    "thm-chaos-canonical-conditional-variance",
                ),
                (
                    "conditional-bias",
                    bias,
                    &bb,
                    2.,
                    2. * e + 2. * r,
                    "thm-chaos-canonical-quantitative-bias",
                ),
                (
                    "conditional-mse",
                    mse,
                    &mb,
                    4.,
                    4. * e + 8. * r,
                    "thm-chaos-canonical-quantitative-bias",
                ),
            ] {
                let comparable = bound.upper_float.unwrap_or(universal).min(universal);
                let passed = observed <= comparable + allowance + 1e-12;
                failed += usize::from(!passed);
                count += 1;
                checks.push(json!({"kind":name,"test_index":j,"observed":observed,"analytic_bound":bound,
      "bounded_test_universal_upper":universal,"comparison_upper":comparable,"allowance":allowance,"passed":passed,
      "source_label":label,"per_comparison_failure_budget":failure,"family_failure_budget":0.01,
      "scope":"Full native Haar/BAOAB output against independent untruncated AtomicMeanField rooted population law at the same entering empirical measure. Hoeffding uncertainty uses independent native updates and rooted draws, not N row replicates. The configuration-only worst-case constants may exceed the universal bounded-test bound; this is explicitly retained, not called a sharp empirical rate."}));
            }
        }
        let operands=output.save_json(&format!("{}-population-operands",case["id"].as_str().ok_or("case id")?),&json!({
   "native_values":actual,"root_values":root,"actual_mean":am,"actual_variance":av,"root_mean":rm,"root_variance":rv,
   "native_input_paths":{"operands":case["native_operands"],"reference":case["root_reference"],"checkpoint":case["input_checkpoint"]},
   "native_replicas":actual.len(),"root_draws":root.len(),"N":n,"d":case["d"],"actual_configuration":cfg,"constants":constants,
   "checks":checks,"no_new_native_steps":true}))?;
        rows.push(json!({"id":case["id"],"N":n,"d":case["d"],"profile":case["profile"],"entering_alive":alive,
   "native_replicas":actual.len(),"root_draws":root.len(),"native_mean":am,"native_variance":av,"reference_mean":rm,
   "reference_variance":rv,"N_times_empirical_variance":av.map(|x|x*n as f64),"constants":constants,"checks":checks,"operands":operands}));
    }
    Ok(
        json!({"input_dataset":input.root(),"input_index_sha256":sha256_file(&input.root().join("archive-index.json"))?,
  "input_report_sha256":sha256_file(&input.root().join("report.json"))?,"cases":rows,
  "summary":{"cases":rows.len(),"comparisons":count,"failed":failed,"reused_native_updates":data["summary"]["new_complete_native_updates"],"new_native_updates":0},
  "independence":"Different native seeds for whole frozen-input updates and different independent root-address draws. Physical row order is never an independent-replica denominator."}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err("gas-chapter09-completion EMPTY_OUTPUT_DIR [CHAPTER08_NATIVE_DATASET]".into());
    }
    let mut output = ArchiveStore::new(&args[0])?;
    output.save_json("chapter09-source",&json!({"module":include_str!("../convergence_chapter09_completion.rs"),
  "runner":include_str!("gas-chapter09-completion.rs"),
  "chapter09":include_str!("../../../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"),
  "binary_sha256":sha256_file(&std::env::current_exe()?)?,"new_native_steps":0}))?;
    let mut profiles = vec![];
    let mut checks = vec![];
    for m in [0.25, 0.5, 1.] {
        for k in [0.25, 0.8, 1.] {
            for sigma in [0.1, 0.5] {
                for exponent in [0., 0.5, 1., 2.] {
                    let constants = quantitative_constants(ConstantsInput {
                        minimum_alive_fraction: m,
                        measurement_weight_lower: k,
                        clone_weight_lower: k,
                        separation_upper: 3.,
                        separation_scale_floor: sigma,
                        reward_map_floor: 0.1,
                        reward_map_amplitude: 2.,
                        separation_map_floor: 0.1,
                        separation_map_amplitude: 2.,
                        reward_exponent: exponent,
                        separation_exponent: exponent,
                        acceptance_epsilon: 1e-6,
                        acceptance_saturation: 1.,
                    })?;
                    let mut rates = vec![];
                    for n in [2, 8, 32, 128, 512, 2048] {
                        let v = constants.variance_upper(n, 1.)?;
                        let bias = constants.bias_upper(n, 1.)?;
                        let mse = constants.mean_square_upper(n, 1.)?;
                        let v4 = constants.variance_upper(4 * n, 1.)?;
                        let bias4 = constants.bias_upper(4 * n, 1.)?;
                        checks.push(json!({"kind":"N-independent variance scaling","N":n,"observed":v.log_upper-v4.log_upper,
    "bound":4_f64.ln(),"passed":(v.log_upper-v4.log_upper-4_f64.ln()).abs()<1e-9,
    "source_label":"thm-chaos-canonical-conditional-variance","scope":"logarithmic constant algebra"}));
                        checks.push(json!({"kind":"N-independent bias scaling","N":n,"observed":bias.log_upper-bias4.log_upper,
    "bound":2_f64.ln(),"passed":(bias.log_upper-bias4.log_upper-2_f64.ln()).abs()<1e-9,
    "source_label":"thm-chaos-canonical-quantitative-bias","scope":"logarithmic constant algebra"}));
                        rates.push(json!({"N":n,"conditional_variance_upper":v,"conditional_bias_upper":bias,"conditional_mse_upper":mse}));
                    }
                    let moments = [1, 2, 3, 4, 8, 16, 32, 64, 128]
                        .into_iter()
                        .map(|p| {
                            Ok(json!({"order":p,"log_upper":log_component_moment(constants.c,p)?}))
                        })
                        .collect::<algorithmic_gas::Result<Vec<_>>>()?;
                    profiles.push(json!({"constants":constants,"rates":rates,"component_log_moments":moments}));
                }
            }
        }
    }
    let native = args
        .get(1)
        .map(|p| native_comparisons(p, &mut output))
        .transpose()?;
    let native_failed = native
        .as_ref()
        .and_then(|v| v["summary"]["failed"].as_u64())
        .unwrap_or(0) as usize;
    let failed = checks.iter().filter(|x| x["passed"] != true).count() + native_failed;
    let report = json!({"summary":{"profiles":profiles.len(),"checks":checks.len(),"failed":failed,"new_native_steps":0},
  "profiles":profiles,"checks":checks,"native_population_comparisons":native,
  "scope":"Exact fixed-parameter algebra of the displayed constants; independent of N. Actual native population/reference observations are bound in the separate source-scoped comparison report. A large bound is preserved even when practically uninformative; no fitting of these constants to observations."});
    output.save_json("chapter09-constants", &report)?;
    output.finish("complete")?;
    fs::write(
        PathBuf::from(&args[0]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let verify = output.verify(true)?;
    fs::write(
        PathBuf::from(&args[0]).join("verification.json"),
        serde_json::to_vec_pretty(&verify)?,
    )?;
    println!("{}", report["summary"]);
    if failed > 0 {
        return Err("constant scaling failure".into());
    }
    Ok(())
}
