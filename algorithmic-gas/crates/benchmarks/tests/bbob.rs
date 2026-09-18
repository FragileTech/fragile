//! The Rust BBOB port against COCO 2.8.2 reference values.
use algorithmic_gas_benchmarks::bbob::{self, BbobProblem};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn close(actual: f64, expected: f64, relative: f64) -> bool {
    (actual - expected).abs() <= relative * expected.abs().max(1.)
}

#[test]
fn native_coco_goldens() {
    let goldens = fixture("objective-goldens.json");
    let mut worst: f64 = 0.;
    let mut failures = vec![];
    for case in goldens["bbob"].as_array().unwrap() {
        let function = case["function"].as_u64().unwrap() as u8;
        let d = case["dimensions"].as_u64().unwrap() as usize;
        let instance = case["instance"].as_u64().unwrap() as u32;
        let problem = BbobProblem::new(function, d, instance).unwrap();
        assert_eq!(problem.problem_id(), case["problem_id"].as_str().unwrap());
        let minimum = case["minimum"].as_f64().unwrap();
        if !close(problem.minimum(), minimum, 1e-9) {
            failures.push(format!(
                "f{function} d{d} i{instance} minimum {} != {minimum}",
                problem.minimum()
            ));
        }
        for (point, expected) in case["points"]
            .as_array()
            .unwrap()
            .iter()
            .zip(numbers(&case["values"]))
        {
            let actual = problem.evaluate(&numbers(point));
            let error = (actual - expected).abs() / expected.abs().max(1.);
            worst = worst.max(error);
            if !close(actual, expected, 1e-9) {
                failures.push(format!(
                    "f{function} d{d} i{instance}: {actual} != {expected}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures, first: {:#?}",
        failures.len(),
        &failures[..failures.len().min(12)]
    );
    assert!(worst < 1e-9, "worst relative error {worst:e}");
}

#[test]
fn upstream_bbob2009_testcases() {
    let upstream = fixture("coco-fixtures.json");
    let points: Vec<Vec<f64>> = upstream["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    let cases = upstream["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1296);
    for case in cases {
        // [function, dimension, instance, point index, value]
        let function = case[0].as_u64().unwrap() as u8;
        let d = case[1].as_u64().unwrap() as usize;
        let instance = case[2].as_u64().unwrap() as u32;
        let point = &points[case[3].as_u64().unwrap() as usize][..d];
        let expected = case[4].as_f64().unwrap();
        let actual = bbob::problem(function, d, instance)
            .unwrap()
            .evaluate(point);
        assert!(
            close(actual, expected, 4e-6),
            "f{function} d{d} i{instance}: {actual} != {expected}"
        );
    }
}

#[test]
fn reference_optimum_and_special_inputs() {
    for function in 1..=bbob::FUNCTIONS {
        for d in [2, 10] {
            let problem = BbobProblem::new(function, d, 3).unwrap();
            let minimum = problem.minimum();
            assert!(
                (minimum - problem.fopt).abs() <= 1e-7,
                "f{function} d{d}: f(best) = {minimum}, fopt = {}",
                problem.fopt
            );
            let mut rng = 0x9e3779b97f4a7c15u64;
            for _ in 0..32 {
                let point: Vec<f64> = (0..d)
                    .map(|_| {
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (rng >> 11) as f64 / (1u64 << 53) as f64 * 10. - 5.
                    })
                    .collect();
                assert!(problem.evaluate(&point) >= minimum - 1e-9);
            }
            let mut point = vec![0.5; d];
            point[1] = f64::NAN;
            assert!(problem.evaluate(&point).is_nan());
            point[0] = f64::NEG_INFINITY;
            assert_eq!(problem.evaluate(&point), f64::INFINITY);
        }
    }
    assert!(BbobProblem::new(0, 2, 1).is_err());
    assert!(BbobProblem::new(25, 2, 1).is_err());
    assert!(BbobProblem::new(1, 4, 1).is_err());
    assert!(BbobProblem::new(1, 2, 0).is_err());
    assert!(BbobProblem::new(1, 2, 1001).is_err());
}
