use algorithmic_gas_benchmarks::{
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_experiments_chapter04::run,
};

#[test]
fn native_cloning_reference_and_revival_records_are_reusable() {
    let root = std::env::temp_dir().join(format!(
        "fragile-chapter04-native-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut store = ArchiveStore::new(&root).unwrap();
    let config = ExperimentConfig {
        samples: 16,
        steps: 2,
        compact: true,
        seed: 202610041,
        archive_chunk_steps: 4,
    };
    let report = futures_lite::future::block_on(run(&config, &mut store)).unwrap();
    assert_eq!(report["summary"]["comparisons_failed"], 0);
    assert_eq!(report["summary"]["native_steps"], 164);
    let cases = report["cases"].as_array().unwrap();
    let permutation = cases
        .iter()
        .find(|c| c["profile"] == "permutation")
        .unwrap();
    assert!(
        permutation["comparisons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"].as_str().unwrap().contains("permutation_exact")
                && c["observed"].as_f64().unwrap().abs() < 1e-12)
    );
    for case in cases {
        if case["profile"] == "singleton" || case["profile"] == "revival" {
            let reset = case["comparisons"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| {
                    c["id"]
                        .as_str()
                        .unwrap()
                        .ends_with("proposal_centered_reset")
                })
                .unwrap();
            assert_eq!(
                reset["source_labels"][0],
                "thm-positional-variance-contraction"
            );
            assert!(reset.get("source_formula").is_none());
        }
        let path = case["raw_archives"][0]["left"].as_str().unwrap();
        let archive = store.load_archive(path).unwrap();
        assert_eq!(archive.steps.len(), 4);
        assert!(
            archive
                .anchors
                .iter()
                .any(|a| a.reason == "external_replace")
        );
        for step in &archive.steps {
            assert!(!step.report.distance_sources.is_empty());
            assert!(!step.report.clone_plan.sources.is_empty());
            assert_eq!(step.report.distance_companions.rows, 4);
            assert!(step.stages.iter().any(|s| s.stage == "post_transform"));
            if case["profile"] == "singleton" {
                assert_eq!(step.report.revivals, 3);
                assert_eq!(
                    step.report
                        .pre_clone_eligible
                        .iter()
                        .filter(|&&x| x)
                        .count(),
                    1
                );
                assert_eq!(
                    step.report
                        .clone_plan
                        .choices
                        .iter()
                        .filter(|c| c.revival)
                        .count(),
                    3
                );
            }
        }
        let frames = store
            .load_json(case["operator_frame_archives"][0].as_str().unwrap())
            .unwrap();
        assert_eq!(
            frames["frames"][0]["proposal"]["positions"][0]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert!(frames["frames"][0]["sampled_fitness"].is_array());
        assert!(frames["frames"][0]["centered_positional_plan"].is_array());
    }
    for reference in report["references"].as_array().unwrap() {
        assert_eq!(reference["N"], 200);
        assert_eq!(reference["d"], 3);
        let archive = store
            .load_archive(reference["raw_archives"][0]["archive"].as_str().unwrap())
            .unwrap();
        assert_eq!(archive.steps.len(), 2);
        assert_eq!(
            archive.steps[0]
                .influences
                .iter()
                .filter(|e| e.field == "viscous_force")
                .count(),
            2 * 200 * 199
        );
        assert!(reference["primitive"]["kappa_F"].as_f64().unwrap() > 0.98);
        assert!(
            reference["primitive"]["log_survival_floor_lower"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
    }
    store.finish("complete").unwrap();
    assert_eq!(store.verify(true).unwrap()["native_recorded_steps"], 164);
    std::fs::remove_dir_all(root).unwrap();
}
