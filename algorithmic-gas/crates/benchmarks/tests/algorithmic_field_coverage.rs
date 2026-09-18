use algorithmic_gas::{
    RecordingConfig,
    boundary::{BoundaryPolicy, BoxDomain},
    physics::field_evolution::{WeakFieldObservable, clone_field_balance, weak_field_balance},
    tracking::RunArchive,
};
use algorithmic_gas_benchmarks::lecture;
use serde_json::json;

fn observable() -> WeakFieldObservable {
    WeakFieldObservable::Momentum {
        k: vec![0.7, -0.4, 0.2],
        component: 0,
    }
}
async fn archive(boundary: Option<(&str, f64)>, tiny_positions: bool) -> RunArchive<f64> {
    let mut config = lecture::gas_config("VI-51", &json!({"walkers":16}), 7).unwrap();
    config.physics_metric = None;
    if tiny_positions {
        config.initial_lower = -1e-10;
        config.initial_upper = 1e-10;
    }
    if let Some((field, width)) = boundary {
        config.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: field.into(),
            domain: BoxDomain {
                lower: vec![-width; 3],
                upper: vec![width; 3],
            },
        };
    }
    let mut gas = config.build::<f64>().await.unwrap();
    gas.start_recording(RecordingConfig::default()).unwrap();
    gas.step().await.unwrap();
    gas.recording().unwrap().clone()
}

#[test]
fn every_required_stage_is_unique_ordered_and_present() {
    futures_lite::future::block_on(async {
        let archive = archive(None, false).await;
        let step = &archive.steps[0];
        weak_field_balance(&archive.gas_config, step, &observable()).unwrap();
        assert!(step.stages.iter().any(|s| s.stage == "B2"));
        for index in 0..step.stages.len() {
            let name = &step.stages[index].stage;
            let mut missing = step.clone();
            missing.stages.remove(index);
            assert!(
                weak_field_balance(&archive.gas_config, &missing, &observable()).is_err(),
                "missing {name}"
            );
            let mut duplicate = step.clone();
            duplicate
                .stages
                .insert(index, duplicate.stages[index].clone());
            assert!(
                weak_field_balance(&archive.gas_config, &duplicate, &observable()).is_err(),
                "duplicate {name}"
            );
            if index + 1 < step.stages.len() {
                let mut reordered = step.clone();
                reordered.stages.swap(index, index + 1);
                assert!(
                    weak_field_balance(&archive.gas_config, &reordered, &observable()).is_err(),
                    "reordered {name}"
                );
            }
        }
        for name in ["pre_clone", "literal_clone"] {
            let index = step.stages.iter().position(|s| s.stage == name).unwrap();
            let mut missing = step.clone();
            missing.stages.remove(index);
            assert!(clone_field_balance(&archive, &missing, &observable()).is_err());
            let mut duplicate = step.clone();
            duplicate.stages.push(duplicate.stages[index].clone());
            assert!(clone_field_balance(&archive, &duplicate, &observable()).is_err());
        }
        let mut swapped = step.clone();
        swapped.stages.swap(0, 1);
        assert!(clone_field_balance(&archive, &swapped, &observable()).is_err());
    });
}

#[test]
fn stage_provenance_and_untouched_coordinates_are_verified() {
    futures_lite::future::block_on(async {
        let archive = archive(None, false).await;
        let step = &archive.steps[0];
        for name in [
            "B1_before_boundary",
            "A1_before_boundary",
            "O_before_boundary",
            "A2_before_boundary",
            "B2_before_boundary",
        ] {
            let index = step.stages.iter().position(|s| s.stage == name).unwrap();
            let mut bad = step.clone();
            bad.stages[index].version = bad.stages[index - 1].version;
            assert!(
                weak_field_balance(&archive.gas_config, &bad, &observable()).is_err(),
                "version {name}"
            );
            let mut bad = step.clone();
            bad.stages[index].generations[0] += 1;
            assert!(
                weak_field_balance(&archive.gas_config, &bad, &observable()).is_err(),
                "generation {name}"
            );
            let mut bad = step.clone();
            let untouched = if name.starts_with('A') {
                "velocities"
            } else {
                "positions"
            };
            bad.stages[index].fields.get_mut(untouched).unwrap().values[0] += 1.;
            assert!(
                weak_field_balance(&archive.gas_config, &bad, &observable()).is_err(),
                "untouched {name}"
            );
        }
        for barrier in ["B1_input", "B2_input"] {
            let mut bad = step.clone();
            let input = bad.stages.iter_mut().find(|s| s.stage == barrier).unwrap();
            input.fields.get_mut("positions").unwrap().values[0] += 0.125;
            assert!(
                weak_field_balance(&archive.gas_config, &bad, &observable()).is_err(),
                "read-only barrier {barrier}"
            );
        }
        let mut bad = step.clone();
        bad.stages[0].fields.get_mut("positions").unwrap().values[0] += 1.;
        assert!(clone_field_balance(&archive, &bad, &observable()).is_err());
        let mut bad = step.clone();
        bad.stages
            .last_mut()
            .unwrap()
            .fields
            .get_mut("positions")
            .unwrap()
            .values[0] += 1.;
        assert!(weak_field_balance(&archive.gas_config, &bad, &observable()).is_err());
    });
}

#[test]
fn actual_absorbing_extinction_after_o_or_a2_allows_only_executed_stages() {
    futures_lite::future::block_on(async {
        for (field, terminal) in [("velocities", "O"), ("positions", "A2")] {
            let archive = archive(Some((field, 1e-6)), true).await;
            let step = &archive.steps[0];
            let terminal_stage = step.stages.iter().find(|s| s.stage == terminal).unwrap();
            assert!(
                terminal_stage
                    .validity
                    .iter()
                    .all(|v| !v.eligible(archive.gas_config.include_truncated))
            );
            assert!(!step.stages.iter().any(|s| s.stage == "B2"));
            if terminal == "O" {
                assert!(!step.stages.iter().any(|s| s.stage == "A2"));
            }
            let report = weak_field_balance(&archive.gas_config, step, &observable()).unwrap();
            assert!(
                report
                    .field_equation_residual
                    .iter()
                    .all(|v| v.abs() < 1e-12)
            );
            let expected = if terminal == "O" { 2 } else { 3 };
            assert_eq!(
                report
                    .stage_sources
                    .iter()
                    .filter(|s| s.deterministic_law_residual.is_some())
                    .count(),
                expected
            );
            let mut missing = step.clone();
            missing.stages.retain(|s| s.stage != terminal);
            assert!(weak_field_balance(&archive.gas_config, &missing, &observable()).is_err());
        }
    });
}

#[test]
fn inactive_raw_kinetic_rows_cannot_change_either_coordinate_field() {
    futures_lite::future::block_on(async {
        let archive = archive(Some(("velocities", 0.3)), true).await;
        let step = &archive.steps[0];
        weak_field_balance(&archive.gas_config, step, &observable()).unwrap();
        let index = step
            .stages
            .iter()
            .position(|s| s.stage == "A2_before_boundary")
            .expect("some walkers survive O");
        let row = step.stages[index - 1]
            .validity
            .iter()
            .position(|v| !v.eligible(archive.gas_config.include_truncated))
            .expect("some walkers die at O");
        for field in ["positions", "velocities"] {
            let mut bad = step.clone();
            bad.stages[index].fields.get_mut(field).unwrap().values[row * 3] += 1.;
            assert!(
                weak_field_balance(&archive.gas_config, &bad, &observable()).is_err(),
                "inactive {field}"
            );
        }
    });
}

#[test]
fn configured_position_diffusion_cap_and_terminal_boundary_have_required_coverage() {
    futures_lite::future::block_on(async {
        for schedule in [
            algorithmic_gas::kinetic::KineticBoundarySchedule::Substeps,
            algorithmic_gas::kinetic::KineticBoundarySchedule::EndOfStep,
        ] {
            let mut config = lecture::gas_config("VI-51", &json!({"walkers":8}), 7).unwrap();
            config.physics_metric = None;
            config.gas.kinetic.position_diffusion = 0.2;
            config.gas.kinetic.velocity_cap = Some(0.8);
            config.gas.kinetic.boundary_schedule = schedule;
            let mut gas = config.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            gas.step().await.unwrap();
            let step = &gas.recording().unwrap().steps[0];
            let result = weak_field_balance(&config.gas, step, &observable()).unwrap();
            assert!(
                result
                    .stage_sources
                    .iter()
                    .any(|s| s.kind == "position_diffusion")
            );
            let cap = result
                .stage_sources
                .iter()
                .find(|s| s.kind == "velocity_cap")
                .unwrap();
            assert!(
                cap.deterministic_law_residual
                    .unwrap()
                    .iter()
                    .all(|x| x.abs() < 1e-12)
            );
            let mut names = vec![
                "position_diffusion_before_boundary",
                "position_diffusion",
                "velocity_cap_before_boundary",
                "velocity_cap",
            ];
            if schedule == algorithmic_gas::kinetic::KineticBoundarySchedule::EndOfStep {
                names.extend(["terminal_before_boundary", "terminal"]);
            }
            for name in names {
                let mut bad = step.clone();
                bad.stages.retain(|s| s.stage != name);
                assert!(
                    weak_field_balance(&config.gas, &bad, &observable()).is_err(),
                    "missing {name}"
                );
            }
        }
    });
}
