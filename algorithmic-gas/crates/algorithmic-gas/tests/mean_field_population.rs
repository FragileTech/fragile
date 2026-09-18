use algorithmic_gas::{
    fitness::PositiveMapping,
    geometry::{AlgorithmicDistance, Distance, InteractionKernel},
    mean_field_population::{AtomicMeanField, MeanFieldLimits},
    *,
};
fn fixture() -> (Population<f64>, GasConfig) {
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(4, 1, vec![-0.6, 0.1, 0.5, 1.]).unwrap());
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(4, 1, vec![-1., 0.2, 0.7, 1.2]).unwrap(),
    );
    let mut p = Population::new(observations).unwrap();
    p.rewards.raw = vec![0.36, 0.01, 0.25, 1.];
    (p, GasConfig::euclidean(1, 0.02).unwrap())
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-12 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
#[test]
fn weighted_sampled_mark_normalizers_match_independent_quadrature() {
    let (mut p, c) = fixture();
    p.validity[0].out_of_bounds = true;
    let solver = AtomicMeanField::new(&p, &c, Default::default()).unwrap();
    close(solver.alive_mass, 0.75);
    assert_eq!(solver.types.len(), 10);
    close(solver.types.iter().map(|t| t.probability).sum(), 1.);
    let mut separations = Vec::new();
    for i in 1..4 {
        let mut weights = Vec::new();
        let mut measurements = Vec::new();
        for j in 1..4 {
            let distance = c
                .distance_donors
                .distance
                .compare(&p.observations, i, &p.observations, j)
                .unwrap();
            let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(
                &c.distance_donors.distance,
            );
            weights.push(
                c.distance_donors
                    .kernel
                    .log_weight(distance, kind)
                    .unwrap()
                    .exp(),
            );
            measurements.push((distance.powi(2) + c.fitness.distance_floor.powi(2)).sqrt());
        }
        let z = weights.iter().sum::<f64>();
        for j in 0..3 {
            separations.push((weights[j] / z / 3., measurements[j]));
        }
    }
    let mean = separations.iter().map(|(w, s)| w * s).sum::<f64>();
    let var = separations
        .iter()
        .map(|(w, s)| w * (s - mean).powi(2))
        .sum::<f64>();
    close(solver.diversity_normalizer.mean, mean);
    close(solver.diversity_normalizer.variance, var);
    let reward_mean = -(0.01 + 0.25 + 1.) / 3.;
    close(solver.reward_normalizer.mean, reward_mean);
    for t in &solver.types {
        if t.alive {
            let r: f64 = c
                .fitness
                .reward_map
                .map((t.oriented_reward - reward_mean) / solver.reward_normalizer.scale)
                .unwrap();
            let d: f64 = c
                .fitness
                .diversity_map
                .map((t.separation - mean) / solver.diversity_normalizer.scale)
                .unwrap();
            close(
                t.fitness,
                r.powf(c.fitness.reward_exponent) * d.powf(c.fitness.diversity_exponent),
            );
        }
    }
    // The full mark law has genuinely varying fitness within the same input atom.
    let first = solver
        .types
        .iter()
        .filter(|t| t.input_slot == 1)
        .collect::<Vec<_>>();
    assert!(
        first
            .iter()
            .any(|t| (t.fitness - first[0].fitness).abs() > 0.01)
    );
    let expected_fitness = solver
        .types
        .iter()
        .map(|t| t.probability * t.fitness)
        .sum::<f64>();
    let fitness_variance = solver
        .types
        .iter()
        .map(|t| t.probability * (t.fitness - expected_fitness).powi(2))
        .sum::<f64>();
    let observed_fitness = (0..8192)
        .map(|draw| {
            let sample = solver.sample_root(195, draw).unwrap();
            solver.types[sample.root_type].fitness
        })
        .sum::<f64>()
        / 8192.;
    assert!((observed_fitness - expected_fitness).abs() < 5. * (fitness_variance / 8192.).sqrt());
    // Every outgoing law has mass <=1, with mandatory weighted revival mass1.
    for (i, t) in solver.types.iter().enumerate() {
        let q = solver
            .types
            .iter()
            .enumerate()
            .map(|(j, u)| u.probability * solver.edge_density(i, j).unwrap())
            .sum::<f64>();
        assert!(q <= 1. + 1e-12);
        if !t.alive {
            close(q, 1.);
        }
    }
    let dead = solver.types.iter().position(|t| !t.alive).unwrap();
    let densities = solver
        .types
        .iter()
        .enumerate()
        .filter(|(_, t)| t.alive)
        .map(|(j, _)| solver.edge_density(dead, j).unwrap())
        .collect::<Vec<_>>();
    assert!(densities.iter().any(|v| (v - densities[0]).abs() > 0.01));
}
#[test]
fn equal_fitness_all_alive_is_singleton_and_retains_sampled_measurement_law() {
    let (p, mut c) = fixture();
    c.fitness.reward_exponent = 0.;
    c.fitness.diversity_exponent = 0.;
    let solver = AtomicMeanField::new(&p, &c, Default::default()).unwrap();
    let mut counts = vec![0_usize; solver.types.len()];
    for draw in 0..8192 {
        let s = solver.sample_root(843, draw).unwrap();
        counts[s.root_type] += 1;
        assert_eq!(s.component.len(), 1);
        assert_eq!(s.forced_incoming_children, 0);
        assert_eq!(s.free_outgoing_draws, 1);
        let slot = solver.types[s.root_type].input_slot;
        assert_eq!(
            s.positions,
            p.observations
                .field("positions")
                .unwrap()
                .row(slot)
                .unwrap()
        );
        assert_eq!(
            s.velocities,
            p.observations
                .field("velocities")
                .unwrap()
                .row(slot)
                .unwrap()
        );
    }
    for (t, n) in solver.types.iter().zip(counts) {
        let se = (8192. * t.probability * (1. - t.probability)).sqrt();
        assert!((n as f64 - 8192. * t.probability).abs() < 5. * se + 2.);
    }
}
#[test]
fn incoming_children_have_forced_parent_and_alpha_zero_uses_all_frozen_velocities() {
    let (mut p, mut c) = fixture();
    p.validity[0].terminated = true;
    c.fitness.reward_exponent = 0.;
    c.fitness.diversity_exponent = 0.;
    c.clone_transform.restitution = Some(0.);
    let solver = AtomicMeanField::new(&p, &c, Default::default()).unwrap();
    let mut sum = 0.;
    let mut squares = 0.;
    let mut nontrivial = 0;
    let mut revived = 0;
    for draw in 0..8192 {
        let s = solver.sample_root(399, draw).unwrap();
        let root_dead = !solver.types[s.root_type].alive;
        assert_eq!(s.free_outgoing_draws, if root_dead { 2 } else { 1 });
        assert_eq!(
            s.component.len(),
            s.free_outgoing_draws + s.forced_incoming_children
        );
        let edges = s.component.iter().filter(|v| v.outgoing.is_some()).count();
        assert_eq!(edges + 1, s.component.len());
        for v in &s.component {
            if v.outgoing_forced {
                assert!(!solver.types[v.type_index].alive);
                assert!(v.outgoing.is_some());
            }
            if let Some(parent) = v.outgoing {
                assert!(solver.types[s.component[parent].type_index].alive);
            }
        }
        let mean = s
            .component
            .iter()
            .map(|v| {
                p.observations.field("velocities").unwrap().values()
                    [solver.types[v.type_index].input_slot]
            })
            .sum::<f64>()
            / s.component.len() as f64;
        close(s.velocities[0], mean);
        close(s.center_of_mass[0], mean);
        if s.component.len() > 1 {
            nontrivial += 1;
        }
        if root_dead {
            revived += 1;
            assert!(s.root_donor_type.is_some());
            let donor = solver.types[s.root_donor_type.unwrap()].input_slot;
            assert_ne!(
                s.positions[0],
                p.observations.field("positions").unwrap().values()[donor]
            );
        }
        sum += s.velocities[0];
        squares += s.velocities[0].powi(2);
    }
    assert!(nontrivial > 1000 && revived > 1000);
    // Uniformly tagged roots preserve mean preclone momentum despite dead slots.
    let mean = sum / 8192.;
    let expected = p
        .observations
        .field("velocities")
        .unwrap()
        .values()
        .iter()
        .sum::<f64>()
        / 4.;
    let se = ((squares / 8192. - mean * mean) / 8191.).sqrt();
    assert!((mean - expected).abs() < 5. * se);
}
#[test]
fn capacity_failure_is_explicit_and_never_conditions_on_small_components() {
    let (mut p, c) = fixture();
    p.validity[0].terminated = true;
    let solver = AtomicMeanField::new(
        &p,
        &c,
        MeanFieldLimits {
            max_nodes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut failures = 0;
    for draw in 0..128 {
        if let Err(error) = solver.sample_root(299, draw) {
            assert!(error.to_string().contains("not truncated or retried"));
            failures += 1;
        }
    }
    assert!(failures > 20);
    let mut local = c.clone();
    local.clone_decision.revival_from_companion = false;
    assert!(AtomicMeanField::new(&p, &local, Default::default()).is_err());
}
