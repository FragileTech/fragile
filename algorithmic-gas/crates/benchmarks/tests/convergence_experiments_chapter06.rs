use algorithmic_gas_benchmarks::convergence_experiments_chapter06::{
    binomial_interval, law_rate_consistent, survivor_mean, swarm_cost,
};
#[test]
fn each_survivor_law_retains_its_own_denominator() {
    assert_eq!(
        survivor_mean(&[(true, 2.), (false, 1e9), (true, 4.)]),
        Some(3.)
    );
    assert_eq!(survivor_mean(&[(true, 10.), (false, 1e9)]), Some(10.));
    assert_eq!(survivor_mean(&[(false, 1e9)]), None);
}
#[test]
fn zero_rare_events_do_not_certify_zero_hazard() {
    let interval = binomial_interval(0, 256).unwrap();
    assert!(interval[0] < 1e-10);
    assert!(interval[1] > 0.01);
    assert!(binomial_interval(0, 1024).unwrap()[1] < interval[1]);
    assert!(binomial_interval(2, 1).is_err());
}
#[test]
fn rejects_a_wrong_declared_law_rate() {
    assert!(law_rate_consistent(0.9, 0.04, -0.9_f64.ln() / 0.04));
    assert!(!law_rate_consistent(0.9, 0.04, 10.));
    assert!(!law_rate_consistent(1., 0.04, 0.));
}
#[test]
fn state_cost_ignores_permutation_and_dead_positions_but_retains_collision_velocities() {
    let a = vec![vec![1., 0.5, 0.1], vec![0., 1e9, 0.2]];
    let b = vec![vec![0., -1e9, 0.2], vec![1., 0.5, 0.1]];
    assert!(swarm_cost(&a, &b).unwrap() < 1e-14);
    let c = vec![vec![0., -1e9, 0.4], vec![1., 0.5, 0.1]];
    assert!((swarm_cost(&a, &c).unwrap() - 0.02).abs() < 1e-12);
}
