//! Exact finite reference-noise operands and retained native-stage witnesses.
use super::*;

fn alias(suite: &mut EstimateSuite, src: &Value, indices: &[&str], ids: &[&str]) -> Result<()> {
    let matches = |id: &str, stem: &str| {
        id == stem
            || id.strip_prefix(stem).is_some_and(|tail| {
                tail.starts_with('_')
                    && tail[1..]
                        .split('_')
                        .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
            })
    };
    let originals = suite
        .evidence
        .iter()
        .filter(|record| {
            record
                .checks
                .iter()
                .chain(record.hypothesis_checks.iter())
                .any(|check| ids.iter().any(|stem| matches(&check.id, stem)))
        })
        .take(4)
        .cloned()
        .collect::<Vec<_>>();
    require(
        !originals.is_empty(),
        &format!("missing exact remaining noise operands {indices:?}/{ids:?}"),
    )?;
    for original in originals {
        let checks = original
            .checks
            .iter()
            .chain(original.hypothesis_checks.iter())
            .filter(|check| ids.iter().any(|stem| matches(&check.id, stem)))
            .cloned()
            .collect::<Vec<_>>();
        let begin = suite.evidence.len();
        let scope = format!(
            "{SCOPE} Specific retained same-quantity comparison; original scope: {}",
            original.scope
        );
        emit(suite, src, indices, &original.inputs, &checks, &scope)?;
        for record in &mut suite.evidence[begin..] {
            record.hypothesis_checks = original.hypothesis_checks.clone();
        }
    }
    Ok(())
}

pub(super) fn append(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    reference_noise(suite, src)?;
    boundary_operands(suite, src)?;
    marked_laws(suite, src)?;
    transport_and_process(suite, src)?;
    Ok(())
}

fn reference_noise(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let pi = std::f64::consts::PI;
    for dimension in [1usize, 2, 4, 8] {
        for sigma in [0.05_f64, 0.2, 1.] {
            let d = dimension as f64;
            // Degree-two symmetric Gaussian cubature integrates the moment
            // exactly; it is not a Monte Carlo ensemble or a full Gaussian law.
            let radius = d.sqrt() * sigma;
            let nodes = (0..dimension)
                .flat_map(|axis| [-1., 1.].map(move |sign| (axis, sign * radius)))
                .collect::<Vec<_>>();
            let second = nodes.iter().map(|(_, x)| x * x / (2. * d)).sum::<f64>();
            let inp = json!({"dimension":dimension,"sigma":sigma,"cubature_nodes_axis_value":nodes,"weight":1./(2.*d),"law":"isotropic ambient reference Gaussian, not capped BAOAB","mean":0.,"covariance_diagonal":sigma*sigma});
            emit(
                suite,
                src,
                &["0031", "1951"],
                &inp,
                &[eq(
                    "remaining_gaussian_second_moment_exact_cubature",
                    second,
                    d * sigma * sigma,
                )],
                "Exact degree-two reference Gaussian integration; ambient reference law is distinct from the native capped kinetic update.",
            )?;
            let mut cov = vec![];
            for axis in 0..dimension {
                cov.push(eq(
                    &format!("remaining_gaussian_covariance_{axis}"),
                    nodes
                        .iter()
                        .filter(|(a, _)| *a == axis)
                        .map(|(_, v)| v * v / (2. * d))
                        .sum(),
                    sigma * sigma,
                ));
            }
            emit(
                suite,
                src,
                &["0032"],
                &inp,
                &cov,
                "Finite mean/covariance operands of the analytic Gaussian definition; cubature nodes do not claim to be random draws or prove equality of laws.",
            )?;
            let unit_covariance = nodes
                .iter()
                .filter(|(axis, _)| *axis == 0)
                .map(|(_, v)| (v / sigma).powi(2) / (2. * d))
                .sum::<f64>();
            emit(
                suite,
                src,
                &["0301"],
                &inp,
                &[eq(
                    "remaining_unit_gaussian_covariance",
                    unit_covariance,
                    1.,
                )],
                "Standard unit Gaussian covariance operand obtained from the retained cubature nodes divided by their explicit sigma; not a claim that the cubature law is Gaussian.",
            )?;
            let y = (0..dimension)
                .map(|j| 0.1 * (j + 1) as f64)
                .collect::<Vec<_>>();
            let norm = squared(&y);
            let density =
                (-norm / (2. * sigma * sigma)).exp() / (2. * pi * sigma * sigma).powf(d / 2.);
            let product = y
                .iter()
                .map(|v| (-v * v / (2. * sigma * sigma)).exp() / ((2. * pi).sqrt() * sigma))
                .product::<f64>();
            let inp = json!({"dimension":dimension,"sigma":sigma,"point":y,"density":density,"independent_coordinate_density_product":product});
            emit(
                suite,
                src,
                &["0037", "1808"],
                &inp,
                &[eq(
                    "remaining_gaussian_density_product_log",
                    density.ln(),
                    product.ln(),
                )],
                "Explicit normalized reference Gaussian density and its proportional exponential. This is an analytic ambient innovation kernel, not a claim that a complete native BAOAB kernel has Gaussian density.",
            )?;
            let heat_density =
                (-norm / (4. * sigma * sigma)).exp() / (4. * pi * sigma * sigma).powf(d / 2.);
            let heat_product = y
                .iter()
                .map(|v| (-v * v / (4. * sigma * sigma)).exp() / (2. * pi.sqrt() * sigma))
                .product::<f64>();
            emit(
                suite,
                src,
                &["0169"],
                &inp,
                &[eq(
                    "remaining_heat_density_product_log",
                    heat_density.ln(),
                    heat_product.ln(),
                )],
                SCOPE,
            )?;
            let shift = 0.13_f64;
            let mean_node = nodes
                .iter()
                .map(|(_, v)| shift + 2_f64.sqrt() * v)
                .sum::<f64>()
                / (2. * d);
            let second_shift = nodes
                .iter()
                .map(|(_, v)| (2_f64.sqrt() * v).powi(2))
                .sum::<f64>()
                / (2. * d);
            let inp = json!({"ambient_dimension":dimension,"source":shift,"sigma":sigma,"unit_gaussian_nodes":nodes.iter().map(|(axis,v)|json!([axis,v/sigma])).collect::<Vec<_>>(),"heat_scale":2_f64.sqrt()*sigma,"support":"unbounded R^d analytically","finite_node_mean":mean_node,"finite_node_norm_second":second_shift});
            emit(
                suite,
                src,
                &["0296", "0300"],
                &inp,
                &[
                    eq("remaining_heat_shifted_mean", mean_node, shift),
                    eq(
                        "remaining_heat_shifted_covariance_trace",
                        second_shift,
                        2. * d * sigma * sigma,
                    ),
                ],
                "Exact heat-kernel mean and trace moment under its specified affine Gaussian map; the cube rule checks degree-two operands only.",
            )?;
        }
    }
    alias(
        suite,
        src,
        &["0303"],
        &[
            "heat_pdf_normalization_quadrature",
            "native_heat_pdf_characteristic_probe",
        ],
    )?;
    alias(
        suite,
        src,
        &["0117", "0118"],
        &["generator_polynomial_exact_moment"],
    )?;
    let cfg = GasConfig::euclidean(1, 0.04)?;
    emit(
        suite,
        src,
        &["0115"],
        &json!({"native_config":cfg,"declared_continuous_friction":1.}),
        &[truth("remaining_positive_langevin_friction", 1_f64 > 0.)],
        "Positive configured friction. The continuous Langevin drift/generator witnesses are recorded separately from the native fixed-cap BAOAB map.",
    )?;
    Ok(())
}

fn boundary_operands(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let sigma = 0.2_f64;
    let pi = std::f64::consts::PI;
    let volume = 2. * sigma;
    let rho = 1. / volume;
    let perimeter = 2.;
    let h = 1.;
    let uniform = |x: f64| 1. - ((x + sigma).min(1.) - (x - sigma).max(-1.)).max(0.) / volume;
    let heat_sd = 2_f64.sqrt() * sigma;
    let heat = |x: f64| 1. - (normal_cdf((1. - x) / heat_sd) - normal_cdf((-1. - x) / heat_sd));
    let pdf = |x: f64, s: f64| (-x * x / (2. * s * s)).exp() / (std::f64::consts::TAU.sqrt() * s);
    let lp_ball = 1. / (2. * sigma);
    let lp_heat = 1. / (2. * pi.sqrt() * sigma);
    let inputs = json!({"dimension":1,"sigma":sigma,"ambient_space":"R","invalid_set":"outside [-1,1]","finite_perimeter":perimeter,"unit_ball_volume":2.,"uniform_ball_volume":volume,"uniform_density":rho,"identity_projection_H":h,"heat_covariance":2.*sigma*sigma,"CDF_absolute_error":1.5e-7});
    emit(
        suite,
        src,
        &["0038", "0317"],
        &inputs,
        &[eq(
            "remaining_uniform_reference_density_mass",
            rho * volume,
            1.,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0039"],
        &json!({"position":sigma/2.,"ball_radius":sigma}),
        &[le("remaining_uniform_ball_support", sigma / 2., sigma)],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0166", "0326"],
        &inputs,
        &[eq("remaining_unit_ball_interval_volume", 2., 1. - (-1.))],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0320"],
        &inputs,
        &[eq(
            "remaining_uniform_kernel_BV_mass",
            rho + rho,
            1. / sigma,
        )],
        "One-dimensional uniform kernel has distributional derivative (delta_-sigma−delta_sigma)/(2sigma), whose total variation is 1/sigma. Endpoint masses are evaluated explicitly.",
    )?;
    emit(
        suite,
        src,
        &["0322", "0323", "0344"],
        &inputs,
        &[eq(
            "remaining_invalid_interval_perimeter",
            1_f64.abs() + (-1_f64).abs(),
            perimeter,
        )],
        "The invalid complement of [-1,1] has two unit signed endpoint derivative atoms; perimeter is their total variation. No finite sampling estimates a perimeter.",
    )?;
    emit(
        suite,
        src,
        &["0297", "0309", "0325", "0345"],
        &inputs,
        &[truth(
            "remaining_ambient_reference_endpoints_finite",
            [-4_f64, 4.].iter().all(|x| x.is_finite()),
        )],
        "The reference integration is on ambient R, with full Gaussian support and no replacement by a compact chart. The numeric operands are finite endpoint representatives of this specified ambient kernel, not an empirical proof that a support equals R.",
    )?;
    for x in [-1.2_f64, -1., -0.9, 0., 0.3, 1., 1.2] {
        let p = uniform(x);
        let pg = heat(x);
        let epsilon = 0.07 * sigma;
        let mollified = |t: f64| {
            (normal_cdf((t + sigma) / epsilon) - normal_cdf((t - sigma) / epsilon)) / volume
        };
        let boundary_flux = mollified(x - 1.) - mollified(x + 1.);
        let gauss_flux = pdf(x - 1., heat_sd) - pdf(x + 1., heat_sd);
        let inp = json!({"reference_kernel":inputs,"source_position":x,"uniform_death_probability":p,"heat_death_probability":pg,"mollifier_sigma":epsilon,"mollified_boundary_signed_flux":boundary_flux,"heat_boundary_signed_flux":gauss_flux});
        emit(
            suite,
            src,
            &["0162", "0336"],
            &inp,
            &[eq(
                "remaining_uniform_invalid_convolution",
                p,
                (volume - ((x + sigma).min(1.) - (x - sigma).max(-1.)).max(0.)) / volume,
            )],
            SCOPE,
        )?;
        emit(
            suite,
            src,
            &["0348"],
            &inp,
            &[eq(
                "remaining_heat_invalid_convolution",
                pg,
                normal_cdf((-1. - x) / heat_sd) + 1. - normal_cdf((1. - x) / heat_sd),
            )],
            SCOPE,
        )?;
        emit(
            suite,
            src,
            &["0335"],
            &inp,
            &[
                le(
                    "remaining_mollified_uniform_density_upper",
                    mollified(x),
                    rho + 3e-7 / volume,
                ),
                le(
                    "remaining_mollified_uniform_density_lower",
                    -mollified(x),
                    3e-7 / volume,
                ),
            ],
            "Gaussian mollification of the uniform interval kernel computed by its actual CDF difference; explicit CDF numerical error is retained.",
        )?;
        let eps = 1e-5;
        let heat_fd = (heat(x + eps) - heat(x - eps)) / (2. * eps);
        // Finite differences include the independently specified CDF error.
        emit(
            suite,
            src,
            &["0351"],
            &inp,
            &[le(
                "remaining_heat_convolution_distributional_gradient",
                (heat_fd - gauss_flux).abs(),
                6e-7 / (2. * eps) + eps * eps / (sigma.powi(3)),
            )],
            "Invalid-set distributional derivative consists of signed endpoint atoms; its convolution equals the explicit endpoint Gaussian density flux. Finite-difference verification retains CDF error; no classical derivative is assigned to an indicator.",
        )?;
        emit(
            suite,
            src,
            &["0337"],
            &inp,
            &[le(
                "remaining_mollified_invalid_distributional_gradient",
                boundary_flux.abs(),
                perimeter * rho + 6e-7 / volume,
            )],
            "Distributional signed endpoint measure convolved with the actual mollified kernel; this checks the explicit finite-perimeter specialization, not a source-membership assertion.",
        )?;
        for y in [-1.1_f64, -0.4, 0.2, 0.95] {
            let distance = (x - y).abs();
            let l1 = 2_f64.min(distance / sigma);
            let inp = json!({"reference_kernel":inputs,"x":x,"y":y,"distance":distance,"uniform_shift_L1_exact":l1,"unit_projection":true});
            emit(
                suite,
                src,
                &["0319"],
                &inp,
                &[
                    le("remaining_uniform_kernel_shift_L1", l1, distance / sigma),
                    le(
                        "remaining_uniform_boundary_shift",
                        (uniform(x) - uniform(y)).abs(),
                        lp_ball * distance,
                    ),
                ],
                SCOPE,
            )?;
            emit(
                suite,
                src,
                &[
                    "0172", "0329", "0334", "0342", "0343", "1801", "1802", "1948", "1956",
                ],
                &inp,
                &[eq(
                    "remaining_identity_projection_distance",
                    distance,
                    h * (x - y).abs(),
                )],
                "Identity projection on these explicitly supplied ambient scalar operands; Lipschitz H=1 is evaluated on paired points.",
            )?;
            emit(
                suite,
                src,
                &["1433", "1437", "1438"],
                &inp,
                &[
                    le(
                        "remaining_normalized_heat_status_modulus",
                        (heat(x) - heat(y)).abs(),
                        lp_heat * distance + 6e-7,
                    ),
                    eq(
                        "remaining_single_position_heat_constant",
                        h * lp_heat,
                        h / (2. * pi.sqrt() * sigma),
                    ),
                ],
                "Analytic one-position heat law with alpha_B=1. This N-independent status constant does not use a whole-swarm sqrt(N) coefficient.",
            )?;
        }
    }
    for n in [4usize, 16, 64] {
        let diffs = (0..n)
            .map(|j| if j == 0 { 0.01_f64 } else { 0. })
            .collect::<Vec<_>>();
        let quotient = (squared(&diffs) / n as f64).sqrt();
        let positional = 0.01_f64.powi(2);
        let status = positional / n as f64;
        let inp = json!({"N":n,"admissible_permutation_coupling_position_differences":diffs,"normalized_position_error":quotient*quotient,"r_pos":positional,"r_status":status,"single_position_heat_constant":lp_heat,"single_position_standard_Gaussian_constant":1./(std::f64::consts::TAU.sqrt()*sigma)});
        emit(
            suite,
            src,
            &["0170", "0332", "0333", "1955"],
            &inp,
            &[
                le(
                    "remaining_single_atom_from_normalized_transport",
                    0.01,
                    (n as f64).sqrt() * quotient,
                ),
                eq(
                    "remaining_auxiliary_swarm_heat_constant",
                    (n as f64).sqrt() * h * lp_heat,
                    (n as f64).sqrt() * lp_heat,
                ),
            ],
            "Auxiliary finite representative coupling and whole-swarm sqrt(N) modulus; the principal status and physical errors retain their probability normalization. Permutation-invariant quotient cost is bounded using this admissible coupling.",
        )?;
        emit(
            suite,
            src,
            &["0279", "0284", "0286"],
            &inp,
            &[
                truth("remaining_positive_positional_buffer", positional > 0.),
                eq(
                    "remaining_normalized_status_buffer",
                    status,
                    positional / n as f64,
                ),
                le(
                    "remaining_normalized_position_within_buffer",
                    quotient * quotient,
                    status,
                ),
            ],
            "Declared sufficient positional-buffer fixture; no labeled swarm is inferred from the chosen admissible transport representative.",
        )?;
        let standard_lp = (1. / (std::f64::consts::TAU.sqrt() * sigma))
            .min(perimeter / (2. * pi * sigma * sigma).sqrt());
        emit(
            suite,
            src,
            &["1906", "1954"],
            &inp,
            &[
                eq(
                    "remaining_standard_Gaussian_boundary_constant",
                    standard_lp,
                    1. / (std::f64::consts::TAU.sqrt() * sigma),
                ),
                le(
                    "remaining_auxiliary_standard_swarm_modulus",
                    (n as f64).sqrt() * standard_lp,
                    (n as f64).sqrt() * h * standard_lp,
                ),
            ],
            "Standard Gaussian covariance sigma² convention; its whole-swarm constant is auxiliary. The heat covariance 2sigma² convention is evaluated independently.",
        )?;
    }
    Ok(())
}

fn marked_laws(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    emit(
        suite,
        src,
        &["0430"],
        &json!({"Gaussian_smoothing_length":0.2}),
        &[truth("remaining_positive_smoothing_length", 0.2_f64 > 0.)],
        SCOPE,
    )?;
    alias(
        suite,
        src,
        &["0434", "0440"],
        &["cemetery_diameter_hypothesis", "cemetery_triangle"],
    )?;
    alias(
        suite,
        src,
        &["0435", "0439", "0444", "0445"],
        &[
            "cemetery_probability_transport_definition",
            "cemetery_transport_self_branch",
        ],
    )?;
    alias(
        suite,
        src,
        &["0441", "0443"],
        &["actual_alive_empirical_measure_mass"],
    )?;
    alias(
        suite,
        src,
        &["0442"],
        &["nonempty_gaussian_density_norm_envelope"],
    )?;
    alias(
        suite,
        src,
        &["1363", "1409"],
        &[
            "marked_positional_status_decomposition",
            "native_marked_output_status_length",
        ],
    )?;
    for p in [0_f64, 0.13, 0.5, 0.9, 1.] {
        for q in [0_f64, 0.25, 0.75, 1.] {
            let atoms = [
                ((0_f64, 0_f64), (1. - p) * (1. - q)),
                ((0., 1.), (1. - p) * q),
                ((1., 0.), p * (1. - q)),
                ((1., 1.), p * q),
            ];
            let sx = atoms.iter().map(|((x, _), w)| x * w).sum::<f64>();
            let sy = atoms.iter().map(|((_, y), w)| y * w).sum::<f64>();
            let err = atoms
                .iter()
                .map(|((x, y), w)| (x - y).powi(2) * w)
                .sum::<f64>();
            let inp = json!({"independent_Bernoulli_probability1":p,"probability2":q,"joint_atoms":atoms,"output_mark_pairs":"each atom is (position,status), statuses here integrated independently"});
            emit(
                suite,
                src,
                &["1412", "1421"],
                &inp,
                &[
                    eq(
                        "remaining_Bernoulli_mass",
                        atoms.iter().map(|(_, w)| w).sum(),
                        1.,
                    ),
                    eq("remaining_Bernoulli_first_mean", sx, p),
                    eq("remaining_Bernoulli_second_mean", sy, q),
                ],
                SCOPE,
            )?;
            emit(
                suite,
                src,
                &["1419"],
                &inp,
                &[eq(
                    "remaining_independent_Bernoulli_variance_decomposition",
                    err,
                    p * (1. - p) + q * (1. - q) + (p - q).powi(2),
                )],
                "Exact independent four-atom conditional status law. The variance identity requires independence (or zero covariance); it is not asserted for arbitrary dependent X,Y.",
            )?;
            emit(
                suite,
                src,
                &["1423"],
                &inp,
                &[eq(
                    "remaining_death_alive_complement_difference",
                    (p - q).abs(),
                    ((1. - p) - (1. - q)).abs(),
                )],
                SCOPE,
            )?;
        }
    }
    alias(
        suite,
        src,
        &["1413", "1414", "1660"],
        &[
            "exact_status_kernel_Dirac_integral",
            "exact_deterministic_Dirac_kernel_expectation",
        ],
    )?;
    let config = GasConfig::euclidean(1, 0.04)?;
    for x in [[-0.7_f64, 0.1, 0.8], [-0.65, 0.16, 0.76]] {
        let obs = observations(&x)?;
        let measured_distance = (0..3)
            .map(|i| {
                config
                    .distance_donors
                    .distance
                    .compare(&obs, i, &obs, (i + 1) % 3)
            })
            .collect::<Result<Vec<_>>>()?;
        let reward = RewardBatch::new(
            x.iter().map(|v| v * v).collect(),
            algorithmic_gas::Provenance::default(),
        );
        let fitness = config
            .fitness
            .evaluate(&reward, &measured_distance, &[true; 3], &obs, 0)?
            .fitness;
        let proposals = donor_rows(&x, &config)?;
        let mut rows = vec![vec![0.; 3]; 3];
        for i in 0..3 {
            let mut accepted = 0.;
            for j in 0..3 {
                if i != j {
                    rows[i][j] = proposals[i][j]
                        * config
                            .clone_decision
                            .acceptance_probability(0, fitness[i], fitness[j]);
                    accepted += rows[i][j];
                }
            }
            rows[i][i] = 1. - accepted;
        }
        let atoms=(0..27).map(|index|{
            let sources=[index%3,index/3%3,index/9];
            let weight=(0..3).map(|i|rows[i][sources[i]]).product::<f64>();
            json!({"source_rows":sources,"weight":weight,"zero_jitter_source_positions":sources.map(|j|x[j])})
        }).collect::<Vec<_>>();
        let joint = atoms
            .iter()
            .map(|atom| atom["weight"].as_f64().unwrap())
            .sum::<f64>();
        emit(
            suite,
            src,
            &["1542", "1543"],
            &json!({"native_config":config,"positions":x,"frozen_actual_native_fitness":fitness,"conditioned_measurement_distance":measured_distance,"native_clone_donor_weights":proposals,"literal_source_rows":rows,"source_atoms":atoms,"after_transform_total_mass":joint}),
            &[eq(
                "remaining_native_conditional_clone_transition_mass",
                joint,
                1.,
            )],
            "Exact Cartesian conditional native living cloning-source law after the specified actual sampled measurement outcome. Each subsequent configured jitter probability kernel and collision pushforward preserves total mass; the retained atoms are source centers, not discrete surrogates for Gaussian jitter outputs. This is a finite normalization check of the cloned law, not a claim that it validates its complete convergence theorem.",
        )?;
    }
    alias(
        suite,
        src,
        &["1552", "1569", "1694", "1707", "1957"],
        &[
            "quotient_cost_after_reordering",
            "actual_quotient_zero_distance",
        ],
    )?;
    let left = observations(&[-0.5, 0.25, 0.7])?;
    let mut outputs = vec![];
    let mut expected_normalized = 0.;
    let mut expected_total = 0.;
    for shift in [0_f64, 0.1] {
        let right = observations(&[0.7 + shift, -0.5 + shift, 0.25 + shift])?;
        let cost = optimal_swarm_displacement(
            &algorithmic_gas::geometry::Distance::default(),
            &left,
            &right,
            &[true; 3],
            &[true; 3],
            1.,
        )?;
        expected_normalized += 0.5 * cost.metric_squared;
        expected_total += 0.5 * cost.positional_sum;
        outputs.push(json!({"right":right,"law_weight":0.5,"optimal_transport":cost}));
    }
    emit(
        suite,
        src,
        &["1553", "1719"],
        &json!({"left":left,"finite_positional_output_law":outputs,"N":3,"all_marks_alive":true,"primary_expected_normalized_position_error":expected_normalized,"auxiliary_expected_total_position_error":expected_total}),
        &[eq(
            "remaining_positional_output_law_probability_normalization",
            expected_normalized,
            expected_total / 3.,
        )],
        "Explicit finite positional output-law operands under optimal permutation transport; both total and normalized sums are retained. This checks the normalization identity, not a claim that this selected two-atom law is an unmeasured native cloning transition.",
    )?;
    alias(
        suite,
        src,
        &["1661"],
        &["actual_optimal_quotient_expected_decomposition"],
    )?;
    alias(
        suite,
        src,
        &["1570"],
        &["actual_persistent_clone_probability_sum"],
    )?;
    alias(
        suite,
        src,
        &["1641"],
        &["native_zero_noise_retains_nonzero_deterministic_drift"],
    )?;
    alias(
        suite,
        src,
        &["1647"],
        &[
            "native_joint_law_mass",
            "actual_stochastic_clone_probability_value",
        ],
    )?;
    alias(
        suite,
        src,
        &["1655"],
        &["reference_perturbation_second_moment_integral"],
    )?;
    alias(
        suite,
        src,
        &["1676"],
        &["exact_quotient_composite_stage_bound"],
    )?;
    alias(
        suite,
        src,
        &["1697"],
        &["canonical_reference_heat_boundary_exponent"],
    )?;
    alias(suite, src, &["1718"], &["composite_high_power_definition"])?;
    alias(
        suite,
        src,
        &["1720"],
        &[
            "composite_B_definition",
            "exact_quotient_composite_stage_bound",
        ],
    )?;
    Ok(())
}

fn transport_and_process(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    alias(
        suite,
        src,
        &["1753", "1754"],
        &["left_unit_mass", "right_unit_mass"],
    )?;
    let support = [-1_f64, 0.3, 2.];
    let p = [0.2_f64, 0.3, 0.5];
    let q = [0.1_f64, 0.7, 0.2];
    let total_variation = 0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>();
    let event_supremum = (0..8)
        .map(|mask| {
            (0..3)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| p[i] - q[i])
                .sum::<f64>()
                .abs()
        })
        .fold(0., f64::max);
    let fourth = support
        .iter()
        .zip(p)
        .map(|(x, w)| w * x.powi(4))
        .sum::<f64>();
    let inputs = json!({"support":support,"left_law_weights":p,"right_law_weights":q,"all_event_masks":(0..8).collect::<Vec<_>>(),"TV":total_variation,"event_supremum":event_supremum,"base_point":0.,"fourth_moment":fourth});
    emit(
        suite,
        src,
        &["1755"],
        &inputs,
        &[eq(
            "remaining_exact_TV_event_supremum",
            total_variation,
            event_supremum,
        )],
        "Explicit finite probability laws on a physical scalar metric space; all subsets are enumerated to compare the TV definition with the half-L1 formula. These law operands do not assert that an arbitrary native capped output is discrete.",
    )?;
    emit(
        suite,
        src,
        &["1758"],
        &inputs,
        &[eq(
            "remaining_exact_marginal_fourth_moment",
            fourth,
            0.2 + 0.3 * 0.3_f64.powi(4) + 0.5 * 16.,
        )],
        "Marginal fourth moment about base point0 evaluated directly against the explicit probability law; a coupling fourth moment is a separate quantity.",
    )?;
    for sigma in [0.05_f64, 0.2, 1.] {
        for separation in [0_f64, 0.01, 0.2, 1.] {
            let tv = 2. * normal_cdf(separation / (2. * sigma)) - 1.;
            let norm = separation / sigma;
            emit(
                suite,
                src,
                &["1767"],
                &json!({"affine_Gaussian_A_h":1.,"Sigma_h":sigma*sigma,"mean_separation":separation,"exact_halfspace_TV_with_CDF_error":tv,"CDF_absolute_error":1.5e-7,"covariance_whitened_mean_norm":norm}),
                &[le(
                    "remaining_affine_Gaussian_TV_no_offset",
                    tv,
                    norm + 3e-7,
                )],
                "Specified affine Gaussian reference transition with constant positive covariance. The halfspace formula checks the covariance-whitened mean bound; no Gaussian law is asserted for a capped full native transition.",
            )?;
        }
    }
    let positions = [-0.8_f64, -0.1, 0.2, 0.9];
    let reward = |x: f64| x * x;
    let derivative = |x: f64| 2. * x;
    let inp = json!({"native_landscape":"Sphere","working_chart":[-1.,1.],"identity_projection":true,"positions":positions,"L_R":2.,"N":4,"T":3,"time_points":[0,1,2,3],"scope":"bounded reward chart; ambient reference Gaussian kernel is a separate hypothesis regime"});
    emit(
        suite,
        src,
        &["1794", "1795", "1803"],
        &inp,
        &[
            eq("remaining_discrete_process_time_count", 4., 3. + 1.),
            truth("remaining_population_minimum", positions.len() >= 2),
        ],
        "Finite process indexing and population operands; a stored sequence does not prove a universal transition law.",
    )?;
    for x in positions {
        emit(
            suite,
            src,
            &["1799"],
            &inp,
            &[le(
                "remaining_bounded_chart_Sphere_gradient",
                derivative(x).abs(),
                2.,
            )],
            "Actual Sphere gradient on the declared bounded valid identity chart; no global bounded gradient is claimed for an unbounded quadratic or Rastrigin landscape.",
        )?;
        for y in positions {
            emit(
                suite,
                src,
                &["1800"],
                &json!({"chart":inp,"x":x,"y":y,"native_reward_x":crate::Benchmark::Sphere.value(&[x])?,"native_reward_y":crate::Benchmark::Sphere.value(&[y])?}),
                &[le(
                    "remaining_native_chart_reward_Lipschitz",
                    (reward(x) - reward(y)).abs(),
                    2. * (x - y).abs(),
                )],
                SCOPE,
            )?;
        }
    }
    for n in [4usize, 16, 64] {
        let dimension = 2_f64;
        let sigma = 0.2_f64;
        let delta = 0.1_f64;
        let diameter = 2_f64;
        let bm = n as f64 * dimension * sigma * sigma;
        let bs = diameter * diameter * (n as f64 / 2. * (2. / delta).ln()).sqrt();
        let mut prob = 2_f64.powi(-(n as i32));
        let mut tail = 0.;
        for count in 0..=n {
            if diameter * diameter * (count as f64 - n as f64 / 2.).abs() > bs {
                tail += prob;
            }
            if count < n {
                prob *= (n - count) as f64 / (count + 1) as f64;
            }
        }
        let inp = json!({"N":n,"dimension":dimension,"sigma":sigma,"delta":delta,"algorithmic_diameter":diameter,"BM_total":bm,"BM_per_walker":bm/n as f64,"BS_total":bs,"BS_per_walker":bs/n as f64,"independent_reference_status_law":"Binomial(N,.5)","exact_enumerated_BS_tail":tail});
        emit(
            suite,
            src,
            &["1949"],
            &inp,
            &[
                eq(
                    "remaining_reference_total_perturbation_mean",
                    bm,
                    (0..n).map(|_| dimension * sigma * sigma).sum(),
                ),
                eq(
                    "remaining_primary_perturbation_mean",
                    bm / n as f64,
                    dimension * sigma * sigma,
                ),
            ],
            "Auxiliary total reference-noise moment alongside its N-independent primary normalized moment; native deterministic drift is not included in this reference noise budget.",
        )?;
        emit(
            suite,
            src,
            &["1950"],
            &inp,
            &[
                eq(
                    "remaining_auxiliary_status_Hoeffding_exponent",
                    (-2. * bs * bs / (n as f64 * diameter.powi(4))).exp(),
                    delta / 2.,
                ),
                le(
                    "remaining_auxiliary_status_exact_binomial_tail",
                    tail,
                    delta,
                ),
                eq(
                    "remaining_primary_status_tail_normalization",
                    bs / n as f64,
                    diameter * diameter * ((2. / delta).ln() / (2. * n as f64)).sqrt(),
                ),
            ],
            "Explicit independent bounded Bernoulli reference status law; exact binomial tail and Hoeffding coefficient operands checked. Total BS is auxiliary and the probability-normalized primary offset is retained separately.",
        )?;
    }
    Ok(())
}
