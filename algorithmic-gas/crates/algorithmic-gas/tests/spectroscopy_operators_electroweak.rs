use algorithmic_gas::{
    GasConfig, GasError, RecordingConfig,
    boundary::{BoundaryPolicy, BoxDomain},
    geometry::Distance,
    physics::spectroscopy::{
        config::{
            ChannelSpec, ChiralityObservable, ElectroweakScales, FrameNormalization,
            MeasurementConfig, PairDistance, PairSelection, Range, Su2Mode, U1Mode,
        },
        contract::{
            Capabilities, Companions, Element, ElementKind, ExchangeParity, Frame, FrameState,
            LocalOperator, NO_COMPANION, OperatorContext, Record, WalkerRole,
        },
        fields::score::{self, AmplitudeDistance},
    },
    random::{RandomStream, Stream},
};

/// Six walkers in three dimensions; the five-walker frames use the first five.
const X: [f64; 18] = [
    0.1, -0.4, 0.25, 0.85, 0.3, -0.55, -0.6, 0.7, 0.15, 0.35, -0.9, 0.8, -0.2, 0.05, -0.75, 0.55,
    0.45, 0.3,
];
const V: [f64; 18] = [
    0.3, 0.1, -0.2, -0.45, 0.25, 0.05, 0.15, -0.35, 0.4, 0.05, 0.5, -0.1, -0.25, -0.15, 0.35, 0.2,
    -0.3, -0.05,
];
const F: [f64; 6] = [1.2, 0.45, 2.1, 0.8, 1.55, 0.95];

/// `ħ_eff = 0.7`, `h_S = 0.35`, `ε_d = 1.3`, `ε_c = 0.9`, `ε_clone = 0.05`,
/// `λ = 0.25` on raw coordinates: the parameters of the reference vectors.
fn scales() -> ElectroweakScales {
    ElectroweakScales {
        h_eff: 0.7,
        h_s: Some(0.35),
        epsilon_d: Range::Fixed { value: 1.3 },
        epsilon_c: Range::Fixed { value: 0.9 },
        distance: PairDistance::Raw,
        lambda: Some(0.25),
    }
}
/// Gas, measurement and capabilities of an operator context.
fn setup(scales: ElectroweakScales) -> (GasConfig, MeasurementConfig, Capabilities) {
    let mut gas = GasConfig::default();
    gas.clone_decision.epsilon = 0.05;
    let measurement = MeasurementConfig {
        electroweak: scales,
        ..MeasurementConfig::default()
    };
    (gas, measurement, Capabilities::nominal())
}
fn context_of(setup: &(GasConfig, MeasurementConfig, Capabilities)) -> OperatorContext<'_> {
    OperatorContext {
        gas: &setup.0,
        measurement: &setup.1,
        capabilities: &setup.2,
    }
}
fn companions(map: &[Option<u32>]) -> Companions {
    Companions {
        count: 1,
        slot: map.iter().map(|j| j.unwrap_or(0)).collect(),
        generation: vec![0; map.len()],
        valid: map.iter().map(Option::is_some).collect(),
        historical: vec![false; map.len()],
        mutual: false,
    }
}
fn frame(
    eligible: &[bool],
    distance: &[Option<u32>],
    cloning: &[Option<u32>],
    cloned: &[bool],
) -> Frame {
    let n = eligible.len();
    let frame = Frame {
        step: 1,
        n,
        d: 3,
        x: X[..3 * n].to_vec(),
        v: Some(V[..3 * n].to_vec()),
        eligible: eligible.to_vec(),
        generation: vec![0; n],
        fitness: Some(F[..n].to_vec()),
        distance: Some(companions(distance)),
        cloning: Some(companions(cloning)),
        companion_fitness: Some(
            cloning
                .iter()
                .map(|k| k.map_or(0., |k| F[k as usize]))
                .collect(),
        ),
        cloned: cloned.to_vec(),
        revived: cloned.iter().zip(eligible).map(|(c, e)| *c && !e).collect(),
        ..Frame::default()
    };
    frame.validate().unwrap();
    frame
}
/// The state an accumulator hands to the operators: the coordinates and
/// companion maps of the frame, then the score fields.
fn state(frame: &Frame, gas: &GasConfig) -> FrameState {
    let map = |companions: &Option<Companions>| {
        let companions = companions.as_ref().unwrap();
        (0..frame.n)
            .map(|i| {
                companions
                    .first(i, &frame.eligible)
                    .map_or(NO_COMPANION, |k| companions.slot[i * companions.count + k])
            })
            .collect::<Vec<u32>>()
    };
    let mut state = FrameState {
        step: frame.step,
        n: frame.n,
        d: frame.d,
        x: frame.x.clone(),
        v: frame.v.clone(),
        fitness: frame.fitness.clone(),
        cloned: frame.cloned.clone(),
        generation: frame.generation.clone(),
        eligible: frame.eligible.clone(),
        distance_companion: Some(map(&frame.distance)),
        cloning_companion: Some(map(&frame.cloning)),
        ..FrameState::default()
    };
    assert!(score::fill(frame, gas, &mut state).is_available());
    state
}
fn some(map: &[u32]) -> Vec<Option<u32>> {
    map.iter().map(|&j| Some(j)).collect()
}
fn element(kind: ElementKind, i: u32, j: u32, k: u32) -> Element {
    Element {
        walkers: [i, j, k],
        kind,
        weight: 1.,
        generation: [0; 3],
    }
}
/// The elements a topology samples from the two companion maps, unmasked.
fn elements(kind: ElementKind, distance: &[Option<u32>], cloning: &[Option<u32>]) -> Vec<Element> {
    (0..distance.len() as u32)
        .map(|i| {
            let (j, k) = (
                distance[i as usize].unwrap_or(i),
                cloning[i as usize].unwrap_or(i),
            );
            match kind {
                ElementKind::Site => element(kind, i, i, i),
                ElementKind::DistancePair => element(kind, i, j, i),
                ElementKind::CloningPair => element(kind, i, k, i),
                ElementKind::Triplet => element(kind, i, j, k),
            }
        })
        .collect()
}
fn values(
    spec: &ChannelSpec,
    elements: &[Element],
    state: &FrameState,
    context: &OperatorContext<'_>,
) -> Vec<Option<[f64; 2]>> {
    let components = spec
        .signature(elements[0].kind, context)
        .map_or(2, |s| s.components);
    let mut values = vec![0.; elements.len() * components];
    let mut valid = vec![false; elements.len()];
    spec.evaluate_all(elements, state, context, &mut values, &mut valid);
    values
        .chunks_exact(components)
        .zip(valid)
        .map(|(v, ok)| ok.then(|| [v[0], *v.get(1).unwrap_or(&0.)]))
        .collect()
}
fn sum(values: &[Option<[f64; 2]>]) -> [f64; 2] {
    values
        .iter()
        .flatten()
        .fold([0.; 2], |s, v| [s[0] + v[0], s[1] + v[1]])
}
/// Frame mean over the valid elements.
fn mean(values: &[Option<[f64; 2]>]) -> [f64; 2] {
    let count = values.iter().flatten().count() as f64;
    sum(values).map(|s| s / count)
}
fn close(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12
}
fn assert_values(actual: &[Option<[f64; 2]>], expected: &[[f64; 2]]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(close(a.unwrap(), *e), "{a:?} differs from {e:?}");
    }
}
fn u1(mode: U1Mode, charge: u8) -> ChannelSpec {
    ChannelSpec::U1 { mode, charge }
}
fn su2(mode: Su2Mode, directed: bool) -> ChannelSpec {
    ChannelSpec::Su2 { mode, directed }
}
fn chirality(observable: ChiralityObservable) -> ChannelSpec {
    ChannelSpec::Chirality { observable }
}

const CYCLE_DISTANCE: [u32; 5] = [1, 2, 3, 4, 0];
const CYCLE_CLONING: [u32; 5] = [2, 3, 4, 0, 1];
fn cycle() -> (Frame, Vec<Option<u32>>, Vec<Option<u32>>) {
    let (distance, cloning) = (some(&CYCLE_DISTANCE), some(&CYCLE_CLONING));
    (
        frame(
            &[true; 5],
            &distance,
            &cloning,
            &[true, false, false, true, false],
        ),
        distance,
        cloning,
    )
}
const INVOLUTION: [u32; 6] = [3, 4, 5, 0, 1, 2];
const COMPONENT: [[f64; 2]; 5] = [
    [-0.2635562972582, 0.498494653764248],
    [-0.13715126855599, 0.299680989091724],
    [0.476947233237002, -0.42759952166946],
    [0.18224953391603, 0.791700189025871],
    [-0.258481419650715, -0.622637098276613],
];

#[test]
fn fitness_phases_on_distance_pairs_reproduce_the_reference_vectors() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, distance, cloning) = cycle();
    let state = state(&frame, &setup.0);
    let pairs = elements(ElementKind::DistancePair, &distance, &cloning);
    let phase = values(&u1(U1Mode::Phase, 1), &pairs, &state, &context);
    assert_values(
        &phase,
        &[
            [0.478870595954639, 0.877885500694737],
            [-0.707777059801935, -0.706435866599458],
            [-0.282449409554445, 0.95928219572884],
            [0.478870595954639, -0.877885500694737],
            [0.877582561890373, 0.479425538604203],
        ],
    );
    assert!(close(mean(&phase), [0.169019456888654, 0.146454373546717]));
    let charged = values(&u1(U1Mode::Phase, 2), &pairs, &state, &context);
    assert_values(
        &charged,
        &[
            [-0.541365904660098, 0.84078710579525],
            [0.00189673276374416, 0.999998201200794],
            [-0.84044466208469, -0.541897379559406],
            [-0.541365904660098, -0.84078710579525],
            [0.54030230586814, 0.841470984807897],
        ],
    );
    assert!(close(
        mean(&charged),
        [-0.276195486554601, 0.259914361289857]
    ));
    let dressed = values(&u1(U1Mode::Dressed, 1), &pairs, &state, &context);
    assert_values(
        &dressed,
        &[
            [0.363985698826604, 0.667273727306458],
            [-0.456595124512184, -0.455729905346348],
            [-0.153309729687196, 0.520685436563389],
            [0.273508924620501, -0.501407940398354],
            [0.70718136656884, 0.386334941327692],
        ],
    );
    assert!(close(
        mean(&dressed),
        [0.146954227163313, 0.123431251890567]
    ));
    let dressed_charged = values(&u1(U1Mode::Dressed, 2), &pairs, &state, &context);
    assert_values(
        &dressed_charged,
        &[
            [-0.411487881680812, 0.639075534920233],
            [0.00122360412849557, 0.645110344939704],
            [-0.456182026241458, -0.294134588241856],
            [-0.309203379076994, -0.480219038469896],
            [0.435391198066966, 0.678081615116245],
        ],
    );
    assert!(close(
        mean(&dressed_charged),
        [-0.148051696960761, 0.237582773652886]
    ));
    // The amplitude is the modulus of the dressed value and is not raised to the charge.
    let amplitudes = [
        0.760091978712934,
        0.645111505365768,
        0.542786511499658,
        0.571154142540858,
        0.805828872722269,
    ];
    for ((one, two), a) in dressed.iter().zip(&dressed_charged).zip(amplitudes) {
        let (one, two) = (one.unwrap(), two.unwrap());
        assert!((one[0].hypot(one[1]) - a).abs() < 1e-12);
        assert!((two[0].hypot(two[1]) - a).abs() < 1e-12);
    }
}

#[test]
fn cloning_phases_components_and_two_hop_doublets_reproduce_the_reference_vectors() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, distance, cloning) = cycle();
    let state = state(&frame, &setup.0);
    let pairs = elements(ElementKind::CloningPair, &distance, &cloning);
    let run = |mode, directed| values(&su2(mode, directed), &pairs, &state, &context);
    let phase = run(Su2Mode::Phase, false);
    assert_values(
        &phase,
        &[
            [-0.46739917810437, 0.884046383572355],
            [-0.416146836547143, 0.909297426825681],
            [0.744575913922293, -0.66753779548935],
            [0.224332974882709, 0.974512553218415],
            [-0.383413460126, -0.92357680708981],
        ],
    );
    assert!(close(
        mean(&phase),
        [-0.0596101171945022, 0.235348352207458]
    ));
    let component = run(Su2Mode::Component, false);
    assert_values(&component, &COMPONENT);
    assert!(close(
        mean(&component),
        [1.55633762546348e-06, 0.107927842387154]
    ));
    let doublet = run(Su2Mode::Doublet, false);
    assert_values(
        &doublet,
        &[
            [0.213390935978802, 0.0708951320947883],
            [0.0450982653600402, 1.0913811781176],
            [0.218465813586287, -1.05023661994607],
            [-0.0813067633421692, 1.29019484279012],
            [-0.395632688206705, -0.322956109184888],
        ],
    );
    let difference = run(Su2Mode::DoubletDiff, false);
    assert_values(
        &difference,
        &[
            [-0.740503530495202, 0.926094175433707],
            [-0.319400802472021, -0.492019199934147],
            [0.735428652887717, 0.195037576607153],
            [0.44580583117423, 0.293205535261624],
            [-0.121330151094725, -0.922318087368337],
        ],
    );
    // A five-cycle is a bijection: the difference has no frame mean and the
    // sum repeats the component.
    assert!(close(mean(&difference), [0., 0.]));
    let twice = mean(&component).map(|m| 2. * m);
    assert!(close(mean(&doublet), twice));
    for (i, (plus, minus)) in doublet.iter().zip(&difference).enumerate() {
        let (plus, minus) = (plus.unwrap(), minus.unwrap());
        let (own, next) = (COMPONENT[i], COMPONENT[CYCLE_CLONING[i] as usize]);
        let norm = |z: [f64; 2]| z[0] * z[0] + z[1] * z[1];
        assert!((norm(plus) + norm(minus) - 2. * (norm(own) + norm(next))).abs() < 1e-14);
        assert!(close(
            [0.5 * (plus[0] + minus[0]), 0.5 * (plus[1] + minus[1])],
            own
        ));
        assert!(close(
            [0.5 * (plus[0] - minus[0]), 0.5 * (plus[1] - minus[1])],
            next
        ));
    }
    assert!(close(
        mean(&run(Su2Mode::Phase, true)),
        [-0.0596101171945022, 0.871794193239122]
    ));
    assert!(close(
        mean(&run(Su2Mode::Component, true)),
        [1.55633762546348e-06, 0.528022490365583]
    ));
    assert!(close(
        mean(&run(Su2Mode::Doublet, true)),
        [3.11267525097136e-06, 1.05604498073117]
    ));
    assert!(close(mean(&run(Su2Mode::DoubletDiff, true)), [0., 0.]));
    for z in run(Su2Mode::Phase, true) {
        assert!(z.unwrap()[1] >= 0.);
    }
}

#[test]
fn mixed_triplet_product_reproduces_the_reference_vectors_and_keeps_coinciding_companions() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame_a, distance, cloning) = cycle();
    let state_a = state(&frame_a, &setup.0);
    let triplets = elements(ElementKind::Triplet, &distance, &cloning);
    let mixed = values(
        &ChannelSpec::ElectroweakMixed,
        &triplets,
        &state_a,
        &context,
    );
    assert_values(
        &mixed,
        &[
            [-0.42856310869729, 0.00558073208513777],
            [0.199196189336198, -0.0743289438911174],
            [0.149524192212138, 0.313894645437228],
            [0.446811635226411, 0.125155703882917],
            [0.057753223249893, -0.54017775813077],
        ],
    );
    assert!(close(
        mean(&mixed),
        [0.0849444262654701, -0.0339751241233209]
    ));
    let dressed = values(
        &u1(U1Mode::Dressed, 1),
        &elements(ElementKind::DistancePair, &distance, &cloning),
        &state_a,
        &context,
    );
    let component = values(
        &su2(Su2Mode::Component, false),
        &elements(ElementKind::CloningPair, &distance, &cloning),
        &state_a,
        &context,
    );
    for ((m, u), a) in mixed.iter().zip(&dressed).zip(&component) {
        let (m, u, a) = (m.unwrap(), u.unwrap(), a.unwrap());
        assert!((m[0].hypot(m[1]) - u[0].hypot(u[1]) * a[0].hypot(a[1])).abs() < 1e-12);
    }
    // One involution in both roles: the two companions coincide on every
    // triplet and the product stays defined.
    let involution = some(&INVOLUTION);
    let frame_b = frame(&[true; 6], &involution, &involution, &[false; 6]);
    let state_b = state(&frame_b, &setup.0);
    let coinciding = elements(ElementKind::Triplet, &involution, &involution);
    let mixed = values(
        &ChannelSpec::ElectroweakMixed,
        &coinciding,
        &state_b,
        &context,
    );
    assert!(close(mean(&mixed), [0.368139497970473, 0.0109408744953286]));
    assert!(
        ChannelSpec::ElectroweakMixed
            .signature(ElementKind::Triplet, &context)
            .unwrap()
            .degenerate
    );
}

#[test]
fn mutual_pairings_cancel_the_imaginary_fitness_phase_and_mirror_the_doublets() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let involution = some(&INVOLUTION);
    let frame = frame(&[true; 6], &involution, &involution, &[false; 6]);
    let state = state(&frame, &setup.0);
    let distance = elements(ElementKind::DistancePair, &involution, &involution);
    let cloning = elements(ElementKind::CloningPair, &involution, &involution);
    for (mode, re) in [
        (U1Mode::Phase, 0.256166167446444),
        (U1Mode::Dressed, 0.234289041151225),
    ] {
        let z = values(&u1(mode, 1), &distance, &state, &context);
        assert!(sum(&z)[1].abs() < 6e-14);
        assert!((mean(&z)[0] - re).abs() < 1e-12);
        for i in 0..6 {
            let (a, b) = (z[i].unwrap(), z[INVOLUTION[i] as usize].unwrap());
            assert!((a[0] - b[0]).abs() < 1e-15 && (a[1] + b[1]).abs() < 1e-15);
        }
    }
    // The cloning phase is only weight antisymmetric: its imaginary mean survives.
    let phase = values(&su2(Su2Mode::Phase, false), &cloning, &state, &context);
    assert!(close(
        mean(&phase),
        [0.0840307817622067, -0.313563317366821]
    ));
    let component = values(&su2(Su2Mode::Component, false), &cloning, &state, &context);
    assert!(close(
        mean(&component),
        [0.081691346019312, -0.200180121695648]
    ));
    for directed in [false, true] {
        let plus = values(&su2(Su2Mode::Doublet, directed), &cloning, &state, &context);
        let minus = values(
            &su2(Su2Mode::DoubletDiff, directed),
            &cloning,
            &state,
            &context,
        );
        for i in 0..6 {
            let j = INVOLUTION[i] as usize;
            let (p, q) = (plus[i].unwrap(), plus[j].unwrap());
            let (m, n) = (minus[i].unwrap(), minus[j].unwrap());
            assert_eq!(p.map(f64::to_bits), q.map(f64::to_bits));
            assert_eq!(m.map(f64::to_bits), n.map(|x| (-x).to_bits()));
        }
        let residual = sum(&minus);
        assert!(residual[0].abs() < 6e-14 && residual[1].abs() < 6e-14);
        if !directed {
            assert_values(
                &plus[..3],
                &[
                    [0.678107061144413, 0.14817132828444],
                    [0.415674863170479, -0.620932167873564],
                    [-0.60363384819902, -0.728319890584765],
                ],
            );
            assert_values(
                &minus[..3],
                &[
                    [0.313607993312352, -1.4352290497673],
                    [0.932637702471909, 0.624342028679662],
                    [0.657865688692342, -0.545241180965989],
                ],
            );
            assert!(close(mean(&plus), [0.163382692038624, -0.400360243391296]));
            // The second hop of a mutual pair is the reversed pair.
            for i in 0..6 {
                let j = INVOLUTION[i];
                let reversed = values(
                    &su2(Su2Mode::Component, false),
                    &[element(ElementKind::CloningPair, j, i as u32, j)],
                    &state,
                    &context,
                )[0]
                .unwrap();
                let own = component[i].unwrap();
                assert!(close(
                    plus[i].unwrap(),
                    [own[0] + reversed[0], own[1] + reversed[1]]
                ));
            }
        }
    }
}

#[test]
fn dead_self_paired_and_broken_second_hop_companions_are_masked_never_zero_filled() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let eligible = [true, true, true, true, true, false];
    let distance = [Some(1), Some(5), Some(0), Some(4), Some(3), None];
    let cloning = some(&[2, 0, 1, 4, 4, 3]);
    let frame_c = frame(
        &eligible,
        &distance,
        &cloning,
        &[true, false, false, false, false, true],
    );
    let state_c = state(&frame_c, &setup.0);
    let valid = |spec: &ChannelSpec, kind| {
        let z = values(
            spec,
            &elements(kind, &distance, &cloning),
            &state_c,
            &context,
        );
        (z.iter().map(Option::is_some).collect::<Vec<bool>>(), z)
    };
    let (mask, z) = valid(&u1(U1Mode::Phase, 1), ElementKind::DistancePair);
    assert_eq!(mask, [true, false, true, true, true, false]);
    assert!(close(mean(&z), [0.429461992472982, 0.459381020915355]));
    let fixed = sum(&z).map(|s| s / 6.);
    assert!(close(fixed, [0.286307994981988, 0.30625401394357]));
    let (mask, z) = valid(&su2(Su2Mode::Phase, false), ElementKind::CloningPair);
    assert_eq!(mask, [true, true, true, true, false, false]);
    assert!(close(mean(&z), [-0.569340028805316, -0.064391524512291]));
    let (mask, z) = valid(&su2(Su2Mode::Component, false), ElementKind::CloningPair);
    assert_eq!(mask, [true, true, true, true, false, false]);
    assert!(close(mean(&z), [-0.245834560067654, -0.0400176881220383]));
    // Walker 3 points at the self-paired walker 4, whose second hop is undefined.
    let (mask, z) = valid(&su2(Su2Mode::Doublet, false), ElementKind::CloningPair);
    assert_eq!(mask, [true, true, true, false, false, false]);
    assert!(close(mean(&z), [-0.486991895034421, -0.227204027763584]));
    // The map restricted to the valid walkers is the cycle 0 → 2 → 1 → 0.
    let (mask, z) = valid(&su2(Su2Mode::DoubletDiff, false), ElementKind::CloningPair);
    assert_eq!(mask, [true, true, true, false, false, false]);
    assert!(close(mean(&z), [0., 0.]));
    let (mask, z) = valid(&ChannelSpec::ElectroweakMixed, ElementKind::Triplet);
    assert_eq!(mask, [true, false, true, true, false, false]);
    assert_values(
        &[z[0], z[2], z[3]],
        &[
            [-0.42856310869729, 0.00558073208513777],
            [0.18760144292108, -0.239824895525355],
            [0.0214652687236389, 0.176213911727548],
        ],
    );
    assert!(close(mean(&z), [-0.0731654656841902, -0.0193434172375566]));

    // One involution for both roles, with an odd self-paired walker and a dead one.
    let mutual = [Some(3), Some(4), Some(2), Some(0), Some(1), None];
    let frame_d = frame(&eligible, &mutual, &mutual, &[false; 6]);
    let state_d = state(&frame_d, &setup.0);
    let run = |spec: &ChannelSpec, kind| {
        values(spec, &elements(kind, &mutual, &mutual), &state_d, &context)
    };
    let z = run(&u1(U1Mode::Phase, 1), ElementKind::DistancePair);
    assert_eq!(
        z.iter().map(Option::is_some).collect::<Vec<_>>(),
        [true, true, false, true, true, false]
    );
    assert!((mean(&z)[0] - 0.420248484411841).abs() < 1e-12);
    assert!((sum(&z)[0] / 6. - 0.280165656274561).abs() < 1e-12);
    assert!(sum(&z)[1].abs() < 1e-15);
    let z = run(&u1(U1Mode::Dressed, 1), ElementKind::DistancePair);
    assert!((mean(&z)[0] - 0.380442783269245).abs() < 1e-12 && sum(&z)[1].abs() < 1e-15);
    let phase = run(&su2(Su2Mode::Phase, false), ElementKind::CloningPair);
    assert!(close(mean(&phase), [0.362818213212795, -0.184665524515826]));
    let component = run(&su2(Su2Mode::Component, false), ElementKind::CloningPair);
    assert!(close(
        mean(&component),
        [0.273445481078723, -0.118190209897281]
    ));
    let doublet = run(&su2(Su2Mode::Doublet, false), ElementKind::CloningPair);
    assert_eq!(
        doublet.iter().map(Option::is_some).collect::<Vec<_>>(),
        [true, true, false, true, true, false]
    );
    assert!(close(
        mean(&doublet),
        [0.546890962157446, -0.236380419794562]
    ));
    let difference = run(&su2(Su2Mode::DoubletDiff, false), ElementKind::CloningPair);
    assert_values(
        &difference[..2],
        &[
            [0.313607993312352, -1.4352290497673],
            [0.932637702471909, 0.624342028679662],
        ],
    );
    let residual = sum(&difference);
    assert!(residual[0].abs() < 6e-14 && residual[1].abs() < 6e-14);
    // A frame without a valid element has no value at all.
    let alone = frame(&[true], &[Some(0)], &[Some(0)], &[false]);
    let state_alone = state(&alone, &setup.0);
    let mut out = [7.; 2];
    for (spec, kind) in [
        (u1(U1Mode::Phase, 1), ElementKind::DistancePair),
        (su2(Su2Mode::Phase, false), ElementKind::CloningPair),
        (ChannelSpec::ElectroweakMixed, ElementKind::Triplet),
    ] {
        assert!(!spec.evaluate(&element(kind, 0, 0, 0), &state_alone, &context, &mut out));
    }
}

#[test]
fn fitness_phase_is_exactly_antisymmetric_shift_invariant_and_of_unit_modulus() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, distance, cloning) = cycle();
    let state_a = state(&frame, &setup.0);
    for (i, j) in [(0usize, 1usize), (2, 4), (3, 1)] {
        let theta = score::u1_phase(F[i], F[j], 0.7);
        assert_eq!(
            theta.to_bits(),
            (-score::u1_phase(F[j], F[i], 0.7)).to_bits()
        );
        let pair = |a: usize, b: usize| {
            values(
                &u1(U1Mode::Dressed, 2),
                &[element(
                    ElementKind::DistancePair,
                    a as u32,
                    b as u32,
                    a as u32,
                )],
                &state_a,
                &context,
            )[0]
            .unwrap()
        };
        let (forward, backward) = (pair(i, j), pair(j, i));
        assert!((forward[0] - backward[0]).abs() < 1e-15);
        assert!((forward[1] + backward[1]).abs() < 1e-15);
    }
    assert!((score::u1_phase(F[0], F[1], 0.7) - 1.07142857142857).abs() < 1e-14);
    let pairs = elements(ElementKind::DistancePair, &distance, &cloning);
    let mut shifted = state_a.clone();
    for f in shifted.fitness.as_mut().unwrap() {
        *f += 3.5;
    }
    for charge in [1, 2] {
        let phase = values(&u1(U1Mode::Phase, charge), &pairs, &state_a, &context);
        let moved = values(&u1(U1Mode::Phase, charge), &pairs, &shifted, &context);
        for (a, b) in phase.iter().zip(&moved) {
            assert!(close(a.unwrap(), b.unwrap()));
            assert!((a.unwrap()[0].hypot(a.unwrap()[1]) - 1.).abs() < 1e-15);
        }
    }
    // The score phase is not shift invariant: it rescales by (F_i + ε)/(F_i + c + ε).
    let moved = score::su2_phase(F[0] + 3.5, F[2] + 3.5, 0.05, 0.35);
    assert!((score::su2_phase(F[0], F[2], 0.05, 0.35) - 2.05714285714286).abs() < 1e-14);
    assert!((moved - 0.541353383458647).abs() < 1e-14);
    assert!((score::su2_phase(-0.3, 0.5, 0.05, 0.35) - 6.53061224489796).abs() < 1e-14);
    // An unbounded range removes the dressing; the charge-2 phase is the square.
    let wide = self::setup(ElectroweakScales {
        epsilon_d: Range::Fixed { value: 1e9 },
        ..scales()
    });
    let dressed = values(
        &u1(U1Mode::Dressed, 1),
        &pairs,
        &state_a,
        &context_of(&wide),
    );
    let phase = values(&u1(U1Mode::Phase, 1), &pairs, &state_a, &context);
    let charged = values(&u1(U1Mode::Phase, 2), &pairs, &state_a, &context);
    for ((d, p), q) in dressed.iter().zip(&phase).zip(&charged) {
        let (d, p, q) = (d.unwrap(), p.unwrap(), q.unwrap());
        assert!(close(d, p));
        assert!(close(q, [p[0] * p[0] - p[1] * p[1], 2. * p[0] * p[1]]));
    }
    assert_eq!(score::amplitude(0., 1.3), 1.);
    assert!(score::amplitude(1., 1.3) > score::amplitude(2., 1.3));
}

#[test]
fn score_phase_scale_defaults_to_the_fitness_scale_and_the_regulariser_is_never_a_range() {
    let (frame, distance, cloning) = cycle();
    let pairs = elements(ElementKind::CloningPair, &distance, &cloning);
    let explicit = setup(ElectroweakScales {
        h_s: Some(0.7),
        ..scales()
    });
    let implicit = setup(ElectroweakScales {
        h_s: None,
        ..scales()
    });
    let state_a = state(&frame, &explicit.0);
    for mode in [Su2Mode::Phase, Su2Mode::Doublet] {
        let a = values(&su2(mode, false), &pairs, &state_a, &context_of(&explicit));
        let b = values(&su2(mode, false), &pairs, &state_a, &context_of(&implicit));
        assert_eq!(a, b);
    }
    let halved: [f64; 5] = [
        1.02857142857143,
        1.,
        -0.365448504983389,
        0.672268907563025,
        -0.982142857142857,
    ];
    let phase = values(
        &su2(Su2Mode::Phase, false),
        &pairs,
        &state_a,
        &context_of(&implicit),
    );
    for (z, theta) in phase.iter().zip(halved) {
        assert!(close(z.unwrap(), [theta.cos(), theta.sin()]));
    }
    // A regulariser of 1e-8 used as a Gaussian range would give amplitude 0.
    let mut tiny = setup(scales());
    tiny.0.clone_decision.epsilon = 1e-8;
    let component = values(
        &su2(Su2Mode::Component, false),
        &pairs,
        &state_a,
        &context_of(&tiny),
    );
    let modulus = component[0].unwrap();
    assert!((modulus[0].hypot(modulus[1]) - 0.56387839261315).abs() < 1e-12);
}

#[test]
fn interaction_ranges_come_from_gaussian_kernels_and_a_uniform_kernel_is_unavailable() {
    let uniform = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let capabilities = Capabilities::of(&uniform, &RecordingConfig::default(), 3);
    let mut setup = (uniform, MeasurementConfig::default(), capabilities);
    assert_eq!(setup.2.distance_kernel_width, None);
    assert_eq!(setup.2.cloning_kernel_width, None);
    let ranged = [
        (u1(U1Mode::Dressed, 1), ElementKind::DistancePair),
        (su2(Su2Mode::Component, false), ElementKind::CloningPair),
        (su2(Su2Mode::Doublet, false), ElementKind::CloningPair),
        (su2(Su2Mode::DoubletDiff, true), ElementKind::CloningPair),
        (ChannelSpec::ElectroweakMixed, ElementKind::Triplet),
    ];
    let (frame, distance, cloning) = cycle();
    let state_a = state(&frame, &setup.0);
    for (spec, kind) in &ranged {
        let Err(GasError::Capability(reason)) = spec.signature(*kind, &context_of(&setup)) else {
            panic!("{} exists without a range", spec.id());
        };
        assert!(reason.contains("Gaussian companion kernel"), "{reason}");
        let z = values(
            spec,
            &elements(*kind, &distance, &cloning),
            &state_a,
            &context_of(&setup),
        );
        assert!(z.iter().all(Option::is_none));
    }
    for (spec, kind) in [
        (u1(U1Mode::Phase, 1), ElementKind::DistancePair),
        (su2(Su2Mode::Phase, false), ElementKind::CloningPair),
        (ChannelSpec::FitnessPhase, ElementKind::Site),
        (ChannelSpec::CloneIndicator, ElementKind::Site),
        (chirality(ChiralityObservable::Chi), ElementKind::Site),
    ] {
        assert!(spec.signature(kind, &context_of(&setup)).is_ok());
    }
    // The gated cloning period is stated where it turns a series into a comb.
    assert_eq!(setup.0.clone_decision.every, 20);
    for spec in [
        ChannelSpec::CloneIndicator,
        chirality(ChiralityObservable::LeftFraction),
    ] {
        let signature = spec
            .signature(ElementKind::Site, &context_of(&setup))
            .unwrap();
        assert!(signature.descriptor.note.contains("comb"));
    }
    for (role, comb) in [
        (WalkerRole::Cloner, true),
        (WalkerRole::StrongResister, true),
        (WalkerRole::WeakResister, false),
        (WalkerRole::Persister, false),
    ] {
        let signature = ChannelSpec::ParityVelocity { role }
            .signature(ElementKind::Site, &context_of(&setup))
            .unwrap();
        assert_eq!(signature.descriptor.note.contains("comb"), comb);
    }
    setup.1.electroweak.epsilon_d = Range::Fixed { value: 1.3 };
    setup.1.electroweak.epsilon_c = Range::Fixed { value: 0.9 };
    for (spec, kind) in &ranged {
        assert!(spec.signature(*kind, &context_of(&setup)).is_ok());
    }
    // The Euclidean Gas has Gaussian kernels of width 2 on a squashed
    // phase-space distance; the configured distance is commensurate with it.
    let gaussian = GasConfig::euclidean(3, 0.01).unwrap();
    let capabilities = Capabilities::of(&gaussian, &RecordingConfig::default(), 3);
    let mut setup = (gaussian, MeasurementConfig::default(), capabilities);
    setup.1.electroweak.distance = PairDistance::Configured;
    assert_eq!(setup.2.distance_kernel_width, Some(2.));
    let dressed = u1(U1Mode::Dressed, 1);
    let signature = dressed
        .signature(ElementKind::DistancePair, &context_of(&setup))
        .unwrap();
    assert!(signature.requires.records.contains(&Record::Velocities));
    let z = values(
        &dressed,
        &[element(ElementKind::DistancePair, 0, 1, 0)],
        &state_a,
        &context_of(&setup),
    )[0]
    .unwrap();
    assert!((z[0].hypot(z[1]) - 0.925467170155511).abs() < 1e-12);
    // The width is the one of the capabilities, whatever the gas configures:
    // raw D² = 2.34 with the donor's velocity weight 1, range 1.3.
    setup.1.electroweak.distance = PairDistance::Raw;
    setup.2.distance_kernel_width = Some(1.3);
    let z = values(
        &dressed,
        &[element(ElementKind::DistancePair, 0, 1, 0)],
        &state_a,
        &context_of(&setup),
    )[0]
    .unwrap();
    assert!((z[0].hypot(z[1]) - 0.7074036474040617).abs() < 1e-12);
    // A fixed range of zero is a configuration error, not a channel without a kernel.
    setup.1.electroweak.epsilon_d = Range::Fixed { value: 0. };
    assert!(matches!(
        dressed.signature(ElementKind::DistancePair, &context_of(&setup)),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn amplitude_distances_follow_the_raw_coordinates_or_the_donor_module() {
    let (frame, ..) = cycle();
    let mut gas = GasConfig::euclidean(3, 0.01).unwrap();
    let state_a = state(&frame, &gas);
    let configured = ElectroweakScales {
        distance: PairDistance::Configured,
        ..ElectroweakScales::default()
    };
    let squared = |gas: &GasConfig, scales: &ElectroweakScales, i, j| {
        AmplitudeDistance::new(gas, &gas.distance_donors.distance, scales)
            .unwrap()
            .squared(&state_a, i, j)
            .unwrap()
    };
    let squashed = squared(&gas, &configured, 0, 1);
    assert!((squashed - 1.23930592235474).abs() < 1e-13);
    assert_eq!(
        squashed.to_bits(),
        squared(&gas, &configured, 1, 0).to_bits()
    );
    assert!((score::amplitude(squashed, 2.) - 0.925467170155511).abs() < 1e-14);
    // Raw coordinates keep the velocity weight of the donor module, here 1.
    let raw = squared(&gas, &ElectroweakScales::default(), 0, 1);
    assert!((raw - 2.34).abs() < 1e-14);
    let weighted = ElectroweakScales {
        lambda: Some(0.25),
        ..ElectroweakScales::default()
    };
    assert!((squared(&gas, &weighted, 0, 1) - 1.854375).abs() < 1e-14);
    assert!((squared(&gas, &weighted, 3, 0) - 0.673125).abs() < 1e-14);
    gas.distance_donors.distance = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 0.5,
        velocity_scale: 2.,
        lambda: 0.25,
        periodic: None,
    };
    assert!((squared(&gas, &configured, 0, 1) - 6.81046875).abs() < 1e-13);
    gas.distance_donors.distance = Distance::Euclidean {
        field: "positions".into(),
        scales: vec![],
        squared: false,
        periodic: Some(BoxDomain {
            lower: vec![0.; 3],
            upper: vec![1.; 3],
        }),
    };
    assert!((squared(&gas, &configured, 0, 1) - 0.1925).abs() < 1e-14);
    // Without a velocity weight the velocities are not read at all.
    let mut still = state_a.clone();
    still.v = None;
    let distance = AmplitudeDistance::new(&gas, &gas.distance_donors.distance, &configured);
    assert!(!distance.as_ref().unwrap().needs_velocities());
    assert!(distance.unwrap().squared(&still, 0, 1).is_some());
    let weighted = AmplitudeDistance::new(&gas, &gas.distance_donors.distance, &weighted).unwrap();
    assert!(weighted.needs_velocities() && weighted.squared(&still, 0, 1).is_none());
    for distance in [
        Distance::Cosine {
            field: "positions".into(),
            zero_tolerance: 1e-12,
        },
        Distance::Euclidean {
            field: "features".into(),
            scales: vec![],
            squared: false,
            periodic: None,
        },
    ] {
        assert!(matches!(
            AmplitudeDistance::new(&gas, &distance, &configured),
            Err(GasError::Capability(_))
        ));
        assert!(AmplitudeDistance::new(&gas, &distance, &ElectroweakScales::default()).is_ok());
    }
}

#[test]
fn unclipped_score_matches_the_clipped_gated_acceptance_law() {
    let mut decision = GasConfig::default().clone_decision;
    for (own, donor, epsilon, saturation, expected_score, probability) in [
        (1.2, 2.1, 0.05, 1., 0.72, 0.72),
        (0.2, 1., 0.05, 1., 3.2, 1.),
        (0.2, 1., 0.05, 4., 3.2, 0.8),
        (2.1, 1.55, 0.05, 1., -0.255813953488372, 0.),
        (1.2, 2.1, 0., 1., 0.75, 0.75),
    ] {
        decision.epsilon = epsilon;
        decision.saturation = saturation;
        decision.every = 20;
        let s = score::score(own, donor, epsilon);
        assert!((s - expected_score).abs() < 1e-14);
        for period in 0..3 {
            let on = decision.acceptance_probability(period * 20, own, donor);
            assert_eq!(on, (s / saturation).clamp(0., 1.));
            assert!((on - probability).abs() < 1e-14);
            for off in 1..20 {
                assert_eq!(
                    decision.acceptance_probability(period * 20 + off, own, donor),
                    0.
                );
            }
        }
        // Weighted antisymmetry, opposite signs and a vanishing self score.
        let back = score::score(donor, own, epsilon);
        assert!(((own + epsilon) * s + (donor + epsilon) * back).abs() < 1e-15);
        assert!(s * back < 0.);
        assert_eq!(score::score(own, own, epsilon), 0.);
    }
}

#[test]
fn recorded_clone_decisions_are_the_clipped_score_on_cloning_steps_and_zero_off_them() {
    use algorithmic_gas::{
        ExecutionContext, GasBuilder, InputBatch, ObservationBatch, Population, Provenance,
        RewardBatch, TensorBatch,
        domain::{GradientProvider, OperatorFuture, RewardSource},
    };
    struct Well;
    impl RewardSource<f64> for Well {
        fn id(&self) -> String {
            "well/v1".into()
        }
        fn evaluate<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: Option<&'a InputBatch<f64>>,
            stage: &'a str,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, RewardBatch<f64>> {
            Box::pin(async move {
                let x = p.observations.field("positions")?;
                let reward = (0..p.len())
                    .map(|i| Ok(-x.row(i)?.iter().map(|a| a * a).sum::<f64>()))
                    .collect::<algorithmic_gas::Result<Vec<f64>>>()?;
                Ok(RewardBatch::new(
                    reward,
                    Provenance {
                        population_version: p.version,
                        stage: stage.into(),
                        ..Default::default()
                    },
                ))
            })
        }
    }
    impl GradientProvider<f64> for Well {
        fn id(&self) -> String {
            "well-gradient/v1".into()
        }
        fn gradient<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, TensorBatch<f64>> {
            Box::pin(async move { TensorBatch::vectors(p.len(), 2, vec![0.; p.len() * 2]) })
        }
    }
    let archive = futures_lite::future::block_on(async {
        let (n, d) = (8, 2);
        let positions = (0..n * d)
            .map(|k| -1. + 2. * ((k * 7 + 3) % 97) as f64 / 97.)
            .collect();
        let mut observations =
            ObservationBatch::positions(TensorBatch::vectors(n, d, positions).unwrap());
        observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(n, d, vec![0.; n * d]).unwrap(),
        );
        let mut config = GasConfig::euclidean(d, 0.05).unwrap();
        config.seed = 7;
        config.clone_decision.every = 3;
        config.clone_decision.saturation = 2.;
        let mut gas = GasBuilder::new(Population::new(observations).unwrap(), Well)
            .config(config)
            .gradient(Well)
            .build()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..9 {
            gas.step().await.unwrap();
        }
        gas.recording().unwrap().clone()
    });
    let decision = &archive.gas_config.clone_decision;
    let (mut on, mut off, mut accepted_scale, mut gated) = (0, 0, 0, 0);
    for step in &archive.steps {
        let cloning = step.report.step.is_multiple_of(decision.every);
        for (i, choice) in step.report.clone_plan.choices.iter().enumerate() {
            let Some(donor) = choice.donors.first().filter(|_| !choice.revival) else {
                continue;
            };
            let own = step.report.pre_clone_fitness.fitness[i];
            let companion = step.donor_fitness[donor.pool_index as usize];
            let s = score::score(own, companion, decision.epsilon);
            let p = decision.acceptance_probability(step.report.step, own, companion);
            assert_eq!(choice.probability, Some(p));
            if cloning {
                assert_eq!(p, (s / decision.saturation).clamp(0., 1.));
                on += 1;
                accepted_scale += usize::from(p > 0. && p < 1.);
            } else {
                assert_eq!(p, 0.);
                assert!(!choice.accepted);
                off += 1;
                gated += usize::from(s > 0.);
            }
        }
    }
    // Both branches are exercised, with unclipped positive scores on each.
    assert!(on >= 8 && off >= 16 && accepted_scale >= 1 && gated >= 1);
}

#[test]
fn walker_roles_follow_the_accepted_decisions_and_the_ungated_score_sign() {
    use WalkerRole::{Cloner, Persister, StrongResister, WeakResister};
    let setup = setup(scales());
    let (frame_a, ..) = cycle();
    let state_a = state(&frame_a, &setup.0);
    let expected = [0.72, 0.7, -0.255813953488372, 0.470588235294117, -0.6875];
    for (s, e) in state_a.score.as_ref().unwrap().iter().zip(expected) {
        assert!((s - e).abs() < 1e-14);
    }
    assert_eq!(
        state_a.role.as_deref().unwrap(),
        [Cloner, WeakResister, StrongResister, Cloner, Persister]
    );
    // Off the cloning period nobody clones; weak resisters keep their role
    // because it follows the ungated score.
    let mut gated = frame_a.clone();
    gated.cloned = vec![false; 5];
    let scores = state_a.score.clone().unwrap();
    assert_eq!(
        score::roles(&gated, &scores),
        [
            WeakResister,
            WeakResister,
            Persister,
            WeakResister,
            Persister
        ]
    );
    // A revived row is no cloner and makes nobody a strong resister; a
    // self-paired walker is a persister.
    let eligible = [true, true, true, true, true, false];
    let frame_c = frame(
        &eligible,
        &[Some(1), Some(5), Some(0), Some(4), Some(3), None],
        &some(&[2, 0, 1, 4, 4, 3]),
        &[true, false, false, false, false, true],
    );
    let state_c = state(&frame_c, &setup.0);
    assert_eq!(
        state_c.role.as_deref().unwrap(),
        [
            Cloner,
            WeakResister,
            StrongResister,
            WeakResister,
            Persister,
            Persister
        ]
    );
    // A historical companion is no walker of the frame: the slot it names is not targeted.
    let mut historical = frame_a.clone();
    historical.cloning.as_mut().unwrap().historical[0] = true;
    assert_eq!(
        score::roles(&historical, &scores),
        [Cloner, WeakResister, Persister, Cloner, Persister]
    );
}

#[test]
fn score_field_masks_rows_without_a_decision_input_and_never_holds_nan() {
    let setup = setup(scales());
    let mut frame_m = frame(
        &[true, true, false, true],
        &some(&[1, 0, 3, 2]),
        &[Some(1), None, Some(0), Some(1)],
        &[false, false, true, true],
    );
    // Walker 3 cloned from a historical source whose rescored fitness is recorded.
    frame_m.cloning.as_mut().unwrap().historical[3] = true;
    frame_m.companion_fitness.as_mut().unwrap()[3] = 9.;
    let state_m = state(&frame_m, &setup.0);
    assert_eq!(state_m.score_valid, [true, false, false, true]);
    let scores = state_m.score.as_ref().unwrap();
    assert!((scores[0] - (0.45 - 1.2) / 1.25).abs() < 1e-15);
    assert_eq!(&scores[1..3], [0., 0.]);
    assert!((scores[3] - (9. - 0.8) / (0.8 + 0.05)).abs() < 1e-15);
    assert_eq!(state_m.cloning_companion.as_ref().unwrap()[3], NO_COMPANION);
    assert_eq!(
        state_m.role.as_deref().unwrap(),
        [
            WalkerRole::Persister,
            WalkerRole::Persister,
            WalkerRole::Persister,
            WalkerRole::Cloner
        ]
    );
    // Walker 0 sees only walker 1, which has no score; walker 3 has no
    // companion of the frame: nobody has a gradient.
    assert!(
        state_m
            .score_gradient
            .as_ref()
            .unwrap()
            .iter()
            .all(|g| *g == 0.)
    );
    // The decision fitness belongs to the first cloning entry: a row whose
    // first entry is unsampled has no score and targets nobody.
    let mut wide = frame_m.clone();
    wide.cloning = Some(Companions {
        count: 2,
        slot: vec![1, 3, 0, 0, 0, 0, 0, 1],
        generation: vec![0; 8],
        valid: vec![true, true, false, true, true, false, false, true],
        historical: vec![false; 8],
        mutual: false,
    });
    wide.validate().unwrap();
    let mut state_w = state_m.clone();
    assert!(score::fill(&wide, &setup.0, &mut state_w).is_available());
    assert_eq!(state_w.score_valid, [true, false, false, false]);
    assert_eq!(state_w.score.as_ref().unwrap()[3], 0.);
    assert_eq!(
        state_w.role.as_deref().unwrap(),
        [
            WalkerRole::Persister,
            WalkerRole::Persister,
            WalkerRole::Persister,
            WalkerRole::Cloner
        ]
    );
    // A nonfinite fitness invalidates the score instead of leaking into it.
    let mut broken = frame_m.clone();
    broken.fitness.as_mut().unwrap()[0] = f64::NAN;
    let state_b = state(&broken, &setup.0);
    assert_eq!(state_b.score_valid, [false, false, false, true]);
    assert_eq!(state_b.score.as_ref().unwrap()[0], 0.);
    let mut missing = frame_m.clone();
    missing.fitness = None;
    let mut untouched = FrameState::default();
    let availability = score::fill(&missing, &setup.0, &mut untouched);
    assert!(availability.reason().unwrap().contains("fitness"));
    assert_eq!(untouched, FrameState::default());
}

#[test]
fn score_gradient_averages_the_companion_differences_along_minimum_image_directions() {
    let mut gas = GasConfig::default();
    gas.clone_decision.epsilon = 0.;
    let cloning = [1u32, 2, 0, 0];
    let distance = [2u32, 0, 1, 0];
    let fitness = [1., 3., 4., 2.];
    let frame = Frame {
        step: 1,
        n: 4,
        d: 2,
        x: vec![0., 0., 3., 4., 0., 2., 0., 0.],
        eligible: vec![true; 4],
        generation: vec![0; 4],
        fitness: Some(fitness.to_vec()),
        distance: Some(companions(&some(&distance))),
        cloning: Some(companions(&some(&cloning))),
        companion_fitness: Some(cloning.iter().map(|&k| fitness[k as usize]).collect()),
        cloned: vec![false; 4],
        revived: vec![false; 4],
        ..Frame::default()
    };
    frame.validate().unwrap();
    let base = FrameState {
        n: 4,
        d: 2,
        x: frame.x.clone(),
        eligible: frame.eligible.clone(),
        distance_companion: Some(distance.to_vec()),
        cloning_companion: Some(cloning.to_vec()),
        ..FrameState::default()
    };
    let mut open = base.clone();
    assert!(score::fill(&frame, &gas, &mut open).is_available());
    // S = (2, 1/3, −3/4, −1/2). Walker 0 sees walker 2 along (0, 1) and
    // walker 1 along (3, 4)/5; walker 3 sits on both of its companions. The
    // gradient divides each difference by its own length, so the pair at
    // distance 5 weighs a fifth of what the pair at distance 2 does.
    let gradient = open.score_gradient.as_ref().unwrap();
    assert!((gradient[0] + 0.1).abs() < 1e-14);
    assert!((gradient[1] + 197. / 240.).abs() < 1e-14);
    assert_eq!(&gradient[6..], [0., 0.]);
    gas.boundary = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![0.; 2],
            upper: vec![5.; 2],
        },
    };
    let mut wrapped = base.clone();
    assert!(score::fill(&frame, &gas, &mut wrapped).is_available());
    // In a box of length 5 the image of (3, 4) is (−2, −1), of squared length 5.
    let gradient = wrapped.score_gradient.as_ref().unwrap();
    assert!((gradient[0] - 0.5 * (10. / 3.) / 5.).abs() < 1e-14);
    assert!((gradient[1] - 0.5 * (-1.375 + (5. / 3.) / 5.)).abs() < 1e-14);
    assert_eq!(open.score, wrapped.score);
    // A box of another dimension gives no direction, never the open one.
    gas.boundary = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![0.; 3],
            upper: vec![5.; 3],
        },
    };
    let mut skewed = base.clone();
    assert!(score::fill(&frame, &gas, &mut skewed).is_available());
    assert!(skewed.score_gradient.unwrap().iter().all(|g| *g == 0.));
    assert_eq!(open.score, skewed.score);
    // Every axis wraps with its own length: in a box of 5 by 10 the image of
    // (3, 4) is (−2, 4), of squared length 20.
    gas.boundary = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![0.; 2],
            upper: vec![5., 10.],
        },
    };
    let mut oblong = base.clone();
    assert!(score::fill(&frame, &gas, &mut oblong).is_available());
    let gradient = oblong.score_gradient.as_ref().unwrap();
    assert!((gradient[0] - 0.5 * (10. / 3.) / 20.).abs() < 1e-14);
    assert!((gradient[1] - 0.5 * (-1.375 - (20. / 3.) / 20.)).abs() < 1e-14);
    assert!((gradient[0] - 0.08333333333333333).abs() < 1e-14);
    assert!((gradient[1] + 0.8541666666666666).abs() < 1e-14);
}

#[test]
fn chirality_site_scalars_and_role_speeds_reproduce_the_reference_frames() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame_a, distance, cloning) = cycle();
    let state_a = state(&frame_a, &setup.0);
    let sites = elements(ElementKind::Site, &distance, &cloning);
    let real = |spec: &ChannelSpec, sites: &[Element], state: &FrameState| {
        values(spec, sites, state, &context)
            .iter()
            .map(|v| v.map(|v| v[0]))
            .collect::<Vec<Option<f64>>>()
    };
    let chi = real(&chirality(ChiralityObservable::Chi), &sites, &state_a);
    assert_eq!(chi, [1., -1., 1., 1., -1.].map(Some));
    let left = real(
        &chirality(ChiralityObservable::LeftFraction),
        &sites,
        &state_a,
    );
    assert_eq!(left, [1., 0., 1., 1., 0.].map(Some));
    let phase = real(&ChannelSpec::FitnessPhase, &sites, &state_a);
    let total: f64 = phase.iter().flatten().sum();
    assert!((total / 5. + 1.74285714285714).abs() < 1e-14);
    assert_eq!(phase[1], Some(-0.45 / 0.7));
    let cloned = real(&ChannelSpec::CloneIndicator, &sites, &state_a);
    assert_eq!(cloned, [1., 0., 0., 1., 0.].map(Some));
    for (role, members, speed) in [
        (WalkerRole::Cloner, vec![0, 3], 0.443256638487687),
        (WalkerRole::StrongResister, vec![2], 0.552268050859363),
        (WalkerRole::WeakResister, vec![1], 0.51720402163943),
        (WalkerRole::Persister, vec![4], 0.455521678957215),
    ] {
        let speeds = real(&ChannelSpec::ParityVelocity { role }, &sites, &state_a);
        let valid: Vec<usize> = (0..5).filter(|&i| speeds[i].is_some()).collect();
        assert_eq!(valid, members);
        let total: f64 = speeds.iter().flatten().sum();
        assert!((total / members.len() as f64 - speed).abs() < 1e-14);
    }
    // A dead, revived walker: masked on every site scalar, and the fixed 1/N
    // means keep the identity with the left fraction.
    let eligible = [true, true, true, true, true, false];
    let distance = [Some(1), Some(5), Some(0), Some(4), Some(3), None];
    let cloning = some(&[2, 0, 1, 4, 4, 3]);
    let frame_c = frame(
        &eligible,
        &distance,
        &cloning,
        &[true, false, false, false, false, true],
    );
    let state_c = state(&frame_c, &setup.0);
    let sites = elements(ElementKind::Site, &distance, &cloning);
    let chi = real(&chirality(ChiralityObservable::Chi), &sites, &state_c);
    assert_eq!(
        chi,
        [Some(1.), Some(-1.), Some(1.), Some(-1.), Some(-1.), None]
    );
    let left = real(
        &chirality(ChiralityObservable::LeftFraction),
        &sites,
        &state_c,
    );
    let total = |v: &[Option<f64>]| v.iter().flatten().sum::<f64>();
    let living = 5.;
    assert!((total(&chi) / 6. + 0.166666666666667).abs() < 1e-14);
    assert!((total(&chi) / living + 0.2).abs() < 1e-15);
    assert!((total(&left) / 6. - 0.333333333333333).abs() < 1e-14);
    assert!((total(&chi) / 6. - (2. * total(&left) / 6. - living / 6.)).abs() < 1e-15);
    assert!((total(&chi) / living - (2. * total(&left) / living - 1.)).abs() < 1e-15);
    // Revivals are no clone events of a living walker.
    let cloned = real(&ChannelSpec::CloneIndicator, &sites, &state_c);
    assert_eq!(cloned[5], None);
    assert!((total(&cloned) / living - 0.2).abs() < 1e-15);
    assert!((total(&cloned) / 6. - 0.166666666666667).abs() < 1e-14);
    let phase = real(&ChannelSpec::FitnessPhase, &sites, &state_c);
    assert!((total(&phase) / living + 1.22 / 0.7).abs() < 1e-14);
    assert!((total(&phase) / 6. + 1.01666666666667 / 0.7).abs() < 1e-13);
    for spec in [
        chirality(ChiralityObservable::Chi),
        chirality(ChiralityObservable::LeftFraction),
        ChannelSpec::CloneIndicator,
    ] {
        let signature = spec.signature(ElementKind::Site, &context).unwrap();
        assert_eq!(signature.normalization, Some(FrameNormalization::FixedN));
    }
}

#[test]
fn role_sets_partition_the_living_walkers_and_no_cloner_points_at_a_right_handed_walker() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let coupling = chirality(ChiralityObservable::LeftRightCoupling);
    let Err(GasError::Capability(reason)) = coupling.signature(ElementKind::CloningPair, &context)
    else {
        panic!("the left-right coupling has an empty support on every frame");
    };
    assert!(reason.contains("identically zero"));
    let n = 12;
    let (mut cloners, mut strong, mut weak) = (0, 0, 0);
    for sample in 0..200 {
        let mut rng = RandomStream::new(7, sample, Stream::Initialize, 0, 0);
        let eligible: Vec<bool> = (0..n).map(|_| rng.uniform::<f64>() < 0.8).collect();
        let map: Vec<Option<u32>> = (0..n)
            .map(|_| (rng.uniform::<f64>() < 0.9).then(|| rng.index(n) as u32))
            .collect();
        let fitness: Vec<f64> = (0..n).map(|_| 0.1 + 2. * rng.uniform::<f64>()).collect();
        let cloned: Vec<bool> = (0..n)
            .map(|i| map[i].is_some() && rng.uniform::<f64>() < 0.4)
            .collect();
        let mut cloning = companions(&map);
        for h in &mut cloning.historical {
            *h = rng.uniform::<f64>() < 0.1;
        }
        let frame = Frame {
            step: 1,
            n,
            d: 1,
            x: (0..n).map(|i| i as f64).collect(),
            eligible: eligible.clone(),
            generation: vec![0; n],
            fitness: Some(fitness.clone()),
            distance: Some(companions(&map)),
            companion_fitness: Some(
                map.iter()
                    .map(|k| k.map_or(0., |k| fitness[k as usize]))
                    .collect(),
            ),
            revived: cloned
                .iter()
                .zip(&eligible)
                .map(|(c, e)| *c && !e)
                .collect(),
            cloned: cloned.clone(),
            cloning: Some(cloning.clone()),
            ..Frame::default()
        };
        frame.validate().unwrap();
        let mut state = FrameState {
            n,
            d: 1,
            x: frame.x.clone(),
            fitness: frame.fitness.clone(),
            eligible: eligible.clone(),
            cloned: cloned.clone(),
            cloning_companion: Some(
                (0..n)
                    .map(|i| {
                        cloning
                            .first(i, &eligible)
                            .map_or(NO_COMPANION, |_| cloning.slot[i])
                    })
                    .collect(),
            ),
            ..FrameState::default()
        };
        assert!(score::fill(&frame, &setup.0, &mut state).is_available());
        let roles = state.role.clone().unwrap();
        // The four sets as the partition defines them, built independently.
        let delta: Vec<bool> = (0..n).map(|i| eligible[i] && cloned[i]).collect();
        let targeted: Vec<bool> = (0..n)
            .map(|s| (0..n).any(|i| delta[i] && !cloning.historical[i] && map[i] == Some(s as u32)))
            .collect();
        for i in 0..n {
            let fitter = map[i].is_some_and(|k| fitness[k as usize] > fitness[i]);
            let expected = if !eligible[i] {
                WalkerRole::Persister
            } else if delta[i] {
                WalkerRole::Cloner
            } else if targeted[i] {
                WalkerRole::StrongResister
            } else if fitter {
                WalkerRole::WeakResister
            } else {
                WalkerRole::Persister
            };
            assert_eq!(roles[i], expected);
            cloners += usize::from(expected == WalkerRole::Cloner);
            strong += usize::from(expected == WalkerRole::StrongResister);
            weak += usize::from(expected == WalkerRole::WeakResister);
        }
        let companion = state.cloning_companion.clone().unwrap();
        let mut out = [0.; 2];
        for i in 0..n {
            let k = companion[i];
            if k == NO_COMPANION {
                continue;
            }
            if roles[i] == WalkerRole::Cloner {
                assert!(matches!(
                    roles[k as usize],
                    WalkerRole::Cloner | WalkerRole::StrongResister
                ));
            }
            let pair = element(ElementKind::CloningPair, i as u32, k, i as u32);
            assert!(!coupling.evaluate(&pair, &state, &context, &mut out));
        }
        let sites = elements(ElementKind::Site, &map, &map);
        let total = |observable| -> f64 {
            values(&chirality(observable), &sites, &state, &context)
                .iter()
                .flatten()
                .map(|v| v[0])
                .sum()
        };
        let living = eligible.iter().filter(|e| **e).count() as f64;
        let (chi, left) = (
            total(ChiralityObservable::Chi),
            total(ChiralityObservable::LeftFraction),
        );
        assert_eq!(chi, 2. * left - living);
    }
    // Every role is exercised. A walker clones with p = 0.8 * 0.9 * 0.4 = 0.288:
    // 691 of 2400 on average, sigma = 22. The two resister roles are rarer, and
    // the floor of 100 lies more than 10 sigma below each of the three counts.
    assert!(cloners > 100 && strong > 100 && weak > 100);
    // The operator is not vacuous: on roles that violate the partition it
    // returns the fitness phase of the pair with the positive sign.
    let broken = FrameState {
        n: 2,
        d: 1,
        fitness: Some(vec![1.2, 0.45]),
        eligible: vec![true; 2],
        role: Some(vec![WalkerRole::Cloner, WalkerRole::Persister]),
        ..FrameState::default()
    };
    let mut out = [0.; 2];
    let pair = element(ElementKind::CloningPair, 0, 1, 0);
    assert!(coupling.evaluate(&pair, &broken, &context, &mut out));
    let theta: f64 = (0.45 - 1.2) / 0.7;
    assert!(close(out, [theta.cos(), theta.sin()]));
}

#[test]
fn electroweak_signatures_state_components_parities_and_no_spatial_parity() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let family = [
        "u1",
        "su2",
        "electroweak_mixed",
        "fitness_phase",
        "clone_indicator",
        "parity_velocity",
        "chirality",
    ];
    let mut described = 0;
    for spec in ChannelSpec::all()
        .into_iter()
        .filter(|s| family.contains(&s.family()))
    {
        for kind in spec.kinds(PairSelection::Both) {
            let Ok(signature) = spec.signature(kind, &context) else {
                assert_eq!(spec, chirality(ChiralityObservable::LeftRightCoupling));
                continue;
            };
            let complex = matches!(
                spec,
                ChannelSpec::U1 { .. } | ChannelSpec::Su2 { .. } | ChannelSpec::ElectroweakMixed
            );
            assert_eq!(signature.components, if complex { 2 } else { 1 });
            assert_eq!(signature.descriptor.spatial_parity, None);
            assert!(!signature.descriptor.definition.is_empty());
            assert!(!signature.descriptor.note.is_empty());
            assert!(signature.correlatable);
            assert_eq!(signature.degenerate, spec == ChannelSpec::ElectroweakMixed);
            let expected = match spec {
                ChannelSpec::Su2 {
                    mode: Su2Mode::Doublet,
                    ..
                } => ExchangeParity::Even,
                ChannelSpec::Su2 {
                    mode: Su2Mode::DoubletDiff,
                    ..
                } => ExchangeParity::Odd,
                _ if complex => ExchangeParity::Mixed,
                _ => ExchangeParity::Even,
            };
            assert_eq!(signature.exchange, expected);
            // Arms the book does not define carry no label.
            let non_book = matches!(
                spec,
                ChannelSpec::Su2 { directed: true, .. }
                    | ChannelSpec::ElectroweakMixed
                    | ChannelSpec::FitnessPhase
                    | ChannelSpec::CloneIndicator
                    | ChannelSpec::ParityVelocity { .. }
            ) || matches!(spec, ChannelSpec::U1 { charge, .. } if charge != 1);
            assert_eq!(signature.descriptor.book_label.is_empty(), non_book);
            described += 1;
            for other in [
                ElementKind::Site,
                ElementKind::DistancePair,
                ElementKind::CloningPair,
                ElementKind::Triplet,
            ] {
                if other != kind {
                    assert!(matches!(
                        spec.signature(other, &context),
                        Err(GasError::Capability(_))
                    ));
                }
            }
        }
    }
    assert!(described >= 17);
    let requires = |spec: ChannelSpec, kind| spec.signature(kind, &context).unwrap().requires;
    let phase = requires(u1(U1Mode::Phase, 1), ElementKind::DistancePair);
    assert_eq!(
        phase.records.into_iter().collect::<Vec<_>>(),
        [Record::Fitness]
    );
    // λ = 0.25 of this setup reads velocities inside every amplitude.
    for (spec, kind) in [
        (u1(U1Mode::Dressed, 1), ElementKind::DistancePair),
        (su2(Su2Mode::Component, false), ElementKind::CloningPair),
        (ChannelSpec::ElectroweakMixed, ElementKind::Triplet),
    ] {
        assert!(requires(spec, kind).records.contains(&Record::Velocities));
    }
    let plain = requires(su2(Su2Mode::Phase, false), ElementKind::CloningPair);
    assert!(plain.records.contains(&Record::ClonePlan));
    assert!(!plain.records.contains(&Record::Velocities));
    let speed = requires(
        ChannelSpec::ParityVelocity {
            role: WalkerRole::Cloner,
        },
        ElementKind::Site,
    );
    assert!(speed.records.contains(&Record::Velocities));
    assert!(speed.records.contains(&Record::CloningCompanions));
}

#[test]
fn wrong_kinds_short_buffers_missing_records_and_nonfinite_fitness_mask_the_element() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, ..) = cycle();
    let state_a = state(&frame, &setup.0);
    let role = WalkerRole::Cloner;
    let cases = [
        (u1(U1Mode::Dressed, 1), ElementKind::DistancePair, [0, 1, 0]),
        (
            su2(Su2Mode::Doublet, false),
            ElementKind::CloningPair,
            [0, 2, 0],
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [0, 1, 2],
        ),
        (ChannelSpec::FitnessPhase, ElementKind::Site, [0; 3]),
        (ChannelSpec::CloneIndicator, ElementKind::Site, [0; 3]),
        (
            ChannelSpec::ParityVelocity { role },
            ElementKind::Site,
            [0; 3],
        ),
        (
            chirality(ChiralityObservable::Chi),
            ElementKind::Site,
            [0; 3],
        ),
        (
            chirality(ChiralityObservable::LeftFraction),
            ElementKind::Site,
            [0; 3],
        ),
    ];
    for (spec, kind, [i, j, k]) in &cases {
        let components = spec.signature(*kind, &context).unwrap().components;
        let mut out = [7.; 2];
        assert!(spec.evaluate(&element(*kind, *i, *j, *k), &state_a, &context, &mut out));
        // A buffer shorter than the components and an element of another kind.
        let mut out = [7.; 2];
        assert!(!spec.evaluate(
            &element(*kind, *i, *j, *k),
            &state_a,
            &context,
            &mut out[..components - 1]
        ));
        for other in [
            ElementKind::Site,
            ElementKind::DistancePair,
            ElementKind::CloningPair,
            ElementKind::Triplet,
        ] {
            if other != *kind {
                assert!(!spec.evaluate(&element(other, *i, *j, *k), &state_a, &context, &mut out));
            }
        }
        // A walker beyond the frame.
        assert!(!spec.evaluate(&element(*kind, 9, *j, *k), &state_a, &context, &mut out));
        assert_eq!(out, [7.; 2]);
    }
    // A nonfinite fitness at either end masks the phase; it never becomes a value.
    let mut broken = state_a.clone();
    broken.fitness.as_mut().unwrap()[2] = f64::INFINITY;
    let mut out = [7.; 2];
    for (spec, kind, walkers) in [
        (u1(U1Mode::Phase, 1), ElementKind::DistancePair, [1, 2, 1]),
        (u1(U1Mode::Phase, 1), ElementKind::DistancePair, [2, 3, 2]),
        (
            su2(Su2Mode::Phase, false),
            ElementKind::CloningPair,
            [0, 2, 0],
        ),
        // Walker 4 points at 1, 1 at 3, 3 at 0, 0 at 2: only the second hop of 3 breaks.
        (
            su2(Su2Mode::Doublet, false),
            ElementKind::CloningPair,
            [3, 0, 3],
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [0, 1, 2],
        ),
        (ChannelSpec::FitnessPhase, ElementKind::Site, [2; 3]),
    ] {
        let [i, j, k] = walkers;
        assert!(!spec.evaluate(&element(kind, i, j, k), &broken, &context, &mut out));
    }
    assert!(su2(Su2Mode::Component, false).evaluate(
        &element(ElementKind::CloningPair, 3, 0, 3),
        &broken,
        &context,
        &mut out
    ));
    // Missing records mask: no roles, no velocities, no fitness, no companion map.
    let site = element(ElementKind::Site, 0, 0, 0);
    let mut bare = state_a.clone();
    bare.role = None;
    bare.v = None;
    let mut out = [7.; 2];
    assert!(!chirality(ChiralityObservable::Chi).evaluate(&site, &bare, &context, &mut out));
    assert!(!ChannelSpec::ParityVelocity { role }.evaluate(&site, &bare, &context, &mut out));
    bare.role = state_a.role.clone();
    assert!(!ChannelSpec::ParityVelocity { role }.evaluate(&site, &bare, &context, &mut out));
    let pair = element(ElementKind::DistancePair, 0, 1, 0);
    assert!(!u1(U1Mode::Dressed, 1).evaluate(&pair, &bare, &context, &mut out));
    assert!(u1(U1Mode::Phase, 1).evaluate(&pair, &bare, &context, &mut out));
    bare.fitness = None;
    assert!(!u1(U1Mode::Phase, 1).evaluate(&pair, &bare, &context, &mut out));
    assert!(!ChannelSpec::FitnessPhase.evaluate(&site, &bare, &context, &mut out));
    let mut unmapped = state_a.clone();
    unmapped.cloning_companion = None;
    let pair = element(ElementKind::CloningPair, 0, 2, 0);
    let mut out = [7.; 2];
    assert!(!su2(Su2Mode::DoubletDiff, false).evaluate(&pair, &unmapped, &context, &mut out));
    assert_eq!(out, [7.; 2]);
    assert!(su2(Su2Mode::Component, false).evaluate(&pair, &unmapped, &context, &mut out));
    // A charge outside 1..=8 is a configuration error, not a constant series.
    for charge in [0, 9] {
        assert!(matches!(
            u1(U1Mode::Phase, charge).signature(ElementKind::DistancePair, &context),
            Err(GasError::Configuration(_))
        ));
    }
}

#[test]
fn pair_operators_follow_the_evaluated_state_and_never_read_its_score() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, distance, cloning) = cycle();
    let state_a = state(&frame, &setup.0);
    // The score fields of a sink belong to other companions: no pair value moves with them.
    let mut scrambled = state_a.clone();
    scrambled.score = Some(vec![-3., 8., 0., f64::NAN, 1e9]);
    scrambled.score_valid = vec![false; 5];
    scrambled.score_gradient = None;
    scrambled.role = None;
    scrambled.cloned = vec![true; 5];
    let mut specs = vec![
        (u1(U1Mode::Dressed, 2), ElementKind::DistancePair),
        (ChannelSpec::ElectroweakMixed, ElementKind::Triplet),
    ];
    for mode in [
        Su2Mode::Phase,
        Su2Mode::Component,
        Su2Mode::Doublet,
        Su2Mode::DoubletDiff,
    ] {
        for directed in [false, true] {
            specs.push((su2(mode, directed), ElementKind::CloningPair));
        }
    }
    for (spec, kind) in &specs {
        let elements = elements(*kind, &distance, &cloning);
        assert_eq!(
            values(spec, &elements, &state_a, &context),
            values(spec, &elements, &scrambled, &context)
        );
    }
    // A frozen pair on a later state: the fitness and the second hop are the
    // ones of that state, the first hop stays the element.
    let mut sink = state_a.clone();
    sink.fitness = Some(vec![0.7, 1.9, 0.6, 1.1, 2.4]);
    let pair = [element(ElementKind::CloningPair, 0, 2, 0)];
    let run = |mode, state: &FrameState| values(&su2(mode, false), &pair, state, &context);
    assert_values(
        &run(Su2Mode::Component, &sink),
        &[[0.5234544902089371, -0.20965266117102252]],
    );
    assert_values(
        &run(Su2Mode::Doublet, &sink),
        &[[0.48625474255245593, 0.429828529325364]],
    );
    sink.cloning_companion.as_mut().unwrap()[2] = 1;
    assert_values(
        &run(Su2Mode::Doublet, &sink),
        &[[0.8610411071264066, -0.42551133236162236]],
    );
    assert_values(
        &run(Su2Mode::Component, &sink),
        &[[0.5234544902089371, -0.20965266117102252]],
    );
    sink.cloning_companion.as_mut().unwrap()[2] = NO_COMPANION;
    assert_eq!(run(Su2Mode::Doublet, &sink), [None]);
    // Equal fitness is the phase zero: the directed value is 1, not a mask.
    sink.fitness.as_mut().unwrap()[2] = 0.7;
    for directed in [false, true] {
        let z = values(&su2(Su2Mode::Phase, directed), &pair, &sink, &context);
        assert_eq!(z, [Some([1., 0.])]);
    }
}

#[test]
fn mixed_triplet_masks_a_self_distance_companion_whatever_its_cloning_companion() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, ..) = cycle();
    let state_a = state(&frame, &setup.0);
    let mixed = |i, j, k| {
        values(
            &ChannelSpec::ElectroweakMixed,
            &[element(ElementKind::Triplet, i, j, k)],
            &state_a,
            &context,
        )[0]
    };
    // The cloning hop 0 → 2 is defined, the distance pair 0 → 0 is not.
    assert_eq!(mixed(0, 0, 2), None);
    assert_eq!(mixed(0, 2, 0), None);
    // Both companions on walker 2: D² = 1.71 + 0.25 · 0.585 in both amplitudes.
    let squared: f64 = 1.71 + 0.25 * 0.585;
    let r = (-squared / (4. * 1.3 * 1.3)).exp() * (-squared / (4. * 0.9 * 0.9)).exp();
    let angle: f64 = -(2.1 - 1.2) / 0.7 + (2.1 - 1.2) / ((1.2 + 0.05) * 0.35);
    let z = mixed(0, 2, 2).unwrap();
    assert!(close(z, [r * angle.cos(), r * angle.sin()]));
    assert!(close(z, [0.30718435174258113, 0.2987195705362341]));
}

#[test]
fn infinite_distances_and_overflowing_phases_mask_the_element_never_a_zero_amplitude() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, ..) = cycle();
    let state_a = state(&frame, &setup.0);
    // An infinite coordinate gives D² = ∞, whose Gaussian amplitude would be a valid 0.
    let mut far = state_a.clone();
    far.x[3] = f64::INFINITY;
    let raw = AmplitudeDistance::new(&setup.0, &setup.0.distance_donors.distance, &scales());
    assert_eq!(raw.as_ref().unwrap().squared(&far, 0, 1), None);
    assert_eq!(raw.as_ref().unwrap().squared(&far, 1, 0), None);
    assert!(raw.unwrap().squared(&far, 0, 2).is_some());
    let mut out = [7.; 2];
    for (spec, kind, [i, j, k], valid) in [
        (
            u1(U1Mode::Dressed, 1),
            ElementKind::DistancePair,
            [0, 1, 0],
            false,
        ),
        (
            u1(U1Mode::Phase, 1),
            ElementKind::DistancePair,
            [0, 1, 0],
            true,
        ),
        (
            su2(Su2Mode::Component, false),
            ElementKind::CloningPair,
            [1, 3, 1],
            false,
        ),
        (
            su2(Su2Mode::Phase, false),
            ElementKind::CloningPair,
            [1, 3, 1],
            true,
        ),
        // The first hop 2 → 4 is finite; the second hop 4 → 1 is not.
        (
            su2(Su2Mode::Component, false),
            ElementKind::CloningPair,
            [2, 4, 2],
            true,
        ),
        (
            su2(Su2Mode::Doublet, false),
            ElementKind::CloningPair,
            [2, 4, 2],
            false,
        ),
        (
            su2(Su2Mode::DoubletDiff, true),
            ElementKind::CloningPair,
            [2, 4, 2],
            false,
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [0, 1, 2],
            false,
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [4, 0, 1],
            false,
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [4, 0, 2],
            true,
        ),
    ] {
        let element = element(kind, i, j, k);
        assert_eq!(
            spec.evaluate(&element, &far, &context, &mut out),
            valid,
            "{} on {:?}",
            spec.id(),
            element.walkers
        );
    }
    // Finite fitness whose difference or quotient overflows: the phase has no
    // cosine, and the element is masked instead of carrying NaN.
    let mut steep = state_a.clone();
    steep.fitness = Some(vec![1.7e308, -1.7e308, 2.1, 0.8, 1.55]);
    steep.role = Some(vec![
        WalkerRole::Cloner,
        WalkerRole::Persister,
        WalkerRole::Persister,
        WalkerRole::Persister,
        WalkerRole::Persister,
    ]);
    for (spec, kind, [i, j, k]) in [
        (u1(U1Mode::Phase, 1), ElementKind::DistancePair, [0, 1, 0]),
        (u1(U1Mode::Dressed, 2), ElementKind::DistancePair, [1, 0, 1]),
        (
            su2(Su2Mode::Phase, false),
            ElementKind::CloningPair,
            [0, 1, 0],
        ),
        (
            su2(Su2Mode::Component, true),
            ElementKind::CloningPair,
            [0, 1, 0],
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [0, 1, 2],
        ),
        (ChannelSpec::FitnessPhase, ElementKind::Site, [0; 3]),
        (
            chirality(ChiralityObservable::LeftRightCoupling),
            ElementKind::CloningPair,
            [0, 1, 0],
        ),
    ] {
        let element = element(kind, i, j, k);
        assert!(
            !spec.evaluate(&element, &steep, &context, &mut out),
            "{} on {:?}",
            spec.id(),
            element.walkers
        );
    }
    // The same walkers with a representable phase are not masked.
    assert!(ChannelSpec::FitnessPhase.evaluate(
        &element(ElementKind::Site, 2, 2, 2),
        &steep,
        &context,
        &mut out
    ));
    assert_eq!(out[0], -2.1 / 0.7);
}

#[test]
fn score_gradient_is_the_mean_over_the_valid_pairs_not_over_the_two_roles() {
    let mut gas = GasConfig::default();
    gas.clone_decision.epsilon = 0.;
    let cloning = [Some(2), Some(3), Some(0), None];
    let distance = some(&[1, 0, 1, 0]);
    let fitness = [1., 3., 2., 5.];
    let frame = Frame {
        step: 1,
        n: 4,
        d: 2,
        x: vec![0., 0., 3., 4., 0., 0., 1., 0.],
        eligible: vec![true; 4],
        generation: vec![0; 4],
        fitness: Some(fitness.to_vec()),
        distance: Some(companions(&distance)),
        cloning: Some(companions(&cloning)),
        companion_fitness: Some(
            cloning
                .iter()
                .map(|k| k.map_or(0., |k| fitness[k as usize]))
                .collect(),
        ),
        cloned: vec![false; 4],
        revived: vec![false; 4],
        ..Frame::default()
    };
    frame.validate().unwrap();
    let base = FrameState {
        n: 4,
        d: 2,
        x: frame.x.clone(),
        eligible: frame.eligible.clone(),
        distance_companion: Some(vec![1, 0, 1, 0]),
        cloning_companion: Some(vec![2, 3, 0, NO_COMPANION]),
        ..FrameState::default()
    };
    // S = (1, 2/3, −1/2, none). Every walker with a score has one pair left:
    // 0 and 2 sit on their cloning companion, the cloning companion of 1 has no score.
    let expected = [
        [-0.04, -4. / 75.],
        [-0.04, -4. / 75.],
        [0.14, 14. / 75.],
        [0., 0.],
    ];
    let check = |state: &FrameState| {
        assert_eq!(state.score_valid, [true, true, true, false]);
        for (g, e) in state
            .score_gradient
            .as_ref()
            .unwrap()
            .chunks_exact(2)
            .zip(expected)
        {
            assert!((g[0] - e[0]).abs() < 1e-15 && (g[1] - e[1]).abs() < 1e-15);
        }
    };
    let mut filled = base.clone();
    assert!(score::fill(&frame, &gas, &mut filled).is_available());
    check(&filled);
    // A map entry beyond the frame is no pair either, and is never indexed.
    let mut beyond = base.clone();
    beyond.cloning_companion = Some(vec![99, 3, 99, NO_COMPANION]);
    assert!(score::fill(&frame, &gas, &mut beyond).is_available());
    check(&beyond);
    // One map alone gives the same single pairs.
    let mut single = base.clone();
    single.cloning_companion = None;
    assert!(score::fill(&frame, &gas, &mut single).is_available());
    check(&single);
    // The pairs are the ones of the state maps, not of the recorded entries:
    // a distance companion without a score leaves walker 0 without a pair.
    let mut moved = base.clone();
    moved.distance_companion = Some(vec![3, 0, 1, 0]);
    assert!(score::fill(&frame, &gas, &mut moved).is_available());
    let gradient = moved.score_gradient.as_ref().unwrap();
    assert_eq!(&gradient[..2], [0., 0.]);
    assert!((gradient[2] + 0.04).abs() < 1e-15 && (gradient[4] - 0.14).abs() < 1e-15);
}

#[test]
fn each_companion_role_takes_the_range_and_the_distance_of_its_own_donor_module() {
    let (frame, distance, cloning) = cycle();
    // Kernel widths 1.3 and 0.9 reproduce the reference vectors of the fixed ranges.
    let mut kernel = setup(ElectroweakScales {
        epsilon_d: Range::FromKernel,
        epsilon_c: Range::FromKernel,
        ..scales()
    });
    kernel.2.distance_kernel_width = Some(1.3);
    kernel.2.cloning_kernel_width = Some(0.9);
    let state_a = state(&frame, &kernel.0);
    let context = context_of(&kernel);
    let component = values(
        &su2(Su2Mode::Component, false),
        &elements(ElementKind::CloningPair, &distance, &cloning),
        &state_a,
        &context,
    );
    assert_values(&component, &COMPONENT);
    let dressed = values(
        &u1(U1Mode::Dressed, 1),
        &elements(ElementKind::DistancePair, &distance, &cloning),
        &state_a,
        &context,
    );
    assert!(close(
        mean(&dressed),
        [0.146954227163313, 0.123431251890567]
    ));
    let mixed = values(
        &ChannelSpec::ElectroweakMixed,
        &elements(ElementKind::Triplet, &distance, &cloning),
        &state_a,
        &context,
    );
    assert!(close(
        mean(&mixed),
        [0.0849444262654701, -0.0339751241233209]
    ));
    // Two donor modules with different distances: positions alone for the
    // distance role, D² = 1.6925, a scaled phase space for the cloning role,
    // D² = 6.81046875, each with the velocity weight of its own module.
    let mut gas = GasConfig::euclidean(3, 0.01).unwrap();
    gas.distance_donors.distance = Distance::Euclidean {
        field: "positions".into(),
        scales: vec![],
        squared: false,
        periodic: None,
    };
    gas.cloning_donors.distance = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 0.5,
        velocity_scale: 2.,
        lambda: 0.25,
        periodic: None,
    };
    let measurement = MeasurementConfig {
        electroweak: ElectroweakScales {
            distance: PairDistance::Configured,
            lambda: None,
            ..scales()
        },
        ..MeasurementConfig::default()
    };
    let donors = (gas, measurement, Capabilities::nominal());
    let context = context_of(&donors);
    let modulus = |spec: &ChannelSpec, kind| {
        let z = values(spec, &[element(kind, 0, 1, 0)], &state_a, &context)[0].unwrap();
        z[0].hypot(z[1])
    };
    let dressed = u1(U1Mode::Dressed, 1);
    let component = su2(Su2Mode::Component, false);
    let expected_d = (-1.6925f64 / (4. * 1.3 * 1.3)).exp();
    let expected_c = (-6.81046875f64 / (4. * 0.9 * 0.9)).exp();
    assert!((modulus(&dressed, ElementKind::DistancePair) - expected_d).abs() < 1e-14);
    assert!((modulus(&component, ElementKind::CloningPair) - expected_c).abs() < 1e-14);
    let z = values(
        &ChannelSpec::ElectroweakMixed,
        &[element(ElementKind::Triplet, 0, 1, 1)],
        &state_a,
        &context,
    )[0]
    .unwrap();
    assert!((z[0].hypot(z[1]) - expected_d * expected_c).abs() < 1e-14);
    // Only the role whose module weighs velocities reads them.
    let velocities = |spec: &ChannelSpec, kind| {
        let signature = spec.signature(kind, &context).unwrap();
        signature.requires.records.contains(&Record::Velocities)
    };
    assert!(!velocities(&dressed, ElementKind::DistancePair));
    assert!(velocities(&component, ElementKind::CloningPair));
    assert!(velocities(
        &ChannelSpec::ElectroweakMixed,
        ElementKind::Triplet
    ));
    let mut still = state_a.clone();
    still.v = None;
    let mut out = [0.; 2];
    let pair = element(ElementKind::DistancePair, 0, 1, 0);
    assert!(dressed.evaluate(&pair, &still, &context, &mut out));
    let pair = element(ElementKind::CloningPair, 0, 1, 0);
    assert!(!component.evaluate(&pair, &still, &context, &mut out));
}

#[test]
fn configured_amplitude_distances_agree_with_the_distance_of_the_engine() {
    use algorithmic_gas::{ObservationBatch, TensorBatch, geometry::AlgorithmicDistance};
    let (frame, ..) = cycle();
    let gas = GasConfig::euclidean(3, 0.01).unwrap();
    let state_a = state(&frame, &gas);
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(5, 3, X[..15].to_vec()).unwrap());
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(5, 3, V[..15].to_vec()).unwrap(),
    );
    let configured = ElectroweakScales {
        distance: PairDistance::Configured,
        ..ElectroweakScales::default()
    };
    let cube = |lower: f64, upper: f64, d: usize| BoxDomain {
        lower: vec![lower; d],
        upper: vec![upper; d],
    };
    // Coordinates above and below the squash radii, a radius per field, a
    // scale per axis and pairs across the seam of a periodic box.
    let squashed = |position_radius, velocity_radius, lambda| Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius,
        velocity_radius,
        lambda,
    };
    let euclidean = |periodic| Distance::Euclidean {
        field: "positions".into(),
        scales: vec![0.5, 2., 4.],
        squared: true,
        periodic,
    };
    let phase_space = |periodic| Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 0.5,
        velocity_scale: 2.,
        lambda: 0.25,
        periodic,
    };
    // Hand values of the pairs (0, 1) and (1, 2); the second crosses the seam.
    let cases = [
        (squashed(0.2, 0.05, 1.), 0.07088357408290313, None),
        (squashed(5., 3., 0.5), 1.456905016138422, None),
        (
            euclidean(Some(cube(-1., 1., 3))),
            2.4125,
            Some(1.2806250000000001),
        ),
        (
            phase_space(Some(cube(-1., 1., 3))),
            6.81046875,
            Some(3.8626562500000006),
        ),
        // Every axis wraps with its own length: Δ = (−0.75, −0.7, 0.8) has the
        // image (−0.75, −0.7, −0.2) in a box of lengths (2, 4, 1).
        (
            euclidean(Some(BoxDomain {
                lower: vec![-1., -2., -0.5],
                upper: vec![1., 2., 0.5],
            })),
            2.375,
            None,
        ),
    ];
    for (distance, first, second) in &cases {
        let amplitude = AmplitudeDistance::new(&gas, distance, &configured).unwrap();
        assert!((amplitude.squared(&state_a, 0, 1).unwrap() - first).abs() < 1e-14);
        if let Some(second) = second {
            assert!((amplitude.squared(&state_a, 1, 2).unwrap() - second).abs() < 1e-14);
        }
        for i in 0..5 {
            for j in 0..5 {
                let engine: f64 = distance
                    .compare(&observations, i, &observations, j)
                    .unwrap();
                let engine = if matches!(distance, Distance::Euclidean { .. }) {
                    engine
                } else {
                    engine * engine
                };
                let own = amplitude.squared(&state_a, i, j).unwrap();
                assert!((own - engine).abs() < 1e-14, "{distance:?} on ({i}, {j})");
            }
        }
    }
    // A box of another dimension is no distance of these coordinates.
    for distance in [
        euclidean(Some(cube(-1., 1., 2))),
        phase_space(Some(cube(-1., 1., 4))),
    ] {
        let amplitude = AmplitudeDistance::new(&gas, &distance, &configured).unwrap();
        assert_eq!(amplitude.squared(&state_a, 0, 1), None);
    }
    // A velocity field the run does not record is unavailable once it has a weight.
    let foreign = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "momenta".into(),
        position_scale: 1.,
        velocity_scale: 1.,
        lambda: 0.25,
        periodic: None,
    };
    assert!(matches!(
        AmplitudeDistance::new(&gas, &foreign, &configured),
        Err(GasError::Capability(_))
    ));
    let weightless = ElectroweakScales {
        lambda: Some(0.),
        ..configured
    };
    let amplitude = AmplitudeDistance::new(&gas, &foreign, &weightless).unwrap();
    assert!((amplitude.squared(&state_a, 0, 1).unwrap() - 1.6925).abs() < 1e-14);
}

#[test]
fn signatures_require_exactly_the_records_they_read_and_fix_the_frame_mean_only_where_stated() {
    use std::collections::BTreeSet;
    let weighted = setup(scales());
    let weightless = setup(ElectroweakScales {
        lambda: Some(0.),
        ..scales()
    });
    let records = |setup: &(GasConfig, MeasurementConfig, Capabilities), spec: &ChannelSpec| {
        let kind = spec.kinds(PairSelection::Both)[0];
        let signature = spec.signature(kind, &context_of(setup)).unwrap();
        assert_eq!(signature.requires.dimension, None);
        signature.requires.records
    };
    let (fitness, plan, velocities, companions) = (
        Record::Fitness,
        Record::ClonePlan,
        Record::Velocities,
        Record::CloningCompanions,
    );
    let role = WalkerRole::WeakResister;
    for (spec, without, with) in [
        (u1(U1Mode::Phase, 3), vec![fitness], vec![fitness]),
        (
            u1(U1Mode::Dressed, 1),
            vec![fitness],
            vec![fitness, velocities],
        ),
        (
            su2(Su2Mode::Phase, true),
            vec![fitness, plan],
            vec![fitness, plan],
        ),
        (
            su2(Su2Mode::Component, false),
            vec![fitness, plan],
            vec![fitness, plan, velocities],
        ),
        (
            su2(Su2Mode::DoubletDiff, false),
            vec![fitness, plan],
            vec![fitness, plan, velocities],
        ),
        (
            ChannelSpec::ElectroweakMixed,
            vec![fitness, plan],
            vec![fitness, plan, velocities],
        ),
        (ChannelSpec::FitnessPhase, vec![fitness], vec![fitness]),
        (ChannelSpec::CloneIndicator, vec![plan], vec![plan]),
        (
            ChannelSpec::ParityVelocity { role },
            vec![velocities, fitness, companions, plan],
            vec![velocities, fitness, companions, plan],
        ),
        (
            chirality(ChiralityObservable::Chi),
            vec![fitness, companions, plan],
            vec![fitness, companions, plan],
        ),
        (
            chirality(ChiralityObservable::LeftFraction),
            vec![fitness, companions, plan],
            vec![fitness, companions, plan],
        ),
    ] {
        assert_eq!(
            records(&weightless, &spec),
            BTreeSet::from_iter(without),
            "{}",
            spec.id()
        );
        assert_eq!(
            records(&weighted, &spec),
            BTreeSet::from_iter(with),
            "{}",
            spec.id()
        );
        // The phases follow the configured frame mean; the book fixes 1/N
        // for the chirality and the clone indicator only.
        let kind = spec.kinds(PairSelection::Both)[0];
        let signature = spec.signature(kind, &context_of(&weighted)).unwrap();
        let fixed = matches!(
            spec,
            ChannelSpec::CloneIndicator | ChannelSpec::Chirality { .. }
        );
        assert_eq!(
            signature.normalization,
            fixed.then_some(FrameNormalization::FixedN),
            "{}",
            spec.id()
        );
    }
}

#[test]
fn a_revival_flagged_on_a_living_row_is_no_cloner_and_targets_nobody() {
    use WalkerRole::{Cloner, Persister, StrongResister, WeakResister};
    let setup = setup(scales());
    let (frame_a, ..) = cycle();
    let scores = state(&frame_a, &setup.0).score.unwrap();
    // Walker 0 points at 2 and is pointed at by the cloner 3. As a revival
    // it stops being a cloner, walker 2 loses its only cloner and falls back
    // on its negative score, and walker 0 becomes the strong resister of 3.
    let mut revived = frame_a.clone();
    revived.revived[0] = true;
    revived.validate().unwrap();
    assert_eq!(
        score::roles(&revived, &scores),
        [StrongResister, WeakResister, Persister, Cloner, Persister]
    );
}

#[test]
fn directed_hops_take_the_absolute_phase_not_the_absolute_imaginary_part() {
    // h_S = 0.17 winds the hop 2 → 1 beyond π: ϑ = −1.65 / (2.15 · 0.17), and
    // e^{i|ϑ|} lies in the lower half plane, where |Im e^{iϑ}| is not.
    let setup = setup(ElectroweakScales {
        h_s: Some(0.17),
        ..scales()
    });
    let context = context_of(&setup);
    let (frame, ..) = cycle();
    let state_a = state(&frame, &setup.0);
    let theta: f64 = (0.45 - 2.1) / ((2.1 + 0.05) * 0.17);
    assert!((theta + 4.51436388508892).abs() < 1e-14);
    assert!(theta.abs() > std::f64::consts::PI && theta.abs().sin() < 0.);
    let pair = [element(ElementKind::CloningPair, 2, 1, 2)];
    let run = |mode, directed| values(&su2(mode, directed), &pair, &state_a, &context)[0].unwrap();
    let plain = run(Su2Mode::Phase, false);
    assert!(close(plain, [theta.cos(), theta.sin()]));
    let directed = run(Su2Mode::Phase, true);
    assert!(close(directed, [-0.19673340652844087, -0.980457019331146]));
    assert!(close(directed, [plain[0], -plain[1]]));
    // Raw D² = 2.7525 + 0.25 · 0.8425 between walkers 2 and 1, range 0.9.
    let r = (-2.963125f64 / (4. * 0.9 * 0.9)).exp();
    assert!(close(
        run(Su2Mode::Component, true),
        [r * theta.abs().cos(), r * theta.abs().sin()]
    ));
    assert!(close(
        run(Su2Mode::Component, true),
        [-0.07883087956914263, -0.39286814871698267]
    ));
    // The second hop 1 → 3 has the positive phase 0.35 / (0.5 · 0.17), which
    // stays as it is, with D² = 3.59625.
    assert!(close(
        run(Su2Mode::Doublet, true),
        [-0.26348966930700124, -0.6658522088048584]
    ));
    assert!(close(
        run(Su2Mode::DoubletDiff, true),
        [0.105827910168716, -0.11988408862910693]
    ));
}

#[test]
fn a_walker_that_died_before_the_sink_masks_every_element_it_enters() {
    let setup = setup(scales());
    let context = context_of(&setup);
    let (frame, ..) = cycle();
    let state_a = state(&frame, &setup.0);
    // The elements are frozen at the source; at the sink walker 2 is dead
    // while its fitness, role and velocity entries are still finite.
    let mut sink = state_a.clone();
    sink.eligible[2] = false;
    assert_eq!(sink.role.as_ref().unwrap()[2], WalkerRole::StrongResister);
    let strong = WalkerRole::StrongResister;
    let weak = WalkerRole::WeakResister;
    let mut out = [7.; 2];
    for (spec, kind, [i, j, k], valid) in [
        (
            u1(U1Mode::Phase, 1),
            ElementKind::DistancePair,
            [1, 2, 1],
            false,
        ),
        (
            u1(U1Mode::Phase, 1),
            ElementKind::DistancePair,
            [2, 3, 2],
            false,
        ),
        (
            u1(U1Mode::Dressed, 2),
            ElementKind::DistancePair,
            [1, 2, 1],
            false,
        ),
        (
            su2(Su2Mode::Phase, false),
            ElementKind::CloningPair,
            [0, 2, 0],
            false,
        ),
        (
            su2(Su2Mode::Phase, true),
            ElementKind::CloningPair,
            [2, 4, 2],
            false,
        ),
        (
            su2(Su2Mode::Component, false),
            ElementKind::CloningPair,
            [0, 2, 0],
            false,
        ),
        // The first hop 3 → 0 lives, the second hop 0 → 2 does not.
        (
            su2(Su2Mode::Doublet, false),
            ElementKind::CloningPair,
            [3, 0, 3],
            false,
        ),
        (
            su2(Su2Mode::DoubletDiff, false),
            ElementKind::CloningPair,
            [3, 0, 3],
            false,
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [0, 1, 2],
            false,
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [1, 2, 3],
            false,
        ),
        (ChannelSpec::FitnessPhase, ElementKind::Site, [2; 3], false),
        (
            ChannelSpec::CloneIndicator,
            ElementKind::Site,
            [2; 3],
            false,
        ),
        (
            ChannelSpec::ParityVelocity { role: strong },
            ElementKind::Site,
            [2; 3],
            false,
        ),
        (
            chirality(ChiralityObservable::Chi),
            ElementKind::Site,
            [2; 3],
            false,
        ),
        (
            chirality(ChiralityObservable::LeftFraction),
            ElementKind::Site,
            [2; 3],
            false,
        ),
    ] {
        let element = element(kind, i, j, k);
        assert!(spec.evaluate(&element, &state_a, &context, &mut [0.; 2]));
        assert_eq!(
            spec.evaluate(&element, &sink, &context, &mut out),
            valid,
            "{} on {:?}",
            spec.id(),
            element.walkers
        );
    }
    assert_eq!(out, [7.; 2]);
    // Elements of the living walkers keep their values.
    for (spec, kind, [i, j, k]) in [
        (u1(U1Mode::Phase, 1), ElementKind::DistancePair, [0, 1, 0]),
        (
            su2(Su2Mode::Component, false),
            ElementKind::CloningPair,
            [3, 0, 3],
        ),
        (
            ChannelSpec::ElectroweakMixed,
            ElementKind::Triplet,
            [4, 0, 1],
        ),
        (
            ChannelSpec::ParityVelocity { role: weak },
            ElementKind::Site,
            [1; 3],
        ),
    ] {
        let element = [element(kind, i, j, k)];
        assert_eq!(
            values(&spec, &element, &sink, &context),
            values(&spec, &element, &state_a, &context)
        );
        assert!(values(&spec, &element, &sink, &context)[0].is_some());
    }
    // A dead row keeps the default role, which is no member of the persisters.
    let eligible = [true, true, true, true, true, false];
    let distance = [Some(1), Some(5), Some(0), Some(4), Some(3), None];
    let cloning = some(&[2, 0, 1, 4, 4, 3]);
    let frame_c = self::frame(
        &eligible,
        &distance,
        &cloning,
        &[true, false, false, false, false, true],
    );
    let state_c = state(&frame_c, &setup.0);
    assert_eq!(state_c.role.as_ref().unwrap()[5], WalkerRole::Persister);
    let role = WalkerRole::Persister;
    let speeds = values(
        &ChannelSpec::ParityVelocity { role },
        &elements(ElementKind::Site, &distance, &cloning),
        &state_c,
        &context,
    );
    let members: Vec<usize> = (0..6).filter(|&i| speeds[i].is_some()).collect();
    assert_eq!(members, [4]);
    assert!((speeds[4].unwrap()[0] - 0.455521678957215).abs() < 1e-14);
}

#[test]
fn score_keeps_the_denominator_of_the_engine_and_the_phase_the_absolute_fitness() {
    // On a negative fitness the two denominators differ: V + ε = −0.25 in the
    // acceptance law of the engine, |V| + ε = 0.35 in the book's phase.
    assert!((score::score(-0.3, 0.5, 0.05) + 3.2).abs() < 1e-14);
    assert!((score::su2_phase(-0.3, 0.5, 0.05, 1.) - 0.8 / 0.35).abs() < 1e-14);
    let mut decision = GasConfig::default().clone_decision;
    decision.epsilon = 0.05;
    decision.saturation = 1.;
    decision.every = 1;
    for (own, donor) in [(-0.3, 0.5), (-0.3, -0.5), (-0.04, 0.5), (1.2, 2.1)] {
        let s = score::score(own, donor, 0.05);
        assert_eq!(
            decision.acceptance_probability(0, own, donor),
            s.clamp(0., 1.)
        );
    }
    assert!((score::score(-0.3, -0.5, 0.05) - 0.8).abs() < 1e-14);
    // On the positive fitness of a record the phase is the score over h_S.
    for own in F {
        for companion in F {
            let phase = score::su2_phase(own, companion, 0.05, 0.35);
            assert!((phase * 0.35 - score::score(own, companion, 0.05)).abs() < 1e-15);
        }
    }
}

#[test]
fn clone_indicator_reads_the_recorded_decisions_and_the_comb_note_needs_a_gated_period() {
    let mut setup = setup(scales());
    let (frame, distance, cloning) = cycle();
    let sites = elements(ElementKind::Site, &distance, &cloning);
    // The indicator requires the clone plan alone: a record without fitness
    // has no score and no roles, and the decisions are still the series.
    let plan_only = FrameState {
        n: 5,
        d: 3,
        x: frame.x.clone(),
        eligible: frame.eligible.clone(),
        cloned: frame.cloned.clone(),
        ..FrameState::default()
    };
    assert_eq!(plan_only.role, None);
    let indicator = ChannelSpec::CloneIndicator;
    let expected = [[1., 0.], [0.; 2], [0.; 2], [1., 0.], [0.; 2]];
    let z = values(&indicator, &sites, &plan_only, &context_of(&setup));
    assert_values(&z, &expected);
    // Roles of another time do not move it either.
    let mut stale = state(&frame, &setup.0);
    stale.role = Some(vec![WalkerRole::Persister; 5]);
    let z = values(&indicator, &sites, &stale, &context_of(&setup));
    assert_values(&z, &expected);
    // Cloning on every step is no comb: the note appears with a period above 1 only.
    let gated = [
        indicator,
        chirality(ChiralityObservable::Chi),
        chirality(ChiralityObservable::LeftFraction),
        ChannelSpec::ParityVelocity {
            role: WalkerRole::Cloner,
        },
        ChannelSpec::ParityVelocity {
            role: WalkerRole::StrongResister,
        },
    ];
    for (every, comb) in [(1, false), (2, true)] {
        setup.0.clone_decision.every = every;
        for spec in &gated {
            let signature = spec
                .signature(ElementKind::Site, &context_of(&setup))
                .unwrap();
            assert_eq!(
                signature.descriptor.note.contains("comb"),
                comb,
                "{} with period {every}",
                spec.id()
            );
        }
    }
}

/// End to end on a recorded run whose cloning companions are matched two by
/// two. `prop-exchange-odd-cancellation` makes the frame mean of
/// `su2/doublet_diff` identically zero for every realization, and the doublet
/// is the component counted twice, so the three are not three observables:
/// the difference reports no rate at all and the doublet carries no
/// information the component does not. The reference implementation fitted all
/// three as independent channels.
#[test]
fn an_involutive_run_declines_the_su2_doublet_difference_and_doubles_the_component() {
    use algorithmic_gas::{
        ExecutionContext, GasBuilder, InputBatch, ObservationBatch, Population, Provenance,
        RewardBatch, TensorBatch,
        domain::{GradientProvider, OperatorFuture, RewardSource},
        donor::SamplingLaw,
        physics::spectroscopy::{
            AnalysisConfig, analyze, contract::EXCHANGE_ODD_REASON, estimators, measure_archive,
            report::EstimatorKind,
        },
    };
    struct Bowl;
    impl RewardSource<f64> for Bowl {
        fn id(&self) -> String {
            "bowl/v1".into()
        }
        fn evaluate<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: Option<&'a InputBatch<f64>>,
            stage: &'a str,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, RewardBatch<f64>> {
            Box::pin(async move {
                let x = p.observations.field("positions")?;
                let reward = (0..p.len())
                    .map(|i| Ok(-x.row(i)?.iter().map(|a| a * a).sum::<f64>()))
                    .collect::<algorithmic_gas::Result<Vec<f64>>>()?;
                Ok(RewardBatch::new(
                    reward,
                    Provenance {
                        population_version: p.version,
                        stage: stage.into(),
                        ..Default::default()
                    },
                ))
            })
        }
    }
    impl GradientProvider<f64> for Bowl {
        fn id(&self) -> String {
            "bowl-gradient/v1".into()
        }
        fn gradient<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, TensorBatch<f64>> {
            Box::pin(async move {
                let x = p.observations.field("positions")?;
                let g = (0..p.len())
                    .map(|i| Ok(x.row(i)?.iter().map(|a| -2. * a).collect::<Vec<f64>>()))
                    .collect::<algorithmic_gas::Result<Vec<Vec<f64>>>>()?
                    .concat();
                TensorBatch::vectors(p.len(), p.observations.field("positions")?.width(), g)
            })
        }
    }
    let archive = futures_lite::future::block_on(async {
        let (n, d) = (8, 3);
        let positions = (0..n * d)
            .map(|k| -1. + 2. * ((k * 11 + 5) % 89) as f64 / 89.)
            .collect();
        let mut observations =
            ObservationBatch::positions(TensorBatch::vectors(n, d, positions).unwrap());
        observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(n, d, vec![0.; n * d]).unwrap(),
        );
        let mut config = GasConfig::euclidean(d, 0.05).unwrap();
        config.seed = 13;
        // Sequential greedy matching pairs the living walkers two by two;
        // a revival draws from the same matched law, not a distance weight.
        config.cloning_donors.law = SamplingLaw::GaussianGreedy;
        config.clone_decision.revival_from_companion = false;
        let mut gas = GasBuilder::new(Population::new(observations).unwrap(), Bowl)
            .config(config)
            .gradient(Bowl)
            .build()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..16 {
            gas.step().await.unwrap();
        }
        gas.recording().unwrap().clone()
    });
    let measurement = MeasurementConfig {
        warmup: 2,
        max_lag: 4,
        electroweak: ElectroweakScales {
            epsilon_c: Range::Fixed { value: 0.9 },
            epsilon_d: Range::Fixed { value: 1.3 },
            ..ElectroweakScales::default()
        },
        channels: vec![
            su2(Su2Mode::DoubletDiff, false),
            su2(Su2Mode::Doublet, false),
            su2(Su2Mode::Component, false),
        ],
        ..MeasurementConfig::default()
    };
    let capabilities =
        Capabilities::of(&archive.gas_config, &archive.config, 3).refine(&measurement);
    assert!(capabilities.mutual_cloning);
    let measured = measure_archive(&measurement, &archive).unwrap();
    let frames = measured.frames();
    assert!(frames >= 8);
    let series = |id: &str| measured.channel(id).unwrap();
    let (diff, doublet, component) = (
        series("su2/doublet_diff/cloning"),
        series("su2/doublet/cloning"),
        series("su2/component/cloning"),
    );
    // The difference has no frame with an unmirrored element: no weight, and
    // therefore no value, on any measured frame.
    assert_eq!(diff.weight, vec![0.; frames]);
    assert_eq!(diff.exchange, ExchangeParity::Odd);
    assert!(diff.involutive_frames > 0);
    // The doublet and the component are measured, and the doublet frame mean
    // is the component's counted twice.
    assert_eq!(doublet.weight, component.weight);
    assert!(doublet.weight.iter().any(|w| *w > 0.));
    for (a, b) in doublet.values.iter().zip(&component.values) {
        assert!(
            (a - 2. * b).abs() <= 1e-13 * (2. * b).abs() + 1e-15,
            "{a} {b}"
        );
    }
    // The estimator declines the difference with the algebraic reason, and the
    // analysis reports it unavailable with no correlator and no rate. No
    // propagator was measured for it, so nothing replaces the frame mean.
    let estimated = estimators::estimate(
        std::slice::from_ref(&measured),
        "su2/doublet_diff/cloning",
        EstimatorKind::FrameMean,
        &AnalysisConfig::default(),
        None,
    );
    assert!(
        matches!(&estimated, Err(GasError::Capability(reason)) if reason == EXCHANGE_ODD_REASON),
        "{estimated:?}"
    );
    let report = analyze(&[measured], &AnalysisConfig::default()).unwrap();
    let declined = report.channel("su2/doublet_diff/cloning").unwrap();
    assert_eq!(declined.availability.reason(), Some(EXCHANGE_ODD_REASON));
    assert!(declined.mass.is_none() && declined.correlator.is_none());
    assert!(declined.estimator.is_none());
    let fitted = report.channel("su2/component/cloning").unwrap();
    assert_eq!(fitted.estimator, Some(EstimatorKind::FrameMean));
    assert!(fitted.correlator.is_some());
}
