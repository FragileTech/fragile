//! Separate causal-slot, selected-interaction, information and material-lineage graphs.
use crate::{Population, Real, donor::SourceRef, tracking::RunArchive};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventRef {
    pub epoch: u64,
    pub step: u64,
    pub slot: u32,
    pub generation: u64,
    pub version: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Cst,
    IgDistance,
    IgCloning,
    HistoricalDistance,
    HistoricalCloning,
    IaDistance,
    IaCloning,
    IaRevival,
    IaTransform,
    Ancestry,
    Persistence,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EdgeAttributes {
    pub fitness_difference: Option<f64>,
    /// Configured joint sampler law; no marginal selection probability is invented.
    pub selection_law: Option<crate::donor::SamplingLaw>,
    pub selection_probability: Option<f64>,
    pub weight: Option<f64>,
    pub weight_provenance: Option<String>,
    /// Supplied connections can decorate topology; missing never means zero phase.
    pub u1_connection: Option<[f64; 2]>,
    pub su2_connection: Option<[f64; 4]>,
    /// Raw recorded chart displacement; periodic winding is not inferred.
    pub position_displacement: Option<[f64; 2]>,
    /// Principal Spin(2) square-root encoding of the raw displacement.
    pub spin2_displacement: Option<[f64; 2]>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub source: EventRef,
    pub target: EventRef,
    pub kind: EdgeKind,
    pub revival: bool,
    pub attributes: EdgeAttributes,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MissingSource {
    pub epoch: u64,
    pub source: SourceRef,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionTriangle {
    pub vertices: [EventRef; 3],
    /// Boundary is CST + IA - IG; entries are indices in FractalSet.edges.
    pub boundary_edges: [usize; 3],
    pub channels: Vec<EdgeKind>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FractalSet {
    pub nodes: Vec<EventRef>,
    pub edges: Vec<Edge>,
    pub triangles: Vec<InteractionTriangle>,
    /// Sources predating recording coverage are explicit, never replaced by current slots.
    pub unresolved_sources: Vec<MissingSource>,
}
fn events<T: Real>(epoch: u64, step: u64, p: &Population<T>) -> Vec<EventRef> {
    (0..p.len())
        .map(|slot| EventRef {
            epoch,
            step,
            slot: slot as u32,
            generation: p.generations[slot],
            version: p.version,
        })
        .collect()
}
fn position<T: Real>(p: &Population<T>, slot: usize) -> Option<[f64; 2]> {
    let field = p
        .observations
        .fields
        .get("positions")
        .filter(|f| f.width() == 2)?;
    let row = field.row(slot).ok()?;
    let point = [row[0].to_f64(), row[1].to_f64()];
    point.iter().all(|x| x.is_finite()).then_some(point)
}

/// Principal complex square root without subtracting nearly equal components.
/// The small component is recovered from 2uv = y, preserving near-axis vectors.
fn spin2([x, y]: [f64; 2]) -> [f64; 2] {
    let scale = x.abs().max(y.abs());
    if scale == 0. {
        return [0., 0.];
    }
    let radius = (x / scale).hypot(y / scale);
    let large = scale.sqrt() * ((radius + x.abs() / scale) * 0.5).sqrt();
    if x >= 0. {
        [large, y / (2. * large)]
    } else {
        let imaginary = if y < 0. { -large } else { large };
        [y / (2. * imaginary), imaginary]
    }
}

fn displacement(a: Option<[f64; 2]>, b: Option<[f64; 2]>) -> Option<[f64; 2]> {
    let (a, b) = (a?, b?);
    let delta = [b[0] - a[0], b[1] - a[1]];
    delta.iter().all(|x| x.is_finite()).then_some(delta)
}

impl FractalSet {
    pub fn from_archive<T: Real>(archive: &RunArchive<T>) -> Self {
        let mut nodes = BTreeSet::new();
        let mut lookup = BTreeMap::new();
        let mut positions = BTreeMap::new();
        for anchor in &archive.anchors {
            nodes.extend(events(anchor.epoch, anchor.step, &anchor.population));
        }
        for s in &archive.steps {
            for e in events(s.epoch, s.report.step - 1, &s.before) {
                lookup.insert((e.epoch, e.step, e.slot, e.generation, e.version), e);
                if let Some(point) = position(&s.before, e.slot as usize) {
                    positions.insert(e, point);
                }
                nodes.insert(e);
            }
            nodes.extend(events(s.epoch, s.report.step, &s.final_population));
        }
        let mut result = Self {
            nodes: nodes.into_iter().collect(),
            ..Self::default()
        };
        let mut triangle_indices = BTreeMap::<[EventRef; 3], usize>::new();
        for s in &archive.steps {
            let before = events(s.epoch, s.report.step - 1, &s.before);
            let after = events(s.epoch, s.report.step, &s.final_population);
            let mut resolve = |source: SourceRef| -> Option<EventRef> {
                let found = lookup
                    .get(&(
                        s.epoch,
                        source.frame,
                        source.slot,
                        source.generation,
                        source.version,
                    ))
                    .copied();
                if found.is_none()
                    && !result
                        .unresolved_sources
                        .iter()
                        .any(|m| m.epoch == s.epoch && m.source == source)
                {
                    result.unresolved_sources.push(MissingSource {
                        epoch: s.epoch,
                        source,
                    });
                }
                found
            };
            for i in 0..before.len() {
                let cst_index = result.edges.len();
                if s.report.pre_clone_eligible[i] {
                    let displacement =
                        displacement(position(&s.before, i), position(&s.final_population, i));
                    let spinor = displacement.map(spin2);
                    result.edges.push(Edge {
                        source: before[i],
                        target: after[i],
                        kind: EdgeKind::Cst,
                        revival: false,
                        attributes: EdgeAttributes {
                            position_displacement: displacement,
                            spin2_displacement: spinor,
                            ..Default::default()
                        },
                    });
                }
                let choice = &s.report.clone_plan.choices[i];
                if choice.accepted {
                    if let Some(donor) = choice
                        .donors
                        .first()
                        .and_then(|d| resolve(s.report.clone_plan.sources[d.pool_index as usize]))
                    {
                        result.edges.push(Edge {
                            source: donor,
                            target: after[i],
                            kind: EdgeKind::Ancestry,
                            revival: choice.revival,
                            attributes: EdgeAttributes {
                                weight: Some(1.),
                                weight_provenance: Some("accepted literal copy".into()),
                                ..Default::default()
                            },
                        });
                        if choice.revival {
                            result.edges.push(Edge {
                                source: after[i],
                                target: donor,
                                kind: EdgeKind::IaRevival,
                                revival: true,
                                attributes: EdgeAttributes {
                                    weight: Some(1.),
                                    weight_provenance: Some(
                                        "actual uniform current-eligible revival donor".into(),
                                    ),
                                    ..Default::default()
                                },
                            });
                        }
                    }
                } else {
                    result.edges.push(Edge {
                        source: before[i],
                        target: after[i],
                        kind: EdgeKind::Persistence,
                        revival: false,
                        attributes: EdgeAttributes::default(),
                    });
                }
                for (companions, sources, ig, ia) in [
                    (
                        &s.report.distance_companions,
                        &s.report.distance_sources,
                        EdgeKind::IgDistance,
                        EdgeKind::IaDistance,
                    ),
                    (
                        &s.report.cloning_companions,
                        &s.report.clone_plan.sources,
                        EdgeKind::IgCloning,
                        EdgeKind::IaCloning,
                    ),
                ] {
                    for index in companions.row(i) {
                        if let Some(donor) = resolve(sources[index as usize]) {
                            // Self draws remain in the report; they do not create degenerate cells.
                            if donor == before[i] {
                                continue;
                            }
                            let ig_index = result.edges.len();
                            let same_frame = donor.step == before[i].step;
                            let donor_fitness = if same_frame {
                                Some(
                                    s.report.pre_clone_fitness.fitness[donor.slot as usize]
                                        .to_f64(),
                                )
                            } else if ig == EdgeKind::IgCloning {
                                Some(s.donor_fitness[index as usize].to_f64())
                            } else {
                                None
                            };
                            let attributes = EdgeAttributes {
                                fitness_difference: donor_fitness
                                    .map(|f| f - s.report.pre_clone_fitness.fitness[i].to_f64()),
                                selection_law: Some(if ig == EdgeKind::IgDistance {
                                    archive.gas_config.distance_donors.law
                                } else {
                                    archive.gas_config.cloning_donors.law
                                }),
                                ..Default::default()
                            };
                            let delta = displacement(
                                position(&s.before, i),
                                positions.get(&donor).copied(),
                            );
                            result.edges.push(Edge {
                                source: before[i],
                                target: donor,
                                kind: if same_frame {
                                    ig
                                } else if ig == EdgeKind::IgDistance {
                                    EdgeKind::HistoricalDistance
                                } else {
                                    EdgeKind::HistoricalCloning
                                },
                                revival: false,
                                attributes: EdgeAttributes {
                                    position_displacement: delta,
                                    spin2_displacement: delta.map(spin2),
                                    ..attributes.clone()
                                },
                            });
                            result.edges.push(Edge {
                                source: after[i],
                                target: donor,
                                kind: ia,
                                revival: false,
                                attributes: attributes.clone(),
                            });
                            if same_frame && s.report.pre_clone_eligible[i] {
                                let vertices = [before[i], after[i], donor];
                                if let Some(&index) = triangle_indices.get(&vertices) {
                                    let triangle = &mut result.triangles[index];
                                    if !triangle.channels.contains(&ig) {
                                        triangle.channels.push(ig);
                                    }
                                } else {
                                    triangle_indices.insert(vertices, result.triangles.len());
                                    result.triangles.push(InteractionTriangle {
                                        vertices,
                                        boundary_edges: [cst_index, ig_index + 1, ig_index],
                                        channels: vec![ig],
                                    });
                                }
                            }
                        }
                    }
                }
            }
            for influence in &s.influences {
                if let Some(donor) = resolve(influence.source) {
                    result.edges.push(Edge {
                        source: after[influence.recipient as usize],
                        target: donor,
                        kind: EdgeKind::IaTransform,
                        revival: false,
                        attributes: EdgeAttributes {
                            weight: Some(influence.weight),
                            weight_provenance: Some(format!(
                                "{}: {} affine coefficient",
                                influence.stage, influence.field
                            )),
                            ..Default::default()
                        },
                    });
                }
            }
        }
        result
    }
    pub fn boundary_squared_zero(&self) -> bool {
        self.triangles.iter().all(|triangle| {
            let mut coefficients = BTreeMap::<EventRef, i32>::new();
            for (index, sign) in triangle.boundary_edges.iter().zip([1, 1, -1]) {
                let Some(edge) = self.edges.get(*index) else {
                    return false;
                };
                *coefficients.entry(edge.source).or_default() -= sign;
                *coefficients.entry(edge.target).or_default() += sign;
            }
            coefficients.values().all(|&x| x == 0)
        })
    }
    /// Strict causal order uses CST only. Genealogy and IA cannot change it.
    pub fn causal_future(&self, start: EventRef) -> Vec<EventRef> {
        self.reachable(start, &[EdgeKind::Cst])
    }
    pub fn descendants(&self, start: EventRef) -> Vec<EventRef> {
        self.reachable(start, &[EdgeKind::Ancestry, EdgeKind::Persistence])
    }
    fn reachable(&self, start: EventRef, kinds: &[EdgeKind]) -> Vec<EventRef> {
        let mut seen = BTreeSet::new();
        let mut adjacency = BTreeMap::<EventRef, Vec<EventRef>>::new();
        for edge in self.edges.iter().filter(|e| kinds.contains(&e.kind)) {
            adjacency.entry(edge.source).or_default().push(edge.target);
        }
        let mut queue = vec![start];
        while let Some(current) = queue.pop() {
            if let Some(targets) = adjacency.get(&current) {
                for &target in targets {
                    if seen.insert(target) {
                        queue.push(target);
                    }
                }
            }
        }
        seen.into_iter().collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderNode {
    pub event: EventRef,
    pub position: [f64; 2],
    pub time: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderComparison {
    pub nodes: Vec<OrderNode>,
    pub cst_pairs: Vec<[usize; 2]>,
    pub lorentz_pairs: Vec<[usize; 2]>,
    pub both: usize,
    pub cst_only: usize,
    pub lorentz_only: usize,
    pub speed: f64,
    pub dt: f64,
    pub sampling: String,
}
impl<T: Real> RunArchive<T> {
    /// Compare recorded CST transitive closure with physical 2+1 light cones.
    /// Display time exaggeration is deliberately absent from this computation.
    pub fn compare_orders(
        &self,
        speed: f64,
        dt: f64,
        max_nodes: usize,
    ) -> crate::Result<OrderComparison> {
        crate::error::require(
            speed.is_finite()
                && speed > 0.
                && dt.is_finite()
                && dt > 0.
                && max_nodes > 0
                && max_nodes <= 512,
            "order comparison requires positive speed/time step and 1..512 nodes",
        )?;
        let terminal_step = self.terminal().0;
        crate::error::require(
            (terminal_step as f64 * dt).is_finite()
                && (terminal_step as f64 * dt * speed).is_finite(),
            "order comparison physical units overflow",
        )?;
        let graph = self.graph();
        let mut coordinates = BTreeMap::<EventRef, [f64; 2]>::new();
        let mut insert = |epoch: u64, step: u64, p: &Population<T>| {
            if let Some(field) = p
                .observations
                .fields
                .get("positions")
                .filter(|f| f.width() == 2)
            {
                for event in events(epoch, step, p) {
                    let row = field
                        .row(event.slot as usize)
                        .expect("validated population rows");
                    let point = [row[0].to_f64(), row[1].to_f64()];
                    if point.iter().all(|x| x.is_finite()) {
                        coordinates.insert(event, point);
                    }
                }
            }
        };
        for a in &self.anchors {
            insert(a.epoch, a.step, &a.population);
        }
        for s in &self.steps {
            insert(s.epoch, s.report.step - 1, &s.before);
            insert(s.epoch, s.report.step, &s.final_population);
        }
        let mut nodes = coordinates
            .into_iter()
            .rev()
            .take(max_nodes)
            .map(|(event, position)| OrderNode {
                event,
                position,
                time: event.step as f64 * dt,
            })
            .collect::<Vec<_>>();
        nodes.reverse();
        let mut adjacency = BTreeMap::<EventRef, Vec<EventRef>>::new();
        for e in graph.edges.iter().filter(|e| e.kind == EdgeKind::Cst) {
            adjacency.entry(e.source).or_default().push(e.target);
        }
        let mut result = OrderComparison {
            nodes,
            cst_pairs: vec![],
            lorentz_pairs: vec![],
            both: 0,
            cst_only: 0,
            lorentz_only: 0,
            speed,
            dt,
            sampling:
                "most recent finite planar recorded events; CST closure uses the full archive"
                    .into(),
        };
        for i in 0..result.nodes.len() {
            let a = &result.nodes[i];
            let mut reachable = BTreeSet::new();
            let mut queue = vec![a.event];
            while let Some(event) = queue.pop() {
                if let Some(next) = adjacency.get(&event) {
                    for &target in next {
                        if reachable.insert(target) {
                            queue.push(target);
                        }
                    }
                }
            }
            for j in i + 1..result.nodes.len() {
                let b = &result.nodes[j];
                if a.event.epoch != b.event.epoch {
                    continue;
                }
                let cst = reachable.contains(&b.event);
                let elapsed = (b.event.step - a.event.step) as f64 * dt;
                let distance = (b.position[0] - a.position[0]).hypot(b.position[1] - a.position[1]);
                let lorentz = elapsed > 0. && distance <= speed * elapsed;
                if cst {
                    result.cst_pairs.push([i, j]);
                }
                if lorentz {
                    result.lorentz_pairs.push([i, j]);
                }
                match (cst, lorentz) {
                    (true, true) => result.both += 1,
                    (true, false) => result.cst_only += 1,
                    (false, true) => result.lorentz_only += 1,
                    _ => {}
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::spin2;

    #[test]
    fn spin2_preserves_small_transverse_components_and_large_finite_vectors() {
        for [x, y] in [
            [1., 1e-9],
            [-1., 1e-9],
            [1., -1e-9],
            [-1., -1e-9],
            [1e308, 1e299],
            [-1e308, -1e299],
            [1e-300, 1e-309],
        ] {
            let [u, v] = spin2([x, y]);
            assert!(u.is_finite() && v.is_finite());
            assert!(((u * u - v * v) / x - 1.).abs() < 1e-14);
            assert!((2. * u * v / y - 1.).abs() < 1e-14);
        }
        let [u, v] = spin2([f64::MAX, f64::MAX]);
        assert!(u.is_finite() && v.is_finite());
        assert_eq!(spin2([0., 0.]), [0., 0.]);
        assert_eq!(spin2([-1., -0.]), [0., 1.]);
    }
}
