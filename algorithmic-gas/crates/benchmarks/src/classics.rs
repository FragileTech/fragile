//! Classic objectives added for parity with the Optimization Lab: scalar f64 values and
//! tensor graphs (value and analytic gradient) that run on every compute backend.
use crate::{benchmark::Benchmark, mixture::Mixture};
use algorithmic_gas::compute::{Binary, Expression, Node, Unary};
use std::f64::consts::PI;

/// Squared pair distances below this are clamped so the energy stays finite in f32.
pub const LENNARD_JONES_R2_FLOOR: f64 = 1e-4;
const CUSP: f64 = 1e-6;
const SIGN_FLOOR: f64 = 1e-12;
const CEILING: f64 = 1e30;

/// A constant matrix passed to the graph as an extra input.
#[derive(Clone, Debug, PartialEq)]
pub struct Constant {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f64>,
}

/// Input 0 is the `[walkers, d]` coordinate batch; inputs `1..` are `constants`.
#[derive(Clone, Debug)]
pub struct ObjectiveGraph {
    pub expression: Expression,
    pub constants: Vec<Constant>,
}

pub fn value(benchmark: Benchmark, mixture: Option<&Mixture>, x: &[f64]) -> f64 {
    let squares = || x.iter().fold(0., |s, v| s + v * v);
    match benchmark {
        Benchmark::QuadraticWell { alpha } => 0.5 * alpha * squares(),
        Benchmark::MexicanHat {
            lambda_h,
            vev,
            field_scale,
            tilt,
        } => {
            let radius = vev / field_scale;
            0.25 * lambda_h * (squares() - radius * radius).powi(2) - tilt * x[0]
        }
        Benchmark::Eggholder => {
            -(x[1] + 47.) * (x[0] / 2. + x[1] + 47.).abs().sqrt().sin()
                - x[0] * (x[0] - x[1] - 47.).abs().sqrt().sin()
        }
        Benchmark::Easom => {
            -x[0].cos() * x[1].cos() * (-(x[0] - PI).powi(2) - (x[1] - PI).powi(2)).exp()
        }
        Benchmark::HolderTable => {
            -(x[0].sin() * x[1].cos() * (1. - x[0].hypot(x[1]) / PI).abs().exp()).abs()
        }
        Benchmark::LennardJones { .. } => {
            let mut energy = 0.;
            for a in (0..x.len()).step_by(3) {
                for b in (0..a).step_by(3) {
                    let r2 = (0..3).fold(0., |s, k| s + (x[a + k] - x[b + k]).powi(2));
                    let r6 = 1. / r2.max(LENNARD_JONES_R2_FLOOR).powi(3);
                    energy += 4. * r6 * (r6 - 1.);
                }
            }
            energy
        }
        Benchmark::Constant | Benchmark::StochasticGaussian { .. } => 0.,
        Benchmark::GaussianMixture { .. } => mixture.expect("mixture components").value(x),
        _ => unreachable!("not a parity classic"),
    }
}

struct Graph {
    e: Expression,
}
impl Graph {
    fn scalar(&mut self, v: f64) -> usize {
        self.e.scalar(v)
    }
    fn un(&mut self, op: Unary, x: usize) -> usize {
        self.e.unary(op, x)
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        self.e.binary(Binary::Add, a, b)
    }
    fn sub(&mut self, a: usize, b: usize) -> usize {
        self.e.binary(Binary::Subtract, a, b)
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        self.e.binary(Binary::Multiply, a, b)
    }
    fn div(&mut self, a: usize, b: usize) -> usize {
        self.e.binary(Binary::Divide, a, b)
    }
    fn scale(&mut self, factor: f64, x: usize) -> usize {
        let c = self.scalar(factor);
        self.mul(c, x)
    }
    fn offset(&mut self, x: usize, value: f64) -> usize {
        let c = self.scalar(value);
        self.add(x, c)
    }
    fn columns(&mut self, source: usize, start: usize, end: usize) -> usize {
        self.e.push(Node::Columns { source, start, end })
    }
    fn concat(&mut self, parts: Vec<usize>) -> usize {
        self.e.push(Node::ConcatColumns(parts))
    }
    fn row(&mut self, source: usize, index: usize) -> usize {
        self.e.push(Node::Gather {
            source,
            indices: vec![index as u32],
        })
    }
    /// `x / max(|x|, floor)`: the sign away from zero, a finite ramp through it.
    fn sign(&mut self, x: usize) -> usize {
        let magnitude = self.un(Unary::Abs, x);
        let safe = self.un(Unary::Clamp(SIGN_FLOOR, CEILING), magnitude);
        self.div(x, safe)
    }
    /// Element-wise maximum: `(a + b + |a - b|) / 2`.
    fn max(&mut self, a: usize, b: usize) -> usize {
        let sum = self.add(a, b);
        let difference = self.sub(a, b);
        let distance = self.un(Unary::Abs, difference);
        let total = self.add(sum, distance);
        self.scale(0.5, total)
    }
}

/// Column-pair difference matrix `[3n, 3P]` laid out as `[dx | dy | dz]`, and its transpose.
fn pair_matrices(atoms: usize) -> (Constant, Constant) {
    let pairs = atoms * (atoms - 1) / 2;
    let (rows, columns) = (3 * atoms, 3 * pairs);
    let mut forward = vec![0.; rows * columns];
    let mut backward = vec![0.; rows * columns];
    let mut pair = 0;
    for a in 0..atoms {
        for b in 0..a {
            for k in 0..3 {
                let column = k * pairs + pair;
                for (atom, sign) in [(a, 1.), (b, -1.)] {
                    forward[(3 * atom + k) * columns + column] = sign;
                    backward[column * rows + 3 * atom + k] = sign;
                }
            }
            pair += 1;
        }
    }
    (
        Constant {
            rows,
            columns,
            values: forward,
        },
        Constant {
            rows: columns,
            columns: rows,
            values: backward,
        },
    )
}

pub fn graph(
    benchmark: Benchmark,
    mixture: Option<&Mixture>,
    d: usize,
    gradient: bool,
) -> ObjectiveGraph {
    let mut g = Graph {
        e: Expression::default(),
    };
    let mut constants = vec![];
    let x = g.e.input(0);
    match benchmark {
        Benchmark::QuadraticWell { alpha } => {
            if gradient {
                g.scale(alpha, x);
            } else {
                let squares = g.un(Unary::Square, x);
                let sum = g.un(Unary::SumRows, squares);
                g.scale(0.5 * alpha, sum);
            }
        }
        Benchmark::MexicanHat {
            lambda_h,
            vev,
            field_scale,
            tilt,
        } => {
            let radius = vev / field_scale;
            let squares = g.un(Unary::Square, x);
            let r2 = g.un(Unary::SumRows, squares);
            let u = g.offset(r2, -radius * radius);
            let first = g.columns(x, 0, 1);
            if gradient {
                let radial = g.mul(u, x);
                let radial = g.scale(lambda_h, radial);
                let mut direction = vec![g.scale(0., first)];
                direction[0] = g.offset(direction[0], tilt);
                if d > 1 {
                    let rest = g.columns(x, 1, d);
                    direction.push(g.scale(0., rest));
                }
                let direction = if direction.len() == 1 {
                    direction[0]
                } else {
                    g.concat(direction)
                };
                g.sub(radial, direction);
            } else {
                let quartic = g.un(Unary::Square, u);
                let quartic = g.scale(0.25 * lambda_h, quartic);
                let slope = g.scale(tilt, first);
                g.sub(quartic, slope);
            }
        }
        Benchmark::Eggholder => {
            let (x0, x1) = (g.columns(x, 0, 1), g.columns(x, 1, 2));
            let shifted = g.offset(x1, 47.);
            let half = g.scale(0.5, x0);
            let a = g.add(half, shifted);
            let b = g.sub(x0, shifted);
            let (abs_a, abs_b) = (g.un(Unary::Abs, a), g.un(Unary::Abs, b));
            let (root_a, root_b) = (g.un(Unary::Sqrt, abs_a), g.un(Unary::Sqrt, abs_b));
            let (sin_a, sin_b) = (g.un(Unary::Sin, root_a), g.un(Unary::Sin, root_b));
            if gradient {
                let slope = |g: &mut Graph, value: usize, root: usize| {
                    let cosine = g.un(Unary::Cos, root);
                    let sign = g.sign(value);
                    let safe = g.un(Unary::Clamp(CUSP, CEILING), root);
                    let numerator = g.mul(cosine, sign);
                    let denominator = g.scale(2., safe);
                    g.div(numerator, denominator)
                };
                let da = slope(&mut g, a, root_a);
                let db = slope(&mut g, b, root_b);
                let lift = g.mul(shifted, da);
                let drag = g.mul(x0, db);
                let half_lift = g.scale(0.5, lift);
                let g0 = g.add(half_lift, sin_b);
                let g0 = g.add(g0, drag);
                let g0 = g.un(Unary::Neg, g0);
                let g1 = g.add(sin_a, lift);
                let g1 = g.sub(drag, g1);
                g.concat(vec![g0, g1]);
            } else {
                let first = g.mul(shifted, sin_a);
                let second = g.mul(x0, sin_b);
                let sum = g.add(first, second);
                g.un(Unary::Neg, sum);
            }
        }
        Benchmark::Easom => {
            let (x0, x1) = (g.columns(x, 0, 1), g.columns(x, 1, 2));
            let (c0, c1) = (g.un(Unary::Cos, x0), g.un(Unary::Cos, x1));
            let (p0, p1) = (g.offset(x0, -PI), g.offset(x1, -PI));
            let (q0, q1) = (g.un(Unary::Square, p0), g.un(Unary::Square, p1));
            let exponent = g.add(q0, q1);
            let exponent = g.un(Unary::Neg, exponent);
            let bump = g.un(Unary::Exp, exponent);
            if gradient {
                let partial = |g: &mut Graph, own: usize, centred: usize, other_cos: usize| {
                    let sine = g.un(Unary::Sin, own);
                    let cosine = g.un(Unary::Cos, own);
                    let pull = g.mul(centred, cosine);
                    let pull = g.scale(2., pull);
                    let inner = g.add(sine, pull);
                    let outer = g.mul(bump, other_cos);
                    g.mul(outer, inner)
                };
                let g0 = partial(&mut g, x0, p0, c1);
                let g1 = partial(&mut g, x1, p1, c0);
                g.concat(vec![g0, g1]);
            } else {
                let product = g.mul(c0, c1);
                let product = g.mul(product, bump);
                g.un(Unary::Neg, product);
            }
        }
        Benchmark::HolderTable => {
            let (x0, x1) = (g.columns(x, 0, 1), g.columns(x, 1, 2));
            let squares = g.un(Unary::Square, x);
            let r2 = g.un(Unary::SumRows, squares);
            let r = g.un(Unary::Sqrt, r2);
            let scaled = g.scale(-1. / PI, r);
            let h = g.offset(scaled, 1.);
            let abs_h = g.un(Unary::Abs, h);
            let e = g.un(Unary::Exp, abs_h);
            let (s0, c1) = (g.un(Unary::Sin, x0), g.un(Unary::Cos, x1));
            let wave = g.mul(s0, c1);
            let f = g.mul(wave, e);
            if gradient {
                let sign_h = g.sign(h);
                let safe_r = g.un(Unary::Clamp(CUSP, CEILING), r);
                let denominator = g.scale(-PI, safe_r);
                let q = g.div(sign_h, denominator);
                let sign_f = g.sign(f);
                let outer = g.un(Unary::Neg, sign_f);
                let fq = g.mul(f, q);
                let (c0, s1) = (g.un(Unary::Cos, x0), g.un(Unary::Sin, x1));
                let along0 = g.mul(c0, c1);
                let along0 = g.mul(along0, e);
                let radial0 = g.mul(fq, x0);
                let g0 = g.add(along0, radial0);
                let g0 = g.mul(outer, g0);
                let along1 = g.mul(s0, s1);
                let along1 = g.mul(along1, e);
                let radial1 = g.mul(fq, x1);
                let g1 = g.sub(radial1, along1);
                let g1 = g.mul(outer, g1);
                g.concat(vec![g0, g1]);
            } else {
                let magnitude = g.un(Unary::Abs, f);
                g.un(Unary::Neg, magnitude);
            }
        }
        Benchmark::LennardJones { n_atoms } => {
            let atoms = n_atoms as usize;
            let pairs = atoms * (atoms - 1) / 2;
            let (forward, backward) = pair_matrices(atoms);
            constants.push(forward);
            let m = g.e.input(1);
            let differences = g.e.binary(Binary::Matmul, x, m);
            let parts: Vec<usize> = (0..3)
                .map(|k| g.columns(differences, k * pairs, (k + 1) * pairs))
                .collect();
            let squares: Vec<usize> = parts.iter().map(|&p| g.un(Unary::Square, p)).collect();
            let r2 = g.add(squares[0], squares[1]);
            let r2 = g.add(r2, squares[2]);
            let r2 = g.un(Unary::Clamp(LENNARD_JONES_R2_FLOOR, CEILING), r2);
            let r6 = g.un(Unary::Power(-3.), r2);
            if gradient {
                constants.push(backward);
                let transpose = g.e.input(2);
                let doubled = g.scale(-2., r6);
                let bracket = g.offset(doubled, 1.);
                let force = g.mul(r6, bracket);
                let force = g.scale(24., force);
                let force = g.div(force, r2);
                let weighted: Vec<usize> = parts.iter().map(|&p| g.mul(force, p)).collect();
                let weighted = g.concat(weighted);
                g.e.binary(Binary::Matmul, weighted, transpose);
            } else {
                let bracket = g.offset(r6, -1.);
                let pair_energy = g.mul(r6, bracket);
                let pair_energy = g.scale(4., pair_energy);
                g.un(Unary::SumRows, pair_energy);
            }
        }
        Benchmark::Constant | Benchmark::StochasticGaussian { .. } => {
            let zeros = g.scale(0., x);
            if !gradient {
                g.un(Unary::SumRows, zeros);
            }
        }
        Benchmark::GaussianMixture { .. } => {
            let mixture = mixture.expect("mixture components");
            let count = mixture.components;
            let inverse: Vec<f64> = mixture.stds.iter().map(|s| 1. / s).collect();
            let offsets: Vec<f64> = (0..count)
                .map(|c| {
                    let log_std: f64 = mixture.stds[c * d..(c + 1) * d]
                        .iter()
                        .map(|s| s.ln())
                        .sum();
                    mixture.weights[c].ln() - 0.5 * d as f64 * (2. * PI).ln() - log_std
                })
                .collect();
            constants.push(Constant {
                rows: count,
                columns: d,
                values: mixture.centers.clone(),
            });
            constants.push(Constant {
                rows: count,
                columns: d,
                values: inverse,
            });
            constants.push(Constant {
                rows: 1,
                columns: count,
                values: offsets,
            });
            let (centers, inverse, offsets) = (g.e.input(1), g.e.input(2), g.e.input(3));
            let mut scaled = vec![];
            let mut quadratic = vec![];
            for c in 0..count {
                let center = g.row(centers, c);
                let weight = g.row(inverse, c);
                let difference = g.sub(x, center);
                let z = g.mul(difference, weight);
                let z2 = g.un(Unary::Square, z);
                quadratic.push(g.un(Unary::SumRows, z2));
                if gradient {
                    // d(z²/2)/dx = z / std.
                    scaled.push(g.mul(z, weight));
                }
            }
            let quadratic = if count == 1 {
                quadratic[0]
            } else {
                g.concat(quadratic)
            };
            let half = g.scale(0.5, quadratic);
            let logs = g.sub(offsets, half);
            let mut maximum = g.columns(logs, 0, 1);
            for c in 1..count {
                let column = g.columns(logs, c, c + 1);
                maximum = g.max(maximum, column);
            }
            let centred = g.sub(logs, maximum);
            let weights = g.un(Unary::Exp, centred);
            let total = g.un(Unary::SumRows, weights);
            if gradient {
                let responsibilities = g.div(weights, total);
                let mut sum = None;
                for (c, &term) in scaled.iter().enumerate() {
                    let share = g.columns(responsibilities, c, c + 1);
                    let term = g.mul(share, term);
                    sum = Some(match sum {
                        Some(previous) => g.add(previous, term),
                        None => term,
                    });
                }
            } else {
                let log_total = g.un(Unary::Log, total);
                let log_density = g.add(maximum, log_total);
                g.un(Unary::Neg, log_density);
            }
        }
        _ => unreachable!("not a parity classic"),
    }
    ObjectiveGraph {
        expression: g.e,
        constants,
    }
}
