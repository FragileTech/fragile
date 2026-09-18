//! Incremental Bowyer-Watson Delaunay tetrahedralization with exact predicates.
//!
//! The convex hull is closed by ghost tetrahedra that share a hull face with a
//! vertex at infinity, so every site has a conflict region and the structure is
//! a triangulated 3-sphere throughout. Orientation convention: a finite
//! tetrahedron satisfies `orient3d(v0, v1, v2, v3) > 0`; a ghost with its
//! infinite vertex replaced by x is positive exactly when x lies outside the
//! hull face. Conflicts are strict: a site on a circumsphere does not destroy
//! the tetrahedron, which yields one valid Delaunay complex for cospherical
//! sites. Sites are inserted in a biased randomized order (rounds from a fixed
//! hash, Hilbert order inside a round); the order is a pure function of the
//! input and uses no engine randomness.
use super::mesh::SiteMesh;
use crate::{GasError, Result};
use robust::{Coord, Coord3D, insphere, orient2d, orient3d};

const INF: u32 = u32::MAX;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct Tet {
    v: [u32; 4],
    /// `adj[k]` is the tetrahedron across the face opposite `v[k]`.
    adj: [u32; 4],
}
impl Tet {
    fn infinite(&self) -> Option<usize> {
        self.v.iter().position(|&x| x == INF)
    }
}

struct Builder<'a> {
    coords: &'a [f64],
    tets: Vec<Tet>,
    alive: Vec<bool>,
    free: Vec<u32>,
    /// Visit stamp per tetrahedron; `stamp == epoch` means visited this insertion.
    stamp: Vec<u32>,
    doomed: Vec<bool>,
    epoch: u32,
}
impl Builder<'_> {
    fn point(&self, i: u32) -> Coord3D<f64> {
        let i = i as usize * 3;
        Coord3D {
            x: self.coords[i],
            y: self.coords[i + 1],
            z: self.coords[i + 2],
        }
    }
    fn orient(&self, v: [u32; 4]) -> f64 {
        orient3d(
            self.point(v[0]),
            self.point(v[1]),
            self.point(v[2]),
            self.point(v[3]),
        )
    }
    fn conflicts(&self, t: u32, p: u32) -> bool {
        let tet = self.tets[t as usize];
        match tet.infinite() {
            None => {
                insphere(
                    self.point(tet.v[0]),
                    self.point(tet.v[1]),
                    self.point(tet.v[2]),
                    self.point(tet.v[3]),
                    self.point(p),
                ) > 0.
            }
            Some(m) => {
                let mut w = tet.v;
                w[m] = p;
                let side = self.orient(w);
                // In the hull-face plane the finite neighbor's circumsphere cuts
                // out exactly the face's circumdisk.
                side > 0. || (side == 0. && self.conflicts(tet.adj[m], p))
            }
        }
    }
    fn allocate(&mut self, tet: Tet) -> u32 {
        if let Some(t) = self.free.pop() {
            self.tets[t as usize] = tet;
            self.alive[t as usize] = true;
            t
        } else {
            self.tets.push(tet);
            self.alive.push(true);
            self.stamp.push(0);
            self.doomed.push(false);
            (self.tets.len() - 1) as u32
        }
    }
    /// Visibility walk; terminates on Delaunay complexes.
    fn locate(&self, start: u32, p: u32) -> Result<u32> {
        let mut t = start;
        for _ in 0..self.tets.len() + 8 {
            let tet = self.tets[t as usize];
            if tet.infinite().is_some() {
                return Ok(t);
            }
            let mut next = NONE;
            for k in 0..4 {
                let mut w = tet.v;
                w[k] = p;
                if self.orient(w) < 0. {
                    next = tet.adj[k];
                    break;
                }
            }
            if next == NONE {
                return Ok(t);
            }
            t = next;
        }
        Err(GasError::Numerical(
            "Delaunay point location did not terminate".into(),
        ))
    }
    /// Insert site `p`, returning a finite tetrahedron incident to it.
    fn insert(&mut self, start: u32, p: u32) -> Result<u32> {
        let seed = self.locate(start, p)?;
        if !self.conflicts(seed, p) {
            return Err(GasError::Numerical(
                "Delaunay insertion found no conflict: duplicate site".into(),
            ));
        }
        self.epoch += 1;
        let mut cavity = vec![seed];
        self.stamp[seed as usize] = self.epoch;
        self.doomed[seed as usize] = true;
        let mut head = 0;
        while head < cavity.len() {
            let t = cavity[head];
            head += 1;
            for k in 0..4 {
                let o = self.tets[t as usize].adj[k];
                if self.stamp[o as usize] == self.epoch {
                    continue;
                }
                self.stamp[o as usize] = self.epoch;
                self.doomed[o as usize] = self.conflicts(o, p);
                if self.doomed[o as usize] {
                    cavity.push(o);
                }
            }
        }
        // One new tetrahedron per cavity boundary facet.
        let mut created = Vec::new();
        // (edge, new tetrahedron, local face) for the faces through p.
        let mut links: Vec<([u32; 2], u32, u8)> = Vec::new();
        for &t in &cavity {
            let old = self.tets[t as usize];
            for k in 0..4 {
                let outside = old.adj[k];
                if self.doomed[outside as usize] && self.stamp[outside as usize] == self.epoch {
                    continue;
                }
                let mut v = old.v;
                v[k] = p;
                let mut adj = [NONE; 4];
                adj[k] = outside;
                let fresh = self.allocate(Tet { v, adj });
                let back = self.tets[outside as usize]
                    .adj
                    .iter()
                    .position(|&x| x == t)
                    .ok_or_else(|| GasError::Numerical("Delaunay adjacency corrupted".into()))?;
                self.tets[outside as usize].adj[back] = fresh;
                for j in 0..4 {
                    if j == k {
                        continue;
                    }
                    let mut e = [0u32; 2];
                    let mut m = 0;
                    for (i, &x) in v.iter().enumerate() {
                        if i != j && i != k {
                            e[m] = x;
                            m += 1;
                        }
                    }
                    e.sort_unstable();
                    links.push((e, fresh, j as u8));
                }
                created.push(fresh);
            }
        }
        links.sort_unstable();
        if !links.len().is_multiple_of(2) {
            return Err(GasError::Numerical("Delaunay cavity is not closed".into()));
        }
        for pair in links.chunks_exact(2) {
            let ((ea, ta, fa), (eb, tb, fb)) = (pair[0], pair[1]);
            if ea != eb || ta == tb {
                return Err(GasError::Numerical("Delaunay cavity is not closed".into()));
            }
            self.tets[ta as usize].adj[fa as usize] = tb;
            self.tets[tb as usize].adj[fb as usize] = ta;
        }
        for &t in &cavity {
            self.alive[t as usize] = false;
            self.doomed[t as usize] = false;
            self.free.push(t);
        }
        created
            .into_iter()
            .find(|&t| self.tets[t as usize].infinite().is_none())
            .ok_or_else(|| GasError::Numerical("Delaunay insertion created no finite cell".into()))
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
/// Hilbert index of a 3D lattice point (Skilling's transform), `bits` per axis.
fn hilbert(mut x: [u32; 3], bits: u32) -> u64 {
    let top = 1u32 << (bits - 1);
    let mut q = top;
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    for i in 1..3 {
        x[i] ^= x[i - 1];
    }
    let mut t = 0;
    let mut q = top;
    while q > 1 {
        if x[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for v in &mut x {
        *v ^= t;
    }
    let mut key = 0u64;
    for b in (0..bits).rev() {
        for v in x {
            key = (key << 1) | u64::from((v >> b) & 1);
        }
    }
    key
}
/// Biased randomized insertion order: geometric rounds, spatially sorted inside.
fn insertion_order(coords: &[f64], n: usize) -> Vec<u32> {
    const BITS: u32 = 16;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in coords.chunks_exact(3) {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let rounds = usize::BITS - n.leading_zeros();
    let mut keyed: Vec<(u32, u64, u32)> = (0..n)
        .map(|i| {
            let cell = std::array::from_fn(|k| {
                let span = hi[k] - lo[k];
                if span > 0. {
                    (((coords[3 * i + k] - lo[k]) / span) * f64::from((1u32 << BITS) - 1)) as u32
                } else {
                    0
                }
            });
            let round = splitmix(i as u64).trailing_zeros().min(rounds);
            (rounds - round, hilbert(cell, BITS), i as u32)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|k| k.2).collect()
}
fn collinear(a: Coord3D<f64>, b: Coord3D<f64>, c: Coord3D<f64>) -> bool {
    let flat = |f: fn(Coord3D<f64>) -> Coord<f64>| orient2d(f(a), f(b), f(c)) == 0.;
    flat(|p| Coord { x: p.x, y: p.y })
        && flat(|p| Coord { x: p.y, y: p.z })
        && flat(|p| Coord { x: p.z, y: p.x })
}
/// Even permutation bringing the vertices into a canonical order.
fn canonical(mut v: [u32; 4]) -> [u32; 4] {
    let k = (0..4).min_by_key(|&k| v[k]).unwrap_or(0);
    if k != 0 {
        v.swap(0, k);
        let (a, b) = match k {
            1 => (2, 3),
            2 => (1, 3),
            _ => (1, 2),
        };
        v.swap(a, b);
    }
    let k = (1..4).min_by_key(|&k| v[k]).unwrap_or(1);
    v[1..].rotate_left(k - 1);
    v
}

pub fn tetrahedralize(coords: &[f64], n: usize) -> Result<SiteMesh> {
    if n < 4 {
        return Err(GasError::Numerical(
            "a tetrahedralization needs four affinely independent sites".into(),
        ));
    }
    let order = insertion_order(coords, n);
    let mut b = Builder {
        coords,
        tets: Vec::with_capacity(7 * n),
        alive: Vec::with_capacity(7 * n),
        free: Vec::new(),
        stamp: Vec::with_capacity(7 * n),
        doomed: Vec::with_capacity(7 * n),
        epoch: 0,
    };
    // First affinely independent quadruple in insertion order.
    let degenerate = || GasError::Numerical("sites are exactly coplanar".into());
    let (i0, i1) = (order[0], order[1]);
    let i2 = *order[2..]
        .iter()
        .find(|&&i| !collinear(b.point(i0), b.point(i1), b.point(i)))
        .ok_or_else(degenerate)?;
    let i3 = *order[2..]
        .iter()
        .find(|&&i| i != i2 && b.orient([i0, i1, i2, i]) != 0.)
        .ok_or_else(degenerate)?;
    let mut v = [i0, i1, i2, i3];
    if b.orient(v) < 0. {
        v.swap(0, 1);
    }
    let root = b.allocate(Tet { v, adj: [NONE; 4] });
    let mut ghosts = [NONE; 4];
    for k in 0..4 {
        let mut g = v;
        g[k] = INF;
        // Flip so that "positive" means outside the hull face.
        let (a, c) = match k {
            0 => (1, 2),
            1 => (0, 2),
            2 => (0, 1),
            _ => (0, 1),
        };
        g.swap(a, c);
        ghosts[k] = b.allocate(Tet {
            v: g,
            adj: [NONE; 4],
        });
    }
    for k in 0..4 {
        b.tets[root as usize].adj[k] = ghosts[k];
        let g = b.tets[ghosts[k] as usize];
        for (slot, &vertex) in g.v.iter().enumerate() {
            b.tets[ghosts[k] as usize].adj[slot] = if vertex == INF {
                root
            } else {
                let j = v
                    .iter()
                    .position(|&x| x == vertex)
                    .ok_or_else(|| GasError::Numerical("Delaunay seed corrupted".into()))?;
                ghosts[j]
            };
        }
    }
    let mut last = root;
    for &p in &order {
        if !v.contains(&p) {
            last = b.insert(last, p)?;
        }
    }
    let mut hull = vec![false; n];
    let mut cells = Vec::new();
    for (t, tet) in b.tets.iter().enumerate() {
        if !b.alive[t] {
            continue;
        }
        if tet.infinite().is_some() {
            for &x in &tet.v {
                if x != INF {
                    hull[x as usize] = true;
                }
            }
        } else {
            cells.push(canonical(tet.v));
        }
    }
    cells.sort_unstable();
    let simplices: Vec<u32> = cells.into_iter().flatten().collect();
    Ok(SiteMesh {
        dimension: 3,
        coords: coords.to_vec(),
        edges: SiteMesh::edges_from_simplices(&simplices, 4),
        simplices,
        hull,
        real: n,
        base: (0..n as u32).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cloud(n: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let t = i as f64 + 1.;
                [
                    (t * 0.819172513).fract(),
                    (t * 0.671043606).fract(),
                    (t * 0.549700477).fract(),
                ]
            })
            .collect()
    }
    fn point(c: &[f64], i: u32) -> Coord3D<f64> {
        let i = i as usize * 3;
        Coord3D {
            x: c[i],
            y: c[i + 1],
            z: c[i + 2],
        }
    }
    /// Positive orientation, empty circumspheres (non-strict), Euler relation
    /// and the hull flag of every site.
    fn check(c: &[f64], n: usize) -> SiteMesh {
        let m = tetrahedralize(c, n).unwrap();
        let mut faces = Vec::new();
        for s in m.simplices.chunks_exact(4) {
            let p: Vec<_> = s.iter().map(|&i| point(c, i)).collect();
            assert!(orient3d(p[0], p[1], p[2], p[3]) > 0.);
            for q in 0..n as u32 {
                if !s.contains(&q) {
                    assert!(insphere(p[0], p[1], p[2], p[3], point(c, q)) <= 0.);
                }
            }
            for k in 0..4 {
                let mut f: Vec<u32> = (0..4).filter(|&j| j != k).map(|j| s[j]).collect();
                f.sort_unstable();
                faces.push(f);
            }
        }
        faces.sort();
        let total = faces.len();
        let mut boundary = Vec::new();
        let mut k = 0;
        while k < faces.len() {
            let run = faces[k..].iter().take_while(|f| **f == faces[k]).count();
            assert!(run <= 2);
            if run == 1 {
                boundary.push(faces[k].clone());
            }
            k += run;
        }
        let distinct = (total + boundary.len()) / 2;
        // Euler characteristic of a triangulated ball.
        assert_eq!(
            n as i64 - m.edges.len() as i64 + distinct as i64 - m.simplex_count() as i64,
            1
        );
        let mut on_hull = vec![false; n];
        for f in &boundary {
            for &i in f {
                on_hull[i as usize] = true;
            }
        }
        assert_eq!(on_hull, m.hull);
        m
    }
    #[test]
    fn generic_cloud_is_delaunay() {
        let n = 150;
        check(&cloud(n), n);
    }
    #[test]
    fn cubic_lattice_with_cospherical_sites_is_valid() {
        let mut c = Vec::new();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    c.extend([x as f64, y as f64, z as f64]);
                }
            }
        }
        let m = check(&c, 64);
        // The lattice cube has volume 27 regardless of the chosen diagonals.
        let volume: f64 = m
            .simplices
            .chunks_exact(4)
            .map(|s| {
                let p: Vec<_> = s.iter().map(|&i| point(&c, i)).collect();
                orient3d(p[0], p[1], p[2], p[3]) / 6.
            })
            .sum();
        assert!((volume - 27.).abs() < 1e-12);
    }
    #[test]
    fn sites_on_a_sphere_and_minimal_inputs() {
        let n = 40;
        let c: Vec<f64> = (0..n)
            .flat_map(|i| {
                let z = 1. - 2. * (i as f64 + 0.5) / n as f64;
                let r = (1. - z * z).sqrt();
                let a = i as f64 * 2.399963229728653;
                [r * a.cos(), r * a.sin(), z]
            })
            .collect();
        let m = check(&c, n);
        assert!(m.hull.iter().all(|&h| h));
        let m = check(&[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.], 4);
        assert_eq!(m.simplex_count(), 1);
        assert!(tetrahedralize(&[0.; 9], 3).is_err());
        let flat: Vec<f64> = (0..5)
            .flat_map(|i| [i as f64, (i * i) as f64, 0.])
            .collect();
        assert!(tetrahedralize(&flat, 5).is_err());
    }
    #[test]
    fn result_does_not_depend_on_site_numbering() {
        let n = 80;
        let c = cloud(n);
        let a = tetrahedralize(&c, n).unwrap();
        let reversed: Vec<f64> = c.chunks_exact(3).rev().flatten().copied().collect();
        let b = tetrahedralize(&reversed, n).unwrap();
        let relabel = |e: &[u32; 2]| {
            let mut e = [n as u32 - 1 - e[0], n as u32 - 1 - e[1]];
            e.sort_unstable();
            e
        };
        let mut mapped: Vec<[u32; 2]> = b.edges.iter().map(relabel).collect();
        mapped.sort_unstable();
        assert_eq!(a.edges, mapped);
    }
    #[test]
    fn hilbert_indices_are_a_bijection_with_unit_steps() {
        let bits = 3;
        let side = 1u32 << bits;
        let mut cells = vec![[0u32; 3]; (side * side * side) as usize];
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    cells[hilbert([x, y, z], bits) as usize] = [x, y, z];
                }
            }
        }
        for w in cells.windows(2) {
            let step: u32 = (0..3).map(|k| w[0][k].abs_diff(w[1][k])).sum();
            assert_eq!(step, 1);
        }
    }
}
