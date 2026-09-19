//! From walkers to tessellation sites: coordinate projection, eligibility
//! compaction and exact deduplication. Coincident walkers share one site; no
//! coordinate is ever perturbed.
use crate::{GasError, Real, Result, TensorBatch, boundary::BoxDomain, error::require};
use serde::{Deserialize, Serialize};

/// Which position coordinates span the tessellated space.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Projection {
    /// Every coordinate.
    #[default]
    Full,
    /// Drop the last coordinate (Euclidean time) when the ambient dimension is
    /// at least `min_ambient`; otherwise keep every coordinate.
    DropLast { min_ambient: usize },
    /// The first `dims` coordinates.
    Leading { dims: usize },
    /// An explicit list of distinct coordinate indices.
    Axes { indices: Vec<usize> },
}
impl Projection {
    pub fn axes(&self, ambient: usize) -> Result<Vec<usize>> {
        let axes: Vec<usize> = match self {
            Self::Full => (0..ambient).collect(),
            Self::DropLast { min_ambient } => (0..if ambient >= *min_ambient && ambient > 1 {
                ambient - 1
            } else {
                ambient
            })
                .collect(),
            Self::Leading { dims } => (0..*dims).collect(),
            Self::Axes { indices } => indices.clone(),
        };
        let mut sorted = axes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        require(
            !axes.is_empty() && sorted.len() == axes.len() && axes.iter().all(|&a| a < ambient),
            "tessellation projection must select distinct coordinates of the position field",
        )?;
        Ok(axes)
    }
}

pub const NO_SITE: u32 = u32::MAX;

/// Unique projected sites in lexicographic order and their walker groups.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteSet {
    pub dimension: usize,
    pub walkers: usize,
    /// `sites() * dimension` coordinates, exact f64 images of the run dtype.
    pub coords: Vec<f64>,
    /// Site of every walker; `NO_SITE` for ineligible walkers.
    pub site_of_walker: Vec<u32>,
    group_offsets: Vec<u32>,
    group_members: Vec<u32>,
}
impl SiteSet {
    pub fn build<T: Real>(
        positions: &TensorBatch<T>,
        axes: &[usize],
        eligible: &[bool],
        wrap: Option<&BoxDomain>,
    ) -> Result<Self> {
        let n = positions.rows();
        let width = positions.width();
        require(
            positions.item_shape().len() == 1 && eligible.len() == n,
            "tessellation positions must be one vector per walker",
        )?;
        require(n <= i32::MAX as usize, "tessellation walker count")?;
        let d = axes.len();
        let values = positions.values();
        let mut projected = vec![0f64; n * d];
        for i in 0..n {
            if !eligible[i] {
                continue;
            }
            for (k, &a) in axes.iter().enumerate() {
                // +0.0 and -0.0 are the same site.
                let mut v = values[i * width + a].to_f64();
                if let Some(b) = wrap {
                    // Walkers one period apart are the same periodic site.
                    v = b.lower[k] + (v - b.lower[k]).rem_euclid(b.upper[k] - b.lower[k]);
                    // rem_euclid rounds a tiny negative offset up to the period:
                    // that point is the lower face, as in the boundary repair.
                    if v >= b.upper[k] {
                        v = b.lower[k];
                    }
                }
                let v = v + 0.;
                if !v.is_finite() {
                    return Err(GasError::Numerical(format!(
                        "non-finite tessellation coordinate at walker {i}"
                    )));
                }
                projected[i * d + k] = v;
            }
        }
        let mut order: Vec<u32> = (0..n as u32).filter(|&i| eligible[i as usize]).collect();
        let row = |i: u32| &projected[i as usize * d..(i as usize + 1) * d];
        order.sort_unstable_by(|&a, &b| {
            row(a)
                .iter()
                .zip(row(b))
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or_else(|| a.cmp(&b))
        });
        let mut out = Self {
            dimension: d,
            walkers: n,
            coords: Vec::new(),
            site_of_walker: vec![NO_SITE; n],
            group_offsets: vec![0],
            group_members: Vec::with_capacity(order.len()),
        };
        for (k, &i) in order.iter().enumerate() {
            if k == 0 || row(order[k - 1]) != row(i) {
                if k > 0 {
                    out.group_offsets.push(k as u32);
                }
                out.coords.extend_from_slice(row(i));
            }
            out.site_of_walker[i as usize] = (out.group_offsets.len() - 1) as u32;
            out.group_members.push(i);
        }
        if order.is_empty() {
            out.group_offsets.clear();
            out.group_offsets.push(0);
        } else {
            out.group_offsets.push(order.len() as u32);
        }
        Ok(out)
    }
    pub fn sites(&self) -> usize {
        self.group_offsets.len() - 1
    }
    pub fn site(&self, s: usize) -> &[f64] {
        &self.coords[s * self.dimension..(s + 1) * self.dimension]
    }
    /// Walkers at site `s`, ascending.
    pub fn group(&self, s: usize) -> &[u32] {
        &self.group_members[self.group_offsets[s] as usize..self.group_offsets[s + 1] as usize]
    }
    pub fn has_duplicates(&self) -> bool {
        self.group_members.len() > self.sites()
    }
    pub fn duplicate_groups(&self) -> usize {
        (0..self.sites())
            .filter(|&s| self.group(s).len() > 1)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_rules() {
        assert_eq!(
            Projection::DropLast { min_ambient: 3 }.axes(3).unwrap(),
            [0, 1]
        );
        assert_eq!(
            Projection::DropLast { min_ambient: 3 }.axes(2).unwrap(),
            [0, 1]
        );
        assert_eq!(Projection::Leading { dims: 2 }.axes(4).unwrap(), [0, 1]);
        assert!(Projection::Leading { dims: 5 }.axes(4).is_err());
        assert!(
            Projection::Axes {
                indices: vec![1, 1]
            }
            .axes(3)
            .is_err()
        );
    }
    #[test]
    fn duplicates_share_a_site_and_dead_walkers_have_none() {
        let x = TensorBatch::vectors(
            6,
            3,
            vec![
                1f32, 0., 9., 0., 0., 9., 1., -0., 7., 5., 5., 5., 0., 0., 1., 1., 0., 3.,
            ],
        )
        .unwrap();
        let s = SiteSet::build(&x, &[0, 1], &[true, true, true, false, true, true], None).unwrap();
        assert_eq!(s.sites(), 2);
        assert_eq!(s.coords, [0., 0., 1., 0.]);
        assert_eq!(s.group(0), &[1, 4]);
        assert_eq!(s.group(1), &[0, 2, 5]);
        assert_eq!(s.site_of_walker, [1, 0, 1, NO_SITE, 0, 1]);
        assert!(s.has_duplicates() && s.duplicate_groups() == 2);
        let none = SiteSet::build(&x, &[0], &[false; 6], None).unwrap();
        assert_eq!(none.sites(), 0);
    }
}
