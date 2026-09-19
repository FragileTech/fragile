//! Tessellation domains. A bounded domain is realized by ghost sites added
//! before triangulating: mirror images across the faces of a clip box make
//! every Voronoi cell end exactly at the box, and translated images close a
//! periodic box. Only images within a margin of the box are generated; the
//! margin grows until every simplex touching a real site has its circumsphere
//! inside the covered region, which certifies the real cells.
use crate::{Result, boundary::BoxDomain, error::require};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TessellationDomain {
    /// All of space: hull cells are unbounded.
    #[default]
    Open,
    /// Cells are clipped to the box; sites must lie inside it.
    ClipBox { bounds: BoxDomain },
    /// Periodic box: neighbors wrap around and every cell is bounded.
    Periodic { bounds: BoxDomain },
}
impl TessellationDomain {
    pub fn bounds(&self) -> Option<&BoxDomain> {
        match self {
            Self::Open => None,
            Self::ClipBox { bounds } | Self::Periodic { bounds } => Some(bounds),
        }
    }
    pub fn periodic(&self) -> bool {
        matches!(self, Self::Periodic { .. })
    }
    pub fn validate(&self, dimension: usize) -> Result<()> {
        self.bounds().map_or(Ok(()), |b| b.validate(dimension))
    }
    /// Box lengths of a periodic domain.
    pub fn period(&self) -> Option<Vec<f64>> {
        match self {
            Self::Periodic { bounds } => Some(
                bounds
                    .lower
                    .iter()
                    .zip(&bounds.upper)
                    .map(|(a, b)| b - a)
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// Real sites followed by their domain images.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedSites {
    pub coords: Vec<f64>,
    /// Real site of every extended site.
    pub base: Vec<u32>,
    pub real: usize,
    /// True when every image the domain can produce is present.
    pub complete: bool,
}

/// Sites with their images within `margin` of the box. Periodic sites are
/// first wrapped into the box.
pub fn extend(
    coords: &[f64],
    n: usize,
    d: usize,
    domain: &TessellationDomain,
    margin: f64,
) -> Result<ExtendedSites> {
    let mut out = ExtendedSites {
        coords: coords.to_vec(),
        base: (0..n as u32).collect(),
        real: n,
        complete: true,
    };
    let Some(bounds) = domain.bounds() else {
        return Ok(out);
    };
    let length: Vec<f64> = (0..d).map(|k| bounds.upper[k] - bounds.lower[k]).collect();
    out.complete = length.iter().all(|&l| margin >= l);
    match domain {
        TessellationDomain::Open => {}
        TessellationDomain::Periodic { .. } => {
            for p in out.coords.chunks_exact_mut(d) {
                for k in 0..d {
                    p[k] = bounds.lower[k] + (p[k] - bounds.lower[k]).rem_euclid(length[k]);
                    if p[k] >= bounds.upper[k] {
                        p[k] = bounds.lower[k];
                    }
                }
            }
            let shifts = 3usize.pow(d as u32);
            for s in 0..n {
                let p = out.coords[s * d..(s + 1) * d].to_vec();
                for code in 0..shifts {
                    if code == shifts / 2 {
                        continue;
                    }
                    let mut c = code;
                    let mut image = p.clone();
                    let mut inside = true;
                    for k in 0..d {
                        image[k] += (c % 3) as f64 * length[k] - length[k];
                        c /= 3;
                        inside &= image[k] >= bounds.lower[k] - margin
                            && image[k] <= bounds.upper[k] + margin;
                    }
                    if inside {
                        out.coords.extend(image);
                        out.base.push(s as u32);
                    }
                }
            }
        }
        TessellationDomain::ClipBox { .. } => {
            for s in 0..n {
                let p = coords[s * d..(s + 1) * d].to_vec();
                require(
                    (0..d).all(|k| p[k] >= bounds.lower[k] && p[k] <= bounds.upper[k]),
                    "a clip-box tessellation requires every site inside the box",
                )?;
                for k in 0..d {
                    for (face, near) in [
                        (bounds.lower[k], p[k] - bounds.lower[k]),
                        (bounds.upper[k], bounds.upper[k] - p[k]),
                    ] {
                        // A site on the face is its own mirror image.
                        if near > 0. && near <= margin {
                            let mut image = p.clone();
                            image[k] = 2. * face - p[k];
                            out.coords.extend(image);
                            out.base.push(s as u32);
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}
/// First margin tried: a few mean site spacings.
pub fn initial_margin(n: usize, d: usize, domain: &TessellationDomain) -> f64 {
    domain.bounds().map_or(0., |b| {
        let volume: f64 = (0..d).map(|k| b.upper[k] - b.lower[k]).product();
        3. * (volume / n.max(1) as f64).powf(1. / d as f64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn unit() -> BoxDomain {
        BoxDomain {
            lower: vec![0., 0.],
            upper: vec![1., 1.],
        }
    }
    #[test]
    fn periodic_images_wrap_and_respect_the_margin() {
        let domain = TessellationDomain::Periodic { bounds: unit() };
        let e = extend(&[1.25, 0.5, 0.1, 0.5], 2, 2, &domain, 0.2).unwrap();
        assert_eq!(&e.coords[..4], &[0.25, 0.5, 0.1, 0.5]);
        // Only the second site is within 0.2 of a face: one image at x = 1.1.
        assert_eq!((e.base.len(), e.complete), (3, false));
        assert_eq!(&e.coords[4..], &[1.1, 0.5]);
        let all = extend(&[0.25, 0.5], 1, 2, &domain, 1.).unwrap();
        assert_eq!((all.base.len(), all.complete), (9, true));
    }
    #[test]
    fn clip_box_mirrors_across_near_faces_only() {
        let domain = TessellationDomain::ClipBox { bounds: unit() };
        let e = extend(&[0.1, 0.5], 1, 2, &domain, 0.2).unwrap();
        assert_eq!(e.coords, [0.1, 0.5, -0.1, 0.5]);
        assert!(extend(&[1.5, 0.5], 1, 2, &domain, 0.2).is_err());
        let on_face = extend(&[0., 0.5], 1, 2, &domain, 2.).unwrap();
        assert_eq!(on_face.base.len(), 4);
    }
}
