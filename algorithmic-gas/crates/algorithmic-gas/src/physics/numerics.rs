//! Statistics and fitting kit for recorded time series: plain `f64`, flat
//! row-major arrays, no physics vocabulary. Bootstrap draws are addressed by
//! the resampling seed and the resample index, never by an engine address.
pub mod covariance;
pub mod eigen;
pub mod least_squares;
pub mod levenberg;
pub mod moments;
pub mod resample;
pub mod shortest_paths;
pub mod special;
use crate::{GasError, Result, error::require};
pub use covariance::{Whitener, submatrix};
pub use eigen::{GeneralizedEigen, generalized_symmetric};
pub use least_squares::{LinearFit, linear_fit};
pub use levenberg::{LevenbergConfig, Model, NonlinearFit, minimize};
pub use moments::{BlockMoments, LagMoments, cross_moments, series_moments};
pub use resample::{
    TauInt, auto_block, errors, resample, resample_blocks, sample_covariance, tau_int,
    tau_int_of_correlator,
};
use serde::{Deserialize, Serialize};
pub use special::{chi2_q, gamma_q};

/// Number of consecutive time origins that form one resampling block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BlockSize {
    /// At least twice the integrated autocorrelation time of the operator
    /// series (Sokal window), growing as `T^{1/3}` with the number of time
    /// origins `T`: `max(2 τ_int, (2 τ_int)^{2/3} T^{1/3})`. A block of exactly
    /// `2 τ_int` underestimates the variance of a slowly mixing series by
    /// tens of percent.
    #[default]
    Auto,
    Fixed {
        frames: usize,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Resampling {
    /// Delete-one-block jackknife.
    BlockJackknife {
        #[serde(default)]
        block: BlockSize,
    },
    /// Whole blocks drawn with replacement. Resample `r` draws from
    /// `RandomStream::new(seed, 0, Stream::Initialize, r, 811)`, so the block
    /// indices are a function of `(seed, blocks, r)` only and series resampled
    /// separately over the same blocks stay jointly consistent.
    Bootstrap {
        #[serde(default)]
        block: BlockSize,
        samples: usize,
        seed: u64,
    },
    /// Every time origin is its own block. Only for independent draws.
    Uncorrelated,
}
impl Default for Resampling {
    fn default() -> Self {
        Self::BlockJackknife {
            block: BlockSize::Auto,
        }
    }
}
impl Resampling {
    pub fn validate(&self) -> Result<()> {
        let block = match self {
            Self::BlockJackknife { block } => Some(block),
            Self::Bootstrap {
                block, samples: n, ..
            } => {
                require(
                    (8..=100_000).contains(n),
                    "bootstrap requires 8..=100000 samples",
                )?;
                Some(block)
            }
            Self::Uncorrelated => None,
        };
        require(
            !matches!(block, Some(BlockSize::Fixed { frames: 0 })),
            "fixed resampling block must hold at least one frame",
        )
    }
    pub fn kind(&self) -> ResampleKind {
        match self {
            Self::BlockJackknife { .. } | Self::Uncorrelated => ResampleKind::Jackknife,
            Self::Bootstrap { .. } => ResampleKind::Bootstrap,
        }
    }
    pub fn block(&self) -> BlockSize {
        match self {
            Self::BlockJackknife { block } | Self::Bootstrap { block, .. } => *block,
            Self::Uncorrelated => BlockSize::Fixed { frames: 1 },
        }
    }
}

/// Disconnected part removed from a lagged moment `Σab/Σn`. Every resample
/// recomputes it from its own sums.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Subtraction {
    /// The raw moment.
    None,
    /// Lag-dependent means of both legs: `Σab/Σn − (Σa/Σn)·(Σb/Σn)`. It decays
    /// to zero when the legs decorrelate, whatever the masks.
    #[default]
    LagMeans,
    /// One mean `Ō = Σa/Σn` at lag 0 subtracted from both legs:
    /// `Σab/Σn − Ō·(Σa/Σn + Σb/Σn) + Ō·Ō`. The global mean of a frame series
    /// and the single source mean of a source-frozen correlator. Undefined
    /// where the lag-0 denominator vanishes.
    GlobalMean,
}

/// Which variance formula a set of resamples obeys: `(n-1)/n Σ(θ_r-θ̄)²` for
/// the jackknife, `1/(n-1) Σ(θ_r-θ̄)²` for the bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResampleKind {
    Jackknife,
    Bootstrap,
}

/// Resampled estimates of one vector-valued quantity (a correlator over lags,
/// or several correlators concatenated). Every resample recomputes its own
/// disconnected part.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Samples {
    pub kind: ResampleKind,
    /// Length of the estimated vector.
    pub dimension: usize,
    /// Number of resamples.
    pub count: usize,
    /// Pooled point estimate, `[dimension]`; not the mean of `values`.
    pub central: Vec<f64>,
    /// `[count, dimension]`.
    pub values: Vec<f64>,
    /// `[dimension]`; false where the central value or any resample has a zero
    /// denominator. Entries of `central`/`values` there are 0 and carry no meaning.
    pub defined: Vec<bool>,
    /// Time origins per block actually used, and the number of such blocks.
    pub effective_block: usize,
    pub blocks: usize,
    /// Integrated autocorrelation time that selected an automatic block.
    pub tau_int: Option<f64>,
}
impl Samples {
    pub fn validate(&self) -> Result<()> {
        require(
            self.central.len() == self.dimension
                && self.defined.len() == self.dimension
                && self.count.checked_mul(self.dimension) == Some(self.values.len())
                && self
                    .central
                    .iter()
                    .chain(&self.values)
                    .all(|x| x.is_finite()),
            "resample table shape or nonfinite entry",
        )
    }
    pub fn sample(&self, r: usize) -> &[f64] {
        &self.values[r * self.dimension..(r + 1) * self.dimension]
    }
    /// Entries `indices` of every resample, in the given order.
    pub fn select(&self, indices: &[usize]) -> Result<Self> {
        require(
            indices.iter().all(|&i| i < self.dimension),
            "resample selection outside the estimated vector",
        )?;
        Ok(Self {
            dimension: indices.len(),
            central: indices.iter().map(|&i| self.central[i]).collect(),
            values: (0..self.count)
                .flat_map(|r| indices.iter().map(move |&i| (r, i)))
                .map(|(r, i)| self.values[r * self.dimension + i])
                .collect(),
            defined: indices.iter().map(|&i| self.defined[i]).collect(),
            ..self.clone()
        })
    }
    /// Joint table of quantities resampled over the same blocks.
    pub fn concat(parts: &[&Self]) -> Result<Self> {
        let first = parts
            .first()
            .ok_or_else(|| GasError::Configuration("no resample tables to join".into()))?;
        require(
            parts.iter().all(|p| {
                p.kind == first.kind
                    && p.count == first.count
                    && p.blocks == first.blocks
                    && p.effective_block == first.effective_block
            }),
            "joined resample tables must share kind, blocks and count",
        )?;
        let dimension = parts.iter().map(|p| p.dimension).sum();
        let mut values = Vec::with_capacity(first.count * dimension);
        for r in 0..first.count {
            for p in parts {
                values.extend_from_slice(p.sample(r));
            }
        }
        Ok(Self {
            kind: first.kind,
            dimension,
            count: first.count,
            central: parts.iter().flat_map(|p| p.central.clone()).collect(),
            values,
            defined: parts.iter().flat_map(|p| p.defined.clone()).collect(),
            effective_block: first.effective_block,
            blocks: first.blocks,
            tau_int: parts
                .iter()
                .filter_map(|p| p.tau_int)
                .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b)))),
        })
    }
}

/// A retained time series: `values` is `[frames, components]`, `weight` is
/// `[frames]` (zero marks a frame with no valid measurement; its values are
/// ignored), `segment` is `[frames]` and lagged products never pair frames of
/// different segments.
#[derive(Clone, Copy, Debug)]
pub struct SeriesView<'a> {
    pub values: &'a [f64],
    pub weight: &'a [f64],
    pub segment: &'a [u32],
    pub components: usize,
}
impl SeriesView<'_> {
    pub fn frames(&self) -> usize {
        self.weight.len()
    }
    pub fn validate(&self) -> Result<()> {
        require(
            self.components > 0
                && self.segment.len() == self.weight.len()
                && self.weight.len().checked_mul(self.components) == Some(self.values.len())
                && self.weight.iter().all(|w| w.is_finite() && *w >= 0.)
                && self
                    .values
                    .chunks_exact(self.components)
                    .zip(self.weight)
                    .all(|(v, &w)| w == 0. || v.iter().all(|x| x.is_finite())),
            "series shape, negative weight or nonfinite weighted value",
        )
    }
}

/// Prior on one fit parameter. A log-normal prior on `x` is a Gaussian prior on
/// the fit parameter `ln x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Prior {
    #[default]
    Flat,
    Gaussian {
        mean: f64,
        sigma: f64,
    },
}
impl Prior {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Flat => Ok(()),
            Self::Gaussian { mean, sigma } => require(
                mean.is_finite() && sigma.is_finite() && *sigma > 0.,
                "Gaussian prior requires a finite mean and a positive width",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resampling_choices_round_trip_and_reject_unknown_fields_and_empty_blocks() {
        let bootstrap: Resampling = serde_json::from_str(
            r#"{"kind":"bootstrap","block":{"kind":"fixed","frames":20},"samples":200,"seed":1}"#,
        )
        .unwrap();
        bootstrap.validate().unwrap();
        assert_eq!(bootstrap.kind(), ResampleKind::Bootstrap);
        assert_eq!(bootstrap.block(), BlockSize::Fixed { frames: 20 });
        let text = serde_json::to_string(&bootstrap).unwrap();
        assert_eq!(
            serde_json::from_str::<Resampling>(&text).unwrap(),
            bootstrap
        );
        assert!(serde_json::from_str::<Resampling>(r#"{"kind":"block_jackknife","x":1}"#).is_err());
        let empty = Resampling::BlockJackknife {
            block: BlockSize::Fixed { frames: 0 },
        };
        assert!(matches!(empty.validate(), Err(GasError::Configuration(_))));
        assert_eq!(
            serde_json::to_string(&Subtraction::GlobalMean).unwrap(),
            r#""global_mean""#
        );
        assert!(
            Prior::Gaussian {
                mean: 0.,
                sigma: 0.
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn resample_tables_select_and_join_over_common_blocks() {
        let table = |central: Vec<f64>, values: Vec<f64>| Samples {
            kind: ResampleKind::Jackknife,
            dimension: central.len(),
            count: 2,
            defined: vec![true; central.len()],
            central,
            values,
            effective_block: 4,
            blocks: 2,
            tau_int: Some(1.5),
        };
        let a = table(vec![1., 2.], vec![0.9, 2.1, 1.1, 1.9]);
        let b = table(vec![3.], vec![2.5, 3.5]);
        a.validate().unwrap();
        let joint = Samples::concat(&[&a, &b]).unwrap();
        assert_eq!(joint.sample(1), &[1.1, 1.9, 3.5]);
        assert_eq!(joint.select(&[2, 0]).unwrap().central, vec![3., 1.]);
        assert!(joint.select(&[3]).is_err());
        let other = Samples { blocks: 3, ..b };
        assert!(Samples::concat(&[&a, &other]).is_err());
    }
}
