//! Measurement → correlator estimates with resamples. Three estimators share
//! one output type: the algorithm-time correlation of frame averages, the
//! source-frozen propagator, and the Euclidean-time slab correlation.
use super::{
    config::{AnalysisConfig, Combine, TimeAxis, TimeUnit},
    contract::{EXCHANGE_ODD_REASON, ExchangeParity},
    measurement::{ChannelSeries, Measurement, SlabMoments},
    report::{CorrelatorEstimate, EstimatorKind, SamplesMeta},
};
use crate::{
    GasError, Result,
    error::require,
    memory::{DEFAULT_MEMORY_BYTES, checked_add, checked_mul, enforce},
    physics::numerics::{
        BlockMoments, BlockSize, Resampling, Samples, SeriesView, Subtraction, Whitener,
        auto_block, cross_moments, errors, moments::straddling_pairs, resample,
        resample::block_below_target, resample_blocks, sample_covariance, series_moments,
        submatrix, tau_int,
    },
};

/// A correlator and the resamples its errors came from; fits read both.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimated {
    pub channel: String,
    pub kind: EstimatorKind,
    /// `covariance` is always `Some(sample_covariance(&samples))` here;
    /// `analyze` drops it from the report unless `report_covariance`.
    pub estimate: CorrelatorEstimate,
    /// Dimension `estimate.lags.len()`. Fits take their times from
    /// `estimate.lags[k] as f64 * estimate.time_step`.
    pub samples: Samples,
}
impl Estimated {
    /// What the resampling behind this estimate does not cover, in words: the
    /// deletion geometry at the lags beyond the block, an automatic block that
    /// stopped below the batch-size rule, and a resample table too small for
    /// the estimated lags. Both block statements count the fewest origins a
    /// table of this shape can hold, `(blocks − 1) · effective_block + 1`, so
    /// that each is a bound the measurement cannot undercut. A fit that scans
    /// a shorter window asks `resample_support_note` again with its own point
    /// count.
    pub fn notes(&self) -> Vec<String> {
        let meta = &self.estimate.samples_meta;
        let (block, blocks) = (meta.effective_block.max(1), meta.blocks);
        let origins = blocks
            .saturating_sub(1)
            .saturating_mul(block)
            .saturating_add(1);
        let mut notes = Vec::new();
        if let Some(&max_lag) = self.estimate.lags.last().filter(|&&lag| lag > block) {
            notes.push(format!(
                "{DELETION_GEOMETRY_NOTE}: at least {} of the pairs up to lag {max_lag} straddle \
                 the block of {block} origins and survive the deletion of it",
                straddling_pairs(max_lag.saturating_add(1), block, origins)
            ));
        }
        if let Some(tau) = meta
            .tau_int
            .filter(|&tau| block_below_target(tau, block, origins))
        {
            notes.push(format!(
                "the automatic block of {block} origins is below the batch size an \
                 autocorrelation time of {tau} asks over {origins} origins, so the resampled \
                 variance is biased low"
            ));
        }
        notes.extend(resample_support_note(blocks, self.estimate.lags.len()));
        notes
    }
}

/// The lag-versus-block relation a delete-one resampling cannot escape: it
/// deletes time origins, and a product whose origin survives keeps its sink.
pub const DELETION_GEOMETRY_NOTE: &str = "delete-one removes origins, not observations; at \
    lags beyond the block the error is a lower bound";

/// The statement a fit of `points` correlated points owes when the resample
/// table cannot support them: `blocks - 1` is the rank a delete-one table can
/// reach, so a larger window leaves a singular lag covariance whose SVD floor
/// fabricates the missing directions. `None` when the table supports them.
pub fn resample_support_note(blocks: usize, points: usize) -> Option<String> {
    let replicas = blocks.saturating_sub(1);
    (replicas < points).then(|| {
        format!(
            "{blocks} resampling blocks leave {replicas} independent replicas against \
             {points} fitted points: the lag covariance is rank deficient by construction"
        )
    })
}

/// Cross-correlation moments of a channel basis over common blocks:
/// `moments[a * channels.len() + b]` holds `⟨O_a(t) O_b(t+τ)⟩`. `gevp` calls
/// `resample(&moments[k], subtraction, &analysis.resampling, tau_int)` with
/// the same `tau_int` for every entry, so that all entries share their blocks
/// and bootstrap draws; an automatic block per entry would be estimated from
/// meaningless off-diagonal ratios.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixCorrelator {
    pub channels: Vec<String>,
    /// Lag of every entry of the moments, in frames.
    pub lags: Vec<usize>,
    pub time_unit: TimeUnit,
    pub time_step: f64,
    /// Measured frames summed over replicas.
    pub frames: u64,
    /// Largest integrated autocorrelation time of the diagonal members.
    pub tau_int: Option<f64>,
    pub moments: Vec<BlockMoments>,
    pub replicas: usize,
}

/// Blocked moments of one channel under one estimator, combined over replicas
/// as `analysis.combine` says (`PooledBlocks` concatenates the replicas'
/// blocks; `RunsAsSamples` makes every replica one block and needs at least 8).
/// `GasError::Capability` when the channel or the estimator was not measured.
/// The Euclidean-time slab correlator has no such form: its disconnected part
/// needs a mean per bin, which lagged moments cannot carry once the bins of a
/// lag are summed. `estimate` resamples its `SlabMoments` directly.
pub fn moments(
    measurements: &[Measurement],
    channel: &str,
    kind: EstimatorKind,
    analysis: &AnalysisConfig,
) -> Result<BlockMoments> {
    analysis.validate()?;
    let members = members(measurements, channel)?;
    let tau = series_tau(&members);
    lagged_moments(&members, kind, analysis, tau)
}

/// Point estimate (the pooled moments), resamples, errors, covariance (always
/// filled), rank and the connected-estimator bias of one channel. The
/// disconnected part is `Subtraction::None` when `!analysis.connected`,
/// `analysis.frame_subtraction` for the frame mean and
/// `analysis.propagator_subtraction` for the source-frozen propagator, always
/// formed from sums pooled over replicas; the Euclidean-time estimator resamples
/// `SlabMoments` through `numerics::resample_blocks` with per-bin means and
/// reports `TimeUnit::Coordinate` with the bin width as `time_step`. Rates
/// fitted downstream are labelled by estimator (`report::RATE_QUANTITY*`).
/// `tau` overrides the autocorrelation time of an automatic block, which is
/// otherwise that of the retained operator series. Too few blocks for the
/// requested resampling is `GasError::Numerical`.
///
/// The measurements need not be validated: every series, block table and slab
/// table this reads is validated where it is read, and a malformed one is an
/// error rather than a panic.
pub fn estimate(
    measurements: &[Measurement],
    channel: &str,
    kind: EstimatorKind,
    analysis: &AnalysisConfig,
    tau: Option<f64>,
) -> Result<Estimated> {
    analysis.validate()?;
    let members = members(measurements, channel)?;
    let tau = tau.or_else(|| series_tau(&members));
    let connected = analysis.connected;
    let (samples, lags, time_unit, time_step) = match kind {
        EstimatorKind::EuclideanTime => {
            let (samples, lags, width) = euclidean(&members, analysis, tau)?;
            (samples, lags, TimeUnit::Coordinate, width)
        }
        EstimatorKind::FrameMean | EstimatorKind::SourceFrozen => {
            let moments = lagged_moments(&members, kind, analysis, tau)?;
            let subtraction = match (connected, kind) {
                (false, _) => Subtraction::None,
                (true, EstimatorKind::FrameMean) => analysis.frame_subtraction,
                (true, _) => analysis.propagator_subtraction,
            };
            let samples = resample(
                &moments,
                subtraction,
                &unit_resampling(analysis.resampling, analysis.combine),
                tau,
            )?;
            // Entry `k` of the stored moments is the frame lag `k`. A
            // propagator evaluates its sinks every `lag_stride` frames, so the
            // lags in between hold no pair at all: they are left out of the
            // correlator instead of entering a fit as undefined points.
            let stride = match kind {
                EstimatorKind::SourceFrozen => members[0].0.config.propagators.lag_stride.max(1),
                _ => 1,
            };
            let lags: Vec<usize> = (0..moments.lags).step_by(stride).collect();
            let samples = if stride > 1 {
                samples.select(&lags)?
            } else {
                samples
            };
            let step = time_step(members[0].0, analysis.time_unit)?;
            (samples, lags, analysis.time_unit, step)
        }
    };
    // The closed form of the centering bias is the frame series'; it is
    // reported with the autocorrelation time that selected the block, so the
    // two numbers of the note come from one estimate.
    let variance = samples
        .defined
        .first()
        .and_then(|defined| defined.then(|| samples.central[0]));
    let connected_bias = match (kind, connected, samples.tau_int, variance) {
        (EstimatorKind::FrameMean, true, Some(tau), Some(variance)) => {
            Some(-2. * tau * variance / retained_weight(&members)?)
        }
        _ => None,
    };
    enforce(
        checked_mul(checked_mul(samples.dimension, samples.dimension)?, 8)?,
        DEFAULT_MEMORY_BYTES,
    )?;
    let covariance = sample_covariance(&samples);
    let covariance_rank = rank(&covariance, &samples.defined, analysis.svd_cut);
    Ok(Estimated {
        channel: channel.into(),
        kind,
        estimate: CorrelatorEstimate {
            lags,
            time_unit,
            time_step,
            value: samples
                .defined
                .iter()
                .zip(&samples.central)
                .map(|(defined, x)| defined.then_some(*x))
                .collect(),
            error: errors(&samples),
            samples_meta: SamplesMeta {
                resampling: samples.kind,
                effective_block: samples.effective_block,
                blocks: samples.blocks,
                tau_int: samples.tau_int,
                covariance_rank,
                replicas: measurements.len(),
                sampling_unit: SamplesMeta::sampling_unit(
                    analysis.combine,
                    samples.effective_block,
                    measurements.len(),
                ),
            },
            covariance: Some(covariance),
            connected,
            connected_bias,
        },
        samples,
    })
}

/// Several channels resampled over the same blocks (the largest automatic
/// block of the members), so that `Samples::concat` yields their joint table.
pub fn joint(
    measurements: &[Measurement],
    channels: &[(String, EstimatorKind)],
    analysis: &AnalysisConfig,
) -> Result<Vec<Estimated>> {
    analysis.validate()?;
    require(!channels.is_empty(), "no channel to estimate jointly")?;
    let mut tau: Option<f64> = None;
    for (channel, _) in channels {
        if let Some(member) = series_tau(&members(measurements, channel)?) {
            tau = Some(tau.map_or(member, |largest: f64| largest.max(member)));
        }
    }
    channels
        .iter()
        .map(|(channel, kind)| estimate(measurements, channel, *kind, analysis, tau))
        .collect()
}

/// Frame-mean cross moments of a channel basis for the GEVP. Products of
/// different lags stay inside their origin block. Every member must have the
/// same `components`; a basis that mixes them is a `GasError::Capability`,
/// because the component count belongs to the measurement and not to the
/// configuration, so `analyze` declines that basis instead of failing.
pub fn matrix(
    measurements: &[Measurement],
    channels: &[String],
    analysis: &AnalysisConfig,
) -> Result<MatrixCorrelator> {
    analysis.validate()?;
    require(
        (2..=16).contains(&channels.len()),
        "a correlator matrix needs 2..=16 channels",
    )?;
    let basis = channels
        .iter()
        .map(|channel| members(measurements, channel))
        .collect::<Result<Vec<_>>>()?;
    let components = basis[0][0].1.components;
    for member in &basis {
        retained_weight(member)?;
        let channel = member[0].1;
        if channel.components != components {
            return Err(GasError::Capability(format!(
                "a correlator matrix needs channels of equal components, and channel `{}` has \
                 {} where the first of the basis has {components}",
                channel.id, channel.components
            )));
        }
    }
    let first = &measurements[0];
    let origins: usize = measurements.iter().map(Measurement::frames).sum();
    let tau = basis
        .iter()
        .filter_map(|member| series_tau(member))
        .fold(None, |largest: Option<f64>, tau| {
            Some(largest.map_or(tau, |largest| largest.max(tau)))
        });
    let block = stored_block(&analysis.resampling, tau, origins);
    let mut moments = Vec::with_capacity(channels.len() * channels.len());
    let mut bytes = 0;
    for source in &basis {
        for sink in &basis {
            let parts = source
                .iter()
                .zip(sink)
                .map(|((measurement, a), (_, b))| {
                    cross_moments(
                        measurement.series(a),
                        measurement.series(b),
                        first.config.max_lag,
                        block,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let entry = join(parts, analysis.combine)?;
            bytes = checked_add(bytes, entry.buffer_bytes()?)?;
            enforce(bytes, DEFAULT_MEMORY_BYTES)?;
            moments.push(entry);
        }
    }
    Ok(MatrixCorrelator {
        channels: channels.to_vec(),
        lags: (0..=first.config.max_lag).collect(),
        time_unit: analysis.time_unit,
        time_step: time_step(first, analysis.time_unit)?,
        frames: measurements.iter().map(|m| m.frames() as u64).sum(),
        tau_int: tau,
        moments,
        replicas: measurements.len(),
    })
}

/// The series of one channel in every replica, in replica order. A channel no
/// replica measured, or one a replica is missing, has no estimate.
fn members<'a>(
    measurements: &'a [Measurement],
    channel: &'a str,
) -> Result<Vec<(&'a Measurement, &'a ChannelSeries)>> {
    let members: Vec<(&Measurement, &ChannelSeries)> = measurements
        .iter()
        .filter_map(|measurement| Some((measurement, measurement.channel(channel)?)))
        .collect();
    if members.is_empty() || members.len() != measurements.len() {
        return Err(GasError::Capability(format!(
            "channel `{channel}` was not measured"
        )));
    }
    Ok(members)
}

/// Measured frames that carry a frame average, `T = Σ_t w_t` over segments and
/// replicas. An exchange-odd operator whose elements all cancel on a mutual
/// pairing has none, and so has a channel whose record never allowed the
/// readout: both are a missing observable, not a correlator of zeros.
fn retained_weight(members: &[(&Measurement, &ChannelSeries)]) -> Result<f64> {
    let weight: f64 = members
        .iter()
        .flat_map(|(_, channel)| &channel.weight)
        .sum();
    if weight > 0. {
        return Ok(weight);
    }
    let channel = members[0].1;
    Err(GasError::Capability(match channel.exchange {
        ExchangeParity::Odd => EXCHANGE_ODD_REASON.into(),
        _ => format!(
            "no measured frame of channel `{}` has an average",
            channel.id
        ),
    }))
}

/// Integrated autocorrelation time of the retained frame series, the replicas
/// joined with one segment range each so that no lagged product pairs two
/// runs. `None` without a retained series or without variance.
fn series_tau(members: &[(&Measurement, &ChannelSeries)]) -> Option<f64> {
    let components = members.first()?.1.components;
    let (mut values, mut weight, mut segment) = (vec![], vec![], vec![]);
    for (measurement, channel) in members {
        values.extend_from_slice(&channel.values);
        weight.extend_from_slice(&channel.weight);
        // A running index relabels the replicas' segments into one ascending
        // range. Only neighbouring frames are ever compared, and no replica
        // continues another one's ids into an overflow.
        for (frame, id) in measurement.segment.iter().enumerate() {
            let opens = frame == 0 || *id != measurement.segment[frame - 1];
            let next = segment
                .last()
                .map_or(0, |last: &u32| last.saturating_add(u32::from(opens)));
            segment.push(next);
        }
    }
    tau_int(SeriesView {
        values: &values,
        weight: &weight,
        segment: &segment,
        components,
    })
    .map(|estimate| estimate.tau)
}

/// Time origins per stored block: the resampling block itself, so that the
/// moments are no larger than the resampling they feed, `resample` regroups
/// them one to one, and channels resampled with a common `tau` share their
/// blocks. Without an autocorrelation time the automatic rule takes the
/// uncorrelated one, as `resample` does for a correlator without variance.
fn stored_block(resampling: &Resampling, tau: Option<f64>, origins: usize) -> usize {
    match resampling.block() {
        BlockSize::Fixed { frames } => frames.max(1),
        BlockSize::Auto => auto_block(tau.unwrap_or(0.5), 1, origins),
    }
}

/// One resampling unit per replica under `Combine::RunsAsSamples`: the
/// replicas' moments are merged into one stored block each, and a resampling
/// block of one stored block keeps the deleted or drawn unit a whole run
/// whatever autocorrelation time an automatic block would read from a short
/// one. `PooledBlocks` keeps the configured block.
fn unit_resampling(resampling: Resampling, combine: Combine) -> Resampling {
    if combine != Combine::RunsAsSamples {
        return resampling;
    }
    let block = BlockSize::Fixed { frames: 1 };
    match resampling {
        Resampling::BlockJackknife { .. } => Resampling::BlockJackknife { block },
        Resampling::Bootstrap { samples, seed, .. } => Resampling::Bootstrap {
            block,
            samples,
            seed,
        },
        Resampling::Uncorrelated => Resampling::Uncorrelated,
    }
}

/// `Combine::RunsAsSamples` resamples whole runs, so it needs the 8 runs the
/// configuration demands. Fewer is a missing capability with its reason, not a
/// resampling error over units that do not exist.
fn replica_floor(replicas: usize) -> Result<()> {
    if replicas < 8 {
        return Err(GasError::Capability(format!(
            "runs-as-samples needs at least 8 independently seeded runs, not {replicas}"
        )));
    }
    Ok(())
}

/// Lagged moments of the frame series or the stored propagator of every
/// replica, joined by `analysis.combine`. `tau` chooses the stored block of a
/// frame series only; a propagator arrives blocked by its accumulator.
fn lagged_moments(
    members: &[(&Measurement, &ChannelSeries)],
    kind: EstimatorKind,
    analysis: &AnalysisConfig,
    tau: Option<f64>,
) -> Result<BlockMoments> {
    let parts: Vec<BlockMoments> = match kind {
        EstimatorKind::FrameMean => {
            retained_weight(members)?;
            let origins: usize = members.iter().map(|(m, _)| m.frames()).sum();
            let block = stored_block(&analysis.resampling, tau, origins);
            members
                .iter()
                .map(|(measurement, channel)| {
                    series_moments(
                        measurement.series(channel),
                        measurement.config.max_lag,
                        block,
                    )
                })
                .collect::<Result<_>>()?
        }
        EstimatorKind::SourceFrozen => members
            .iter()
            .map(|(measurement, channel)| {
                let moments = channel.propagator.clone().ok_or_else(|| {
                    GasError::Capability(format!(
                        "channel `{}` has no source-frozen propagator",
                        channel.id
                    ))
                })?;
                // The accumulator stores one entry per frame lag, empty where
                // the lag stride evaluated no sink. A table of another length
                // belongs to another lag range and has no place on this axis.
                let lags = checked_add(measurement.config.max_lag, 1)?;
                if moments.lags != lags {
                    return Err(GasError::Capability(format!(
                        "channel `{}` stores {} propagator lags, not the {lags} of this measurement",
                        channel.id, moments.lags
                    )));
                }
                Ok(moments)
            })
            .collect::<Result<_>>()?,
        EstimatorKind::EuclideanTime => {
            return Err(GasError::Capability(
                "the Euclidean-time slab correlator has no lagged-moment form".into(),
            ));
        }
    };
    join(parts, analysis.combine)
}

/// Blocks of one replica regrouped to the coarsest stored block of the set.
/// Replicas whose stored blocks do not nest cannot be pooled; that is a
/// property of the measurements, so the channel declines instead of failing
/// the analysis.
fn nested_blocks(base: usize, origins_per_block: usize) -> Result<()> {
    if origins_per_block == 0 || !base.is_multiple_of(origins_per_block) {
        return Err(GasError::Capability(format!(
            "replicas store blocks of {origins_per_block} and {base} origins, which do not nest"
        )));
    }
    Ok(())
}

/// The replicas' blocks in one table. `PooledBlocks` concatenates them, the
/// coarser stored block winning; `RunsAsSamples` first merges every replica
/// into a single block, which the resampling then deletes or draws whole.
fn join(mut parts: Vec<BlockMoments>, combine: Combine) -> Result<BlockMoments> {
    // A stored propagator arrives with the measurement and is regrouped by
    // index below, so its shape is checked before it is indexed.
    for part in &parts {
        part.validate()?;
    }
    if combine == Combine::RunsAsSamples {
        replica_floor(parts.len())?;
        for part in &mut parts {
            *part = part.coarsen(part.blocks.max(1))?;
        }
        // The merged block of a replica holds the origins of its run, not the
        // stored block size rounded up to them, so that the reported
        // `effective_block` is the length of a run.
        let base = parts
            .iter()
            .map(BlockMoments::origins)
            .max()
            .unwrap_or(1)
            .max(1);
        for part in &mut parts {
            part.origins_per_block = base;
        }
    } else {
        let base = parts
            .iter()
            .map(|part| part.origins_per_block)
            .max()
            .unwrap_or(1);
        for part in &mut parts {
            nested_blocks(base, part.origins_per_block)?;
            *part = part.coarsen(base / part.origins_per_block)?;
        }
    }
    BlockMoments::concat(&parts.iter().collect::<Vec<_>>())
}

/// Sums of `k` consecutive slab blocks, `BlockMoments::coarsen` for slabs.
fn grouped_slabs(slabs: &SlabMoments, k: usize) -> SlabMoments {
    let k = k.max(1);
    let (square, row) = (slabs.bins * slabs.bins, slabs.bins * slabs.components);
    let blocks = slabs.blocks.div_ceil(k);
    let mut out = SlabMoments {
        bins: slabs.bins,
        components: slabs.components,
        origins_per_block: slabs.origins_per_block.saturating_mul(k),
        blocks,
        pair_ab: vec![0.; blocks * square],
        pair_n: vec![0.; blocks * square],
        profile: vec![0.; blocks * row],
        profile_n: vec![0.; blocks * slabs.bins],
    };
    for block in 0..slabs.blocks {
        let to = block / k;
        for (sum, x) in out.pair_ab[to * square..]
            .iter_mut()
            .zip(&slabs.pair_ab[block * square..(block + 1) * square])
        {
            *sum += x;
        }
        for (sum, x) in out.pair_n[to * square..]
            .iter_mut()
            .zip(&slabs.pair_n[block * square..(block + 1) * square])
        {
            *sum += x;
        }
        for (sum, x) in out.profile[to * row..]
            .iter_mut()
            .zip(&slabs.profile[block * row..(block + 1) * row])
        {
            *sum += x;
        }
        for (sum, x) in out.profile_n[to * slabs.bins..]
            .iter_mut()
            .zip(&slabs.profile_n[block * slabs.bins..(block + 1) * slabs.bins])
        {
            *sum += x;
        }
    }
    out
}

/// The replicas' slab blocks in one table, as `join` does for lagged moments.
fn join_slabs(mut parts: Vec<SlabMoments>, combine: Combine) -> Result<SlabMoments> {
    let first = parts
        .first()
        .ok_or_else(|| GasError::Configuration("no slab moments to join".into()))?;
    let (bins, components) = (first.bins, first.components);
    require(
        parts
            .iter()
            .all(|part| part.bins == bins && part.components == components),
        "joined slab moments must share bins and components",
    )?;
    if combine == Combine::RunsAsSamples {
        replica_floor(parts.len())?;
        for part in &mut parts {
            *part = grouped_slabs(part, part.blocks.max(1));
        }
        let base = parts
            .iter()
            .map(|part| part.origins_per_block)
            .max()
            .unwrap_or(1);
        for part in &mut parts {
            part.origins_per_block = base;
        }
    } else {
        let base = parts
            .iter()
            .map(|part| part.origins_per_block)
            .max()
            .unwrap_or(1);
        for part in &mut parts {
            nested_blocks(base, part.origins_per_block)?;
            *part = grouped_slabs(part, base / part.origins_per_block);
        }
    }
    let mut out = SlabMoments {
        bins,
        components,
        origins_per_block: parts[0].origins_per_block,
        blocks: 0,
        pair_ab: vec![],
        pair_n: vec![],
        profile: vec![],
        profile_n: vec![],
    };
    for part in parts.iter().filter(|part| part.blocks > 0) {
        out.blocks = checked_add(out.blocks, part.blocks)?;
        out.pair_ab.extend_from_slice(&part.pair_ab);
        out.pair_n.extend_from_slice(&part.pair_n);
        out.profile.extend_from_slice(&part.profile);
        out.profile_n.extend_from_slice(&part.profile_n);
    }
    out.validate()?;
    Ok(out)
}

/// Resampled slab correlator, its lags in bins and the bin width. A slab pair
/// contributes `⟨S_b S_b'⟩ − ⟨S_b⟩⟨S_b'⟩` with a mean per bin: on an open axis
/// the slab mean varies along the coordinate, and a subtraction by separation
/// alone would leave the static profile in the correlator. The lag averages
/// the bin pairs that hold data, `None` where none does.
fn euclidean(
    members: &[(&Measurement, &ChannelSeries)],
    analysis: &AnalysisConfig,
    tau: Option<f64>,
) -> Result<(Samples, Vec<usize>, f64)> {
    let first = members[0].0;
    let TimeAxis::Euclidean {
        bins,
        range,
        periodic,
        ..
    } = first.config.time
    else {
        return Err(GasError::Capability(
            "no Euclidean-time axis was measured".into(),
        ));
    };
    let range = range
        .or_else(|| {
            first
                .calibration
                .as_ref()
                .and_then(|calibration| calibration.euclidean_range)
        })
        .ok_or_else(|| {
            GasError::Capability("the Euclidean-time range was not calibrated".into())
        })?;
    let width = (range[1] - range[0]) / bins as f64;
    require(
        width.is_finite() && width > 0.,
        "euclidean slabs need a positive bin width",
    )?;
    let parts = members
        .iter()
        .map(|(_, channel)| {
            channel.euclidean.clone().ok_or_else(|| {
                GasError::Capability(format!(
                    "channel `{}` has no Euclidean-time slabs",
                    channel.id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for part in &parts {
        part.validate()?;
    }
    require(
        parts.iter().all(|part| part.bins == bins),
        "measured slabs disagree with the configured bin count",
    )?;
    let slabs = join_slabs(parts, analysis.combine)?;
    let square = checked_mul(bins, bins)?;
    let row = checked_mul(bins, slabs.components)?;
    // Every resample sums the kept blocks into these four tables.
    enforce(
        checked_mul(
            checked_add(checked_mul(square, 2)?, checked_add(row, bins)?)?,
            8,
        )?,
        DEFAULT_MEMORY_BYTES,
    )?;
    let components = slabs.components;
    let connected = analysis.connected;
    let lags: Vec<usize> = (0..=first.config.max_lag.min(bins - 1)).collect();
    let statistic = |multiplicity: &[f64]| {
        let (mut ab, mut pairs) = (vec![0.; square], vec![0.; square]);
        let (mut profile, mut frames) = (vec![0.; row], vec![0.; bins]);
        for (block, &m) in multiplicity
            .iter()
            .enumerate()
            .take(slabs.blocks)
            .filter(|(_, m)| **m != 0.)
        {
            for (sum, x) in ab.iter_mut().zip(&slabs.pair_ab[block * square..]) {
                *sum += m * x;
            }
            for (sum, x) in pairs.iter_mut().zip(&slabs.pair_n[block * square..]) {
                *sum += m * x;
            }
            for (sum, x) in profile.iter_mut().zip(&slabs.profile[block * row..]) {
                *sum += m * x;
            }
            for (sum, x) in frames.iter_mut().zip(&slabs.profile_n[block * bins..]) {
                *sum += m * x;
            }
        }
        lags.iter()
            .map(|&lag| {
                let (mut sum, mut held) = (0., 0usize);
                for b in 0..bins {
                    let other = if periodic { (b + lag) % bins } else { b + lag };
                    if other >= bins {
                        break;
                    }
                    let entry = b.min(other) * bins + b.max(other);
                    if pairs[entry] <= 0. || (connected && frames[b] * frames[other] <= 0.) {
                        continue;
                    }
                    sum += ab[entry] / pairs[entry];
                    if connected {
                        sum -= (0..components)
                            .map(|k| {
                                (profile[b * components + k] / frames[b])
                                    * (profile[other * components + k] / frames[other])
                            })
                            .sum::<f64>();
                    }
                    held += 1;
                }
                (held > 0).then(|| sum / held as f64)
            })
            .collect()
    };
    let samples = resample_blocks(
        slabs.blocks,
        slabs.origins_per_block,
        checked_mul(slabs.blocks, slabs.origins_per_block)?,
        &unit_resampling(analysis.resampling, analysis.combine),
        tau.or(Some(0.5)),
        &statistic,
    )?;
    Ok((samples, lags, width))
}

/// Lag unit of an algorithm-time estimator: one frame, or the `stride · dt` of
/// the integrator. A run whose kinetic operator has no time step cannot be
/// read in `StepDt`.
fn time_step(measurement: &Measurement, unit: TimeUnit) -> Result<f64> {
    match unit {
        TimeUnit::Frames => Ok(1.),
        TimeUnit::StepDt => {
            let dt = measurement
                .calibration
                .as_ref()
                .and_then(|calibration| calibration.dt)
                .ok_or_else(|| {
                    GasError::Capability("this run has no recorded integrator time step".into())
                })?;
            let step = measurement.config.stride as f64 * dt;
            require(
                step.is_finite() && step > 0.,
                "a lag in integrator time needs a positive recorded time step",
            )?;
            Ok(step)
        }
        TimeUnit::Coordinate => Err(GasError::Capability(
            "the coordinate time unit belongs to the Euclidean-time estimator".into(),
        )),
    }
}

/// Rank of the whitened covariance over the defined lags: the number of
/// directions a correlated fit can use before the SVD floor. A covariance the
/// whitener rejects, a defined lag without variance among them, has rank 0.
fn rank(covariance: &[f64], defined: &[bool], svd_cut: f64) -> usize {
    let kept: Vec<usize> = (0..defined.len()).filter(|&lag| defined[lag]).collect();
    if kept.is_empty() {
        return 0;
    }
    Whitener::new(
        &submatrix(covariance, defined.len(), &kept),
        kept.len(),
        svd_cut,
    )
    .map_or(0, |whitener| whitener.rank())
}
