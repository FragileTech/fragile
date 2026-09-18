//! Data-parallel shim. Every helper returns exactly what its serial loop returns:
//! each output element is a pure function of its index, and no floating-point
//! reduction happens across threads. wasm32 and `--no-default-features` builds
//! compile the serial bodies only.
use crate::Result;
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Below this many items a parallel dispatch costs more than it saves.
pub const PAR_MIN_ITEMS: usize = 2048;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parallelism {
    /// Use the thread pool when the build has one and the batch is large enough.
    #[default]
    Auto,
    /// Force the serial path, e.g. to verify bit equality.
    Serial,
    /// Use the thread pool regardless of batch size.
    Always,
}
impl Parallelism {
    #[cfg_attr(
        not(all(feature = "parallel", not(target_arch = "wasm32"))),
        allow(dead_code)
    )]
    fn active(self, items: usize) -> bool {
        match self {
            Self::Auto => items >= PAR_MIN_ITEMS,
            Self::Serial => false,
            Self::Always => items > 1,
        }
    }
}
/// True when this build can run the helpers on more than one thread.
pub const fn available() -> bool {
    cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
}

pub fn map_indexed<R: Send>(n: usize, par: Parallelism, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if par.active(n) {
        return (0..n).into_par_iter().map(&f).collect();
    }
    let _ = par;
    (0..n).map(f).collect()
}
/// The error with the lowest index wins, as in the serial loop.
pub fn try_map_indexed<R: Send>(
    n: usize,
    par: Parallelism,
    f: impl Fn(usize) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    let mut out = Vec::with_capacity(n);
    for r in map_indexed(n, par, f) {
        out.push(r?);
    }
    Ok(out)
}
/// Fill fixed-width rows of `out`; `f(row, slice)` owns its slice exclusively.
pub fn fill_rows<T: Send>(
    out: &mut [T],
    width: usize,
    par: Parallelism,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    if width == 0 {
        return;
    }
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if par.active(out.len() / width) {
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
        return;
    }
    let _ = par;
    for (i, row) in out.chunks_mut(width).enumerate() {
        f(i, row);
    }
}
/// Fill CSR-shaped rows: row `i` is `out[offsets[i]..offsets[i + 1]]`.
pub fn fill_ragged<T: Send>(
    out: &mut [T],
    offsets: &[u32],
    par: Parallelism,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let rows = offsets.len().saturating_sub(1);
    let mut slices = Vec::with_capacity(rows);
    let mut rest = out;
    for i in 0..rows {
        let (head, tail) = rest.split_at_mut((offsets[i + 1] - offsets[i]) as usize);
        slices.push(head);
        rest = tail;
    }
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if par.active(rows) {
        slices
            .into_par_iter()
            .enumerate()
            .for_each(|(i, row)| f(i, row));
        return;
    }
    let _ = par;
    for (i, row) in slices.into_iter().enumerate() {
        f(i, row);
    }
}
/// Coarse tasks (replicas, time slices): parallel whenever there is more than one.
pub fn map_tasks<R: Send>(tasks: usize, par: Parallelism, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let par = match par {
        Parallelism::Serial => Parallelism::Serial,
        _ => Parallelism::Always,
    };
    map_indexed(tasks, par, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn helpers_match_serial_loops() {
        let n = 5000;
        let f = |i: usize| ((i as f64) * 0.37).sin();
        let serial = map_indexed(n, Parallelism::Serial, f);
        let auto = map_indexed(n, Parallelism::Always, f);
        assert!(
            serial
                .iter()
                .zip(&auto)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let mut rows = vec![0.; n * 3];
        fill_rows(&mut rows, 3, Parallelism::Always, |i, r| {
            for (k, x) in r.iter_mut().enumerate() {
                *x = f(i) + k as f64;
            }
        });
        assert_eq!(rows[3 * 17 + 2].to_bits(), (f(17) + 2.).to_bits());
        let offsets: Vec<u32> = (0..=n as u32).map(|i| i * (i + 1) / 2 % 7 + i).collect();
        let mut offsets = offsets;
        offsets.sort_unstable();
        offsets[0] = 0;
        let mut ragged = vec![0usize; *offsets.last().unwrap() as usize];
        fill_ragged(&mut ragged, &offsets, Parallelism::Always, |i, r| {
            r.fill(i + 1)
        });
        for i in 0..n {
            assert!(
                ragged[offsets[i] as usize..offsets[i + 1] as usize]
                    .iter()
                    .all(|&v| v == i + 1)
            );
        }
    }
    #[test]
    fn lowest_index_error_wins() {
        let e = try_map_indexed(4000, Parallelism::Always, |i| {
            if i % 1000 == 999 {
                Err(crate::GasError::Numerical(format!("{i}")))
            } else {
                Ok(i)
            }
        })
        .unwrap_err();
        assert!(e.to_string().ends_with("999"));
    }
}
