//! Checked admission budgets for engine-owned host buffers and tensor graphs.
//! This is not an OS/driver allocator sandbox; custom providers must cooperate.
use crate::{GasConfig, GasError, InputBatch, Population, Real, Result};

pub const DEFAULT_MEMORY_BYTES: usize = 512 * 1024 * 1024;
pub fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| GasError::Capability("memory accounting overflow".into()))
}
pub fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| GasError::Capability("memory accounting overflow".into()))
}
pub fn enforce(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        Err(GasError::Capability(format!(
            "engine memory budget exceeded: {bytes} > {limit} bytes"
        )))
    } else {
        Ok(())
    }
}
impl<T: Real> Population<T> {
    pub fn buffer_bytes(&self) -> Result<usize> {
        let mut total = checked_mul(
            self.len(),
            std::mem::size_of::<T>() + std::mem::size_of::<crate::Validity>() + 9,
        )?;
        for (name, t) in &self.observations.fields {
            total = checked_add(
                total,
                checked_mul(t.values().len(), std::mem::size_of::<T>())?,
            )?;
            total = checked_add(
                total,
                checked_add(
                    name.len(),
                    checked_mul(t.item_shape().len(), std::mem::size_of::<usize>())?,
                )?,
            )?;
        }
        if let Some(states) = &self.states {
            total = checked_add(total, states.codec.len())?;
            total = checked_add(
                total,
                checked_mul(states.snapshots.len(), std::mem::size_of::<Vec<u8>>())?,
            )?;
            for bytes in &states.snapshots {
                total = checked_add(total, bytes.len())?;
            }
        }
        Ok(total)
    }
}
impl<T: Real> InputBatch<T> {
    pub fn buffer_bytes(&self) -> Result<usize> {
        let mut total = 0;
        for t in self.numerical.values() {
            total = checked_add(
                total,
                checked_mul(t.values().len(), std::mem::size_of::<T>())?,
            )?;
        }
        for rows in self.bytes.values() {
            total = checked_add(
                total,
                checked_mul(rows.len(), std::mem::size_of::<Vec<u8>>())?,
            )?;
            for row in rows {
                total = checked_add(total, row.len())?;
            }
        }
        Ok(total)
    }
}
impl GasConfig {
    /// Conservative reservation for frozen pools, simultaneous destinations,
    /// reports/checkpoint copies and retained frames. Tensor graph scratch is
    /// separately checked against the remaining allowance at execution time.
    pub fn working_set_bytes<T: Real>(
        &self,
        p: &Population<T>,
        history: &[(u64, Population<T>)],
        input: Option<&InputBatch<T>>,
    ) -> Result<usize> {
        let mut largest = p.buffer_bytes()?;
        for (_, old) in history {
            largest = largest.max(old.buffer_bytes()?);
        }
        let window = self
            .distance_donors
            .history_window
            .max(self.cloning_donors.history_window);
        let copies = checked_add(12, checked_mul(window, 4)?)?;
        let mut total = checked_mul(largest, copies)?;
        let edges = checked_mul(
            p.len(),
            checked_add(self.distance_donors.count, self.cloning_donors.count)?,
        )?;
        total = checked_add(total, checked_mul(edges, 128)?)?;
        let source_rows = checked_mul(p.len(), checked_add(window, 1)?)?;
        total = checked_add(total, checked_mul(source_rows, 256)?)?;
        if let Some(input) = input {
            total = checked_add(total, checked_mul(input.buffer_bytes()?, 3)?)?;
        }
        enforce(total, self.max_memory_bytes)?;
        Ok(total)
    }
}
