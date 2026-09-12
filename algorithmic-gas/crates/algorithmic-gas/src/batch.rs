use crate::{GasError, Real, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Row-major [walkers, ...item_shape]. Images keep their logical shape;
/// a distance must explicitly select and interpret a named field.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(bound = "T: Real")]
pub struct TensorBatch<T: Real> {
    rows: usize,
    item_shape: Vec<usize>,
    values: Vec<T>,
}
impl<'de, T: Real> Deserialize<'de> for TensorBatch<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(bound = "T: Real", deny_unknown_fields)]
        struct Wire<T: Real> {
            rows: usize,
            item_shape: Vec<usize>,
            values: Vec<T>,
        }
        let wire = Wire::<T>::deserialize(deserializer)?;
        Self::new(wire.rows, wire.item_shape, wire.values).map_err(serde::de::Error::custom)
    }
}
impl<T: Real> TensorBatch<T> {
    pub fn new(rows: usize, item_shape: Vec<usize>, values: Vec<T>) -> Result<Self> {
        let b = Self {
            rows,
            item_shape,
            values,
        };
        b.validate()?;
        Ok(b)
    }
    pub fn vectors(rows: usize, width: usize, values: Vec<T>) -> Result<Self> {
        Self::new(rows, vec![width], values)
    }
    pub fn scalars(values: Vec<T>) -> Result<Self> {
        Self::new(values.len(), vec![], values)
    }
    pub fn validate(&self) -> Result<()> {
        let width = self
            .item_shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| GasError::Shape("tensor size overflow".into()))?;
        if self.rows == 0 || width == 0 || self.rows.checked_mul(width) != Some(self.values.len()) {
            return Err(GasError::Shape(
                "nonempty [N, ...shape] must match buffer length".into(),
            ));
        }
        if self.rows > i32::MAX as usize || self.values.len() > i32::MAX as usize {
            return Err(GasError::Shape(
                "batch exceeds 32-bit index contract".into(),
            ));
        }
        Ok(())
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn width(&self) -> usize {
        self.item_shape.iter().product()
    }
    pub fn item_shape(&self) -> &[usize] {
        &self.item_shape
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
    pub fn row(&self, i: usize) -> Result<&[T]> {
        if i >= self.rows {
            return Err(GasError::Shape("row index out of range".into()));
        }
        Ok(&self.values[i * self.width()..(i + 1) * self.width()])
    }
    pub fn gather(&self, indices: &[u32]) -> Result<Self> {
        let size = indices
            .len()
            .checked_mul(self.width())
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| GasError::Shape("gather exceeds index limit".into()))?;
        if indices.is_empty() || indices.iter().any(|&i| i as usize >= self.rows) {
            return Err(GasError::Shape("gather indices".into()));
        }
        let mut v = Vec::new();
        v.try_reserve_exact(size)
            .map_err(|e| GasError::Execution(format!("gather allocation: {e}")))?;
        for &i in indices {
            v.extend_from_slice(self.row(i as usize)?);
        }
        Self::new(indices.len(), self.item_shape.clone(), v)
    }
    pub fn replace_row(&mut self, i: usize, v: &[T]) -> Result<()> {
        let w = self.width();
        if i >= self.rows || v.len() != w {
            return Err(GasError::Shape("replacement row".into()));
        }
        self.values[i * w..(i + 1) * w].copy_from_slice(v);
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub input_version: u64,
    pub population_version: u64,
    pub stage: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct ObservationBatch<T: Real> {
    pub fields: BTreeMap<String, TensorBatch<T>>,
    pub provenance: Provenance,
}
impl<T: Real> ObservationBatch<T> {
    pub fn positions(p: TensorBatch<T>) -> Self {
        Self {
            fields: BTreeMap::from([("positions".into(), p)]),
            provenance: Provenance::default(),
        }
    }
    pub fn field(&self, name: &str) -> Result<&TensorBatch<T>> {
        self.fields
            .get(name)
            .ok_or_else(|| GasError::MissingField(name.into()))
    }
    pub fn field_mut(&mut self, name: &str) -> Result<&mut TensorBatch<T>> {
        self.fields
            .get_mut(name)
            .ok_or_else(|| GasError::MissingField(name.into()))
    }
    pub fn validate(&self, n: usize) -> Result<()> {
        if self.fields.is_empty() {
            return Err(GasError::Shape("observations require a field".into()));
        }
        for f in self.fields.values() {
            f.validate()?;
            if f.rows() != n {
                return Err(GasError::Shape("observation walker count".into()));
            }
        }
        Ok(())
    }
    pub fn gather(&self, indices: &[u32]) -> Result<Self> {
        Ok(Self {
            fields: self
                .fields
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.gather(indices)?)))
                .collect::<Result<_>>()?,
            provenance: self.provenance.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct RewardBatch<T: Real> {
    pub raw: Vec<T>,
    pub valid: Vec<bool>,
    pub provenance: Provenance,
}
impl<T: Real> RewardBatch<T> {
    pub fn new(raw: Vec<T>, provenance: Provenance) -> Self {
        let valid = raw.iter().map(|x| x.is_finite()).collect();
        Self {
            raw,
            valid,
            provenance,
        }
    }
    pub fn validate(&self, n: usize) -> Result<()> {
        if self.raw.len() != n || self.valid.len() != n {
            return Err(GasError::Shape(
                "one reward scalar and validity bit per walker required".into(),
            ));
        }
        Ok(())
    }
    pub fn gather(&self, indices: &[u32]) -> Result<Self> {
        self.validate(self.raw.len())?;
        if indices.iter().any(|&i| i as usize >= self.raw.len()) {
            return Err(GasError::Shape("reward gather index".into()));
        }
        Ok(Self {
            raw: indices.iter().map(|&i| self.raw[i as usize]).collect(),
            valid: indices.iter().map(|&i| self.valid[i as usize]).collect(),
            provenance: self.provenance.clone(),
        })
    }
}

/// Shared immutable inputs. Raw RAM/images require an explicit extractor;
/// optional algorithm-state views are supplied through ExtractionContext.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct InputBatch<T: Real> {
    pub rows: usize,
    pub version: u64,
    pub numerical: BTreeMap<String, TensorBatch<T>>,
    pub bytes: BTreeMap<String, Vec<Vec<u8>>>,
}
impl<T: Real> InputBatch<T> {
    pub fn validate(&self) -> Result<()> {
        if self.rows == 0 {
            return Err(GasError::Shape("empty input batch".into()));
        }
        for t in self.numerical.values() {
            t.validate()?;
            if t.rows() != self.rows {
                return Err(GasError::Shape("input row count".into()));
            }
        }
        if self.bytes.values().any(|v| v.len() != self.rows) {
            return Err(GasError::Shape("byte input row count".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validity {
    pub invalid: bool,
    pub out_of_bounds: bool,
    pub terminated: bool,
    pub truncated: bool,
}
impl Validity {
    pub fn eligible(self, include_truncated: bool) -> bool {
        !self.invalid
            && !self.out_of_bounds
            && !self.terminated
            && (!self.truncated || include_truncated)
    }
}

/// Numerical operators never inspect these payloads. Domain adapters own their
/// meaning. Clone operations deep-copy them from the immutable donor snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateStore {
    pub snapshots: Vec<Vec<u8>>,
    pub codec: String,
}
impl StateStore {
    pub fn gather(&self, indices: &[u32]) -> Result<Self> {
        let snapshots = indices
            .iter()
            .map(|&i| {
                self.snapshots
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| GasError::Shape("state donor index".into()))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            snapshots,
            codec: self.codec.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct Population<T: Real> {
    pub observations: ObservationBatch<T>,
    pub rewards: RewardBatch<T>,
    pub validity: Vec<Validity>,
    pub states: Option<StateStore>,
    pub generations: Vec<u64>,
    pub version: u64,
}
impl<T: Real> Population<T> {
    pub fn new(observations: ObservationBatch<T>) -> Result<Self> {
        let n = observations
            .fields
            .values()
            .next()
            .ok_or_else(|| GasError::Shape("empty observations".into()))?
            .rows();
        let p = Self {
            observations,
            rewards: RewardBatch::new(vec![T::ZERO; n], Provenance::default()),
            validity: vec![Validity::default(); n],
            states: None,
            generations: vec![0; n],
            version: 0,
        };
        p.validate()?;
        Ok(p)
    }
    pub fn len(&self) -> usize {
        self.validity.len()
    }
    pub fn is_empty(&self) -> bool {
        self.validity.is_empty()
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        self.observations.validate(n)?;
        self.rewards.validate(n)?;
        if self.generations.len() != n
            || self.states.as_ref().is_some_and(|s| s.snapshots.len() != n)
        {
            return Err(GasError::Shape("population metadata length".into()));
        }
        Ok(())
    }
    pub fn eligible(&self, include_truncated: bool) -> Vec<bool> {
        self.validity
            .iter()
            .map(|s| s.eligible(include_truncated))
            .collect()
    }
}
pub struct ExtractionContext<'a, T: Real> {
    pub population: &'a Population<T>,
    pub stage: &'a str,
}
pub trait ObservationExtractor<T: Real> {
    fn extract(
        &self,
        input: &InputBatch<T>,
        context: &ExtractionContext<'_, T>,
    ) -> Result<ObservationBatch<T>>;
}
pub trait RewardExtractor<T: Real> {
    fn extract(
        &self,
        input: &InputBatch<T>,
        context: &ExtractionContext<'_, T>,
    ) -> Result<RewardBatch<T>>;
}
pub trait DerivedFieldProvider<T: Real> {
    fn evaluate(
        &self,
        input: &InputBatch<T>,
        context: &ExtractionContext<'_, T>,
    ) -> Result<BTreeMap<String, TensorBatch<T>>>;
}
pub trait InputProvider<T: Real> {
    fn acquire(&mut self, population: &Population<T>) -> Result<InputBatch<T>>;
}
