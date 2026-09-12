#![no_main]
use algorithmic_gas::{Checkpoint, Population, TensorBatch};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    if let Ok(t) = serde_json::from_slice::<TensorBatch<f64>>(data) {
        t.validate().unwrap();
        assert!(t.row(t.rows()).is_err());
        assert!(t.gather(&[u32::MAX]).is_err());
        t.row(0).unwrap();
    }
    if let Ok(p) = serde_json::from_slice::<Population<f32>>(data) {
        if p.validate().is_ok() {
            for t in p.observations.fields.values() {
                t.row(0).unwrap();
            }
        }
    }
    if let Ok(saved) = Checkpoint::<f64>::from_bytes(data) {
        saved.validate().unwrap();
        saved.population.validate().unwrap();
    }
});
