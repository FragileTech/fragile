use algorithmic_gas::{GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::{
    Benchmark, BenchmarkModel, convergence_experiments::ArchiveStore,
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "fragile-proof-archive-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn immutable_partial_progress_and_checksum_detection() {
    let dir = Directory::new();
    let mut store = ArchiveStore::new(&dir.0).unwrap();
    let path = store
        .save_json("../raw seed", &json!({"positions":[[1.,2.]],"noise":[0.3]}))
        .unwrap();
    assert!(!path.contains(".."));
    drop(store);
    assert!(ArchiveStore::new(&dir.0).is_err());
    let mut stored = ArchiveStore::open(&dir.0).unwrap();
    assert_eq!(stored.status(), "recording");
    assert_eq!(stored.load_json(&path).unwrap()["noise"], json!([0.3]));
    assert!(stored.save_json("overwrite", &json!({})).is_err());
    assert!(stored.finish("completed").is_err());
    assert_eq!(stored.verify(true).unwrap()["archives"], 1);
    fs::write(dir.0.join(path), b"modified").unwrap();
    assert!(stored.verify(false).is_err());
}
#[test]
fn untrusted_index_cannot_escape_dataset() {
    let dir = Directory::new();
    let mut store = ArchiveStore::new(&dir.0).unwrap();
    store.save_json("raw", &json!({})).unwrap();
    store.finish("completed").unwrap();
    drop(store);
    let index = dir.0.join("archive-index.json");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&index).unwrap()).unwrap();
    value["entries"][0]["path"] = json!("../outside.gz");
    fs::write(index, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(ArchiveStore::open(&dir.0).is_err());
}
#[test]
fn native_archive_and_checkpoint_survive_compression_and_restore() {
    futures_lite::future::block_on(async {
        let dir = Directory::new();
        let mut store = ArchiveStore::new(&dir.0).unwrap();
        let mut obs = ObservationBatch::positions(
            TensorBatch::vectors(4, 1, vec![-0.3, -0.1, 0.1, 0.3]).unwrap(),
        );
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(4, 1, vec![0.; 4]).unwrap(),
        );
        let mut config = GasConfig::euclidean(1, 0.04).unwrap();
        config.seed = 1729;
        let model = BenchmarkModel {
            benchmark: Benchmark::Quadratic,
            field: "positions".into(),
            direction: config.fitness.direction,
        };
        let mut gas = GasBuilder::new(Population::new(obs).unwrap(), model.clone())
            .gradient(model)
            .config(config)
            .build()
            .await
            .unwrap();
        gas.start_recording(Default::default()).unwrap();
        gas.step().await.unwrap();
        let raw = store
            .save_archive("native", gas.recording().unwrap())
            .unwrap();
        let checkpoint = store.save_checkpoint("resume", &gas.checkpoint()).unwrap();
        let native = store.load_archive(&raw).unwrap();
        assert_eq!(native.steps.len(), 1);
        assert_eq!(
            native.to_bytes().unwrap(),
            gas.recording().unwrap().to_bytes().unwrap()
        );
        let expected = gas.step().await.unwrap();
        let expected_population = gas.population().clone();
        gas.restore(store.load_checkpoint(&checkpoint).unwrap())
            .unwrap();
        let observed = gas.step().await.unwrap();
        assert_eq!(
            serde_json::to_value(expected).unwrap(),
            serde_json::to_value(observed).unwrap()
        );
        assert_eq!(
            serde_json::to_value(expected_population).unwrap(),
            serde_json::to_value(gas.population()).unwrap()
        );
        store.finish("completed").unwrap();
        assert_eq!(store.verify(true).unwrap()["native_recorded_steps"], 1);
    });
}
