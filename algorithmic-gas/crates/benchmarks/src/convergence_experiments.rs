//! Durable experimental data for source-bound convergence validation.
//! Raw records are immutable gzip/CBOR archives; the index is committed after
//! each chunk so a failed experiment retains its completed measurements.
use algorithmic_gas::{Checkpoint, GasError, Result, RunArchive};
use flate2::{Compression, GzBuilder, read::GzDecoder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentConfig {
    pub samples: usize,
    pub steps: usize,
    pub compact: bool,
    pub seed: u64,
    pub archive_chunk_steps: usize,
}
impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            samples: 256,
            steps: 128,
            compact: false,
            seed: 20261004,
            archive_chunk_steps: 32,
        }
    }
}
impl ExperimentConfig {
    pub fn validate(&self) -> Result<()> {
        if self.samples < 16
            || self.samples > 100_000
            || self.steps < 2
            || self.steps > 10_000
            || self.archive_chunk_steps == 0
            || self.archive_chunk_steps > 128
        {
            return Err(error(
                "experiments need 16..100000 independent samples, 2..10000 steps and 1..128 archive chunk steps",
            ));
        }
        Ok(())
    }
}
fn error(message: impl Into<String>) -> GasError {
    GasError::Configuration(message.into())
}
fn io_error(e: impl std::fmt::Display) -> GasError {
    error(e.to_string())
}
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path).map_err(io_error)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub path: String,
    pub tag: String,
    pub kind: String,
    pub sha256: String,
    pub compressed_bytes: u64,
    pub metadata: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArchiveIndex {
    schema_version: u32,
    status: String,
    entries: Vec<ArchiveEntry>,
}
pub struct ArchiveStore {
    root: PathBuf,
    index: ArchiveIndex,
    writable: bool,
}
impl ArchiveStore {
    /// Start an immutable dataset in an empty directory. Re-analysis uses open.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(io_error)?;
        if fs::read_dir(&root).map_err(io_error)?.next().is_some() {
            return Err(error(
                "experiment output directory must be empty; existing datasets are immutable",
            ));
        }
        fs::create_dir(root.join("archives")).map_err(io_error)?;
        let store = Self {
            root,
            index: ArchiveIndex {
                schema_version: 1,
                status: "recording".into(),
                entries: vec![],
            },
            writable: true,
        };
        store.commit_index()?;
        Ok(store)
    }
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let mut index: ArchiveIndex = serde_json::from_reader(BufReader::new(
            File::open(root.join("archive-index.json")).map_err(io_error)?,
        ))
        .map_err(io_error)?;
        if index.schema_version != 1 {
            return Err(error("unsupported experiment archive schema"));
        }
        let journal = root.join("archive-journal.jsonl");
        if journal.exists() {
            let committed = index.entries.len();
            for (position, line) in BufReader::new(File::open(journal).map_err(io_error)?)
                .lines()
                .enumerate()
            {
                if position >= committed {
                    index
                        .entries
                        .push(serde_json::from_str(&line.map_err(io_error)?).map_err(io_error)?);
                }
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for entry in &index.entries {
            let components: Vec<_> = Path::new(&entry.path).components().collect();
            if components.len() != 2
                || components[0] != std::path::Component::Normal("archives".as_ref())
                || !matches!(components[1], std::path::Component::Normal(_))
                || !seen.insert(&entry.path)
            {
                return Err(error("invalid or duplicate indexed archive path"));
            }
        }
        Ok(Self {
            root,
            index,
            writable: false,
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn entries(&self) -> &[ArchiveEntry] {
        &self.index.entries
    }
    pub fn status(&self) -> &str {
        &self.index.status
    }
    fn commit_index(&self) -> Result<()> {
        let temporary = self.root.join("archive-index.json.partial");
        let mut writer = BufWriter::new(File::create(&temporary).map_err(io_error)?);
        serde_json::to_writer_pretty(&mut writer, &self.index).map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        fs::rename(temporary, self.root.join("archive-index.json")).map_err(io_error)
    }
    /// Append/fsync each completed chunk; periodic index snapshots avoid
    /// quadratic rewriting of full configurations in long ensemble datasets.
    fn commit_entry(&self, entry: &ArchiveEntry) -> Result<()> {
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.root.join("archive-journal.jsonl"))
                .map_err(io_error)?,
        );
        serde_json::to_writer(&mut writer, entry).map_err(io_error)?;
        writer.write_all(b"\n").map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        if self.index.entries.len().is_multiple_of(256) {
            self.commit_index()?;
        }
        Ok(())
    }
    fn save<T: Serialize>(
        &mut self,
        tag: &str,
        kind: &str,
        extension: &str,
        value: &T,
        metadata: Value,
        json_format: bool,
    ) -> Result<String> {
        if !self.writable {
            return Err(error(
                "opened datasets are read-only; write derived results to a new directory",
            ));
        }
        let safe: String = tag
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                    c
                } else {
                    '_'
                }
            })
            .take(120)
            .collect();
        let relative = format!(
            "archives/{:06}-{safe}.{extension}.gz",
            self.index.entries.len()
        );
        let path = self.root.join(&relative);
        let temporary = path.with_extension("partial");
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error)?;
        let mut gzip = GzBuilder::new()
            .mtime(0)
            .write(BufWriter::new(file), Compression::fast());
        if json_format {
            serde_json::to_writer(&mut gzip, value).map_err(io_error)?;
        } else {
            ciborium::ser::into_writer(value, &mut gzip).map_err(io_error)?;
        }
        let mut writer = gzip.finish().map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        if path.exists() {
            return Err(error("refusing to overwrite an immutable archive"));
        }
        fs::rename(&temporary, &path).map_err(io_error)?;
        let entry = ArchiveEntry {
            path: relative.clone(),
            tag: tag.into(),
            kind: kind.into(),
            sha256: sha256_file(&path)?,
            compressed_bytes: fs::metadata(&path).map_err(io_error)?.len(),
            metadata,
        };
        self.index.entries.push(entry);
        self.commit_entry(self.index.entries.last().unwrap())?;
        Ok(relative)
    }
    pub fn save_archive(&mut self, tag: &str, archive: &RunArchive<f64>) -> Result<String> {
        archive.validate()?;
        let metadata = json!({"recorded_steps":archive.steps.len(),"epoch":archive.epoch,"first_step":archive.steps.first().map(|s|s.report.step),"last_step":archive.steps.last().map(|s|s.report.step),"native_config":archive.gas_config,"providers":archive.providers,"schema_version":archive.schema_version});
        self.save(tag, "native_run_archive", "cbor", archive, metadata, false)
    }
    pub fn save_checkpoint(&mut self, tag: &str, checkpoint: &Checkpoint<f64>) -> Result<String> {
        checkpoint.validate()?;
        self.save(
            tag,
            "native_checkpoint",
            "cbor",
            checkpoint,
            json!({"step":checkpoint.step,"native_config":checkpoint.config}),
            false,
        )
    }
    pub fn save_json(&mut self, tag: &str, value: &Value) -> Result<String> {
        self.save(tag, "raw_json", "json", value, json!({}), true)
    }
    pub fn finish(&mut self, status: &str) -> Result<()> {
        if !self.writable {
            return Err(error("cannot modify a read-only dataset"));
        }
        self.index.status = status.into();
        self.commit_index()
    }
    fn verified_path(&self, relative: &str) -> Result<PathBuf> {
        let entry = self
            .index
            .entries
            .iter()
            .find(|e| e.path == relative)
            .ok_or_else(|| error("archive path is not in dataset index"))?;
        let path = self.root.join(&entry.path);
        if sha256_file(&path)? != entry.sha256 {
            return Err(error(format!("archive checksum mismatch: {relative}")));
        }
        Ok(path)
    }
    fn bytes(&self, relative: &str) -> Result<Vec<u8>> {
        let path = self.verified_path(relative)?;
        let mut reader = GzDecoder::new(BufReader::new(File::open(path).map_err(io_error)?))
            .take(256 * 1024 * 1024 + 1);
        let mut bytes = vec![];
        reader.read_to_end(&mut bytes).map_err(io_error)?;
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(error("decoded archive exceeds 256MiB; use smaller chunks"));
        }
        Ok(bytes)
    }
    pub fn load_archive(&self, relative: &str) -> Result<RunArchive<f64>> {
        RunArchive::from_bytes(&self.bytes(relative)?)
    }
    pub fn load_checkpoint(&self, relative: &str) -> Result<Checkpoint<f64>> {
        Checkpoint::from_bytes(&self.bytes(relative)?)
    }
    pub fn load_json(&self, relative: &str) -> Result<Value> {
        serde_json::from_slice(&self.bytes(relative)?).map_err(io_error)
    }
    pub fn verify(&self, deep: bool) -> Result<Value> {
        let mut steps = 0usize;
        let mut bytes = 0u64;
        for entry in &self.index.entries {
            self.verified_path(&entry.path)?;
            if deep {
                match entry.kind.as_str() {
                    "native_run_archive" => {
                        let archive = self.load_archive(&entry.path)?;
                        steps += archive.steps.len();
                    }
                    "native_checkpoint" => {
                        self.load_checkpoint(&entry.path)?;
                    }
                    "raw_json" => {
                        self.load_json(&entry.path)?;
                    }
                    _ => return Err(error("unknown archive kind")),
                }
            } else {
                steps += entry.metadata["recorded_steps"].as_u64().unwrap_or(0) as usize;
            }
            bytes += entry.compressed_bytes;
        }
        Ok(
            json!({"status":self.index.status,"archives":self.index.entries.len(),"native_recorded_steps":steps,"compressed_bytes":bytes,"checksums_verified":true,"deep_decode":deep}),
        )
    }
}
