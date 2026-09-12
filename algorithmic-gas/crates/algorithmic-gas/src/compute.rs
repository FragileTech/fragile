//! Private Burn adapters behind a small batched expression API. The initial
//! engine is host-orchestrated: every accelerator upload/readback is explicit
//! and counted. This is not a claim of a device-resident selection loop.
use crate::{GasError, Precision, Real, Result, TensorBatch};
use burn::tensor::{DType, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

async fn guarded<T>(future: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use futures_util::FutureExt;
        std::panic::AssertUnwindSafe(future)
            .catch_unwind()
            .await
            .map_err(|panic| {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("backend initialization/kernel panic");
                GasError::Execution(message.into())
            })?
    }
    #[cfg(target_arch = "wasm32")]
    {
        future.await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub evaluations: u64,
    pub synchronizations: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
    pub peak_batch_elements: usize,
}

#[derive(Clone, Debug)]
pub enum Unary {
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Abs,
    Square,
    SumRows,
    Clamp(f64, f64),
    Power(f64),
}
#[derive(Clone, Debug)]
pub enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Matmul,
}
#[derive(Clone, Debug)]
pub enum Node {
    Input(usize),
    Scalar(f64),
    Unary(Unary, usize),
    Binary(Binary, usize, usize),
    Columns {
        source: usize,
        start: usize,
        end: usize,
    },
    Gather {
        source: usize,
        indices: Vec<u32>,
    },
    ConcatColumns(Vec<usize>),
}
/// Topologically ordered batch graph. Integer node references are not tensors.
/// Every node is shape-checked before Burn receives it. Last node is the output.
#[derive(Clone, Debug, Default)]
pub struct Expression {
    pub nodes: Vec<Node>,
}
impl Expression {
    pub fn push(&mut self, node: Node) -> usize {
        let i = self.nodes.len();
        self.nodes.push(node);
        i
    }
    pub fn input(&mut self, index: usize) -> usize {
        self.push(Node::Input(index))
    }
    pub fn scalar(&mut self, value: f64) -> usize {
        self.push(Node::Scalar(value))
    }
    pub fn unary(&mut self, op: Unary, x: usize) -> usize {
        self.push(Node::Unary(op, x))
    }
    pub fn binary(&mut self, op: Binary, x: usize, y: usize) -> usize {
        self.push(Node::Binary(op, x, y))
    }
    fn validate<T: Real>(&self, inputs: &[TensorBatch<T>]) -> Result<Vec<[usize; 2]>> {
        for x in inputs {
            x.validate()?;
        }
        let mut shapes: Vec<[usize; 2]> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let get = |i: usize| {
                shapes.get(i).copied().ok_or_else(|| {
                    GasError::Shape("expression must reference an earlier node".into())
                })
            };
            let shape = match node {
                Node::Input(i) => {
                    let t = inputs
                        .get(*i)
                        .ok_or_else(|| GasError::Shape("missing expression input".into()))?;
                    [t.rows(), t.width()]
                }
                Node::Scalar(v) => {
                    if !v.is_finite() {
                        return Err(GasError::Numerical("nonfinite scalar".into()));
                    }
                    [1, 1]
                }
                Node::Unary(op, i) => {
                    let [n, d] = get(*i)?;
                    match op {
                        Unary::SumRows => [n, 1],
                        Unary::Clamp(a, b) if !a.is_finite() || !b.is_finite() || a > b => {
                            return Err(GasError::Configuration("invalid clamp".into()));
                        }
                        Unary::Power(p) if !p.is_finite() => {
                            return Err(GasError::Configuration("invalid power".into()));
                        }
                        _ => [n, d],
                    }
                }
                Node::Binary(op, i, j) => {
                    let a = get(*i)?;
                    let b = get(*j)?;
                    if matches!(op, Binary::Matmul) {
                        if a[1] != b[0] {
                            return Err(GasError::Shape("matmul inner dimension".into()));
                        }
                        [a[0], b[1]]
                    } else {
                        let mut s = [0; 2];
                        for k in 0..2 {
                            if a[k] != b[k] && a[k] != 1 && b[k] != 1 {
                                return Err(GasError::Shape("incompatible broadcast".into()));
                            }
                            s[k] = a[k].max(b[k]);
                        }
                        s
                    }
                }
                Node::Columns { source, start, end } => {
                    let [n, d] = get(*source)?;
                    if start >= end || *end > d {
                        return Err(GasError::Shape("column slice".into()));
                    }
                    [n, end - start]
                }
                Node::Gather { source, indices } => {
                    let [n, d] = get(*source)?;
                    if indices.is_empty() || indices.iter().any(|&i| i as usize >= n) {
                        return Err(GasError::Shape("gather indices".into()));
                    }
                    [indices.len(), d]
                }
                Node::ConcatColumns(ids) => {
                    let first = ids
                        .first()
                        .ok_or_else(|| GasError::Shape("empty concatenation".into()))?;
                    let n = get(*first)?[0];
                    let mut d = 0usize;
                    for i in ids {
                        let s = get(*i)?;
                        if s[0] != n {
                            return Err(GasError::Shape("concatenation rows".into()));
                        }
                        d = d
                            .checked_add(s[1])
                            .ok_or_else(|| GasError::Shape("concatenation overflow".into()))?;
                    }
                    [n, d]
                }
            };
            if shape[0]
                .checked_mul(shape[1])
                .is_none_or(|n| n > i32::MAX as usize)
            {
                return Err(GasError::Shape("expression exceeds index limit".into()));
            }
            shapes.push(shape);
        }
        if shapes.is_empty() {
            return Err(GasError::Shape("empty expression".into()));
        }
        Ok(shapes)
    }
}

#[allow(async_fn_in_trait)]
pub trait ComputeBackend {
    fn kind(&self) -> BackendKind;
    fn precision(&self) -> Precision;
    async fn evaluate<T: Real>(
        &mut self,
        expression: &Expression,
        inputs: &[TensorBatch<T>],
    ) -> Result<TensorBatch<T>>;
}

#[derive(Clone, Debug)]
enum Device {
    #[cfg(feature = "cpu")]
    Cpu(burn::backend::flex::FlexDevice),
    #[cfg(feature = "wgpu")]
    Wgpu(burn::backend::wgpu::WgpuDevice),
    #[cfg(feature = "cuda")]
    Cuda(burn::backend::cuda::CudaDevice),
}
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    /// Opt-in, bounded lecture diagnostics; never consumes random draws.
    pub(crate) stage_trace: Option<Vec<serde_json::Value>>,
    pub(crate) recorded_stages: Option<Vec<crate::tracking::StageSnapshot>>,
    pub(crate) recorded_noise: Option<Vec<crate::tracking::NoiseSnapshot>>,
    pub(crate) recorded_fields: Option<Vec<crate::tracking::FieldEvaluation>>,
    pub(crate) recorded_influences: Option<Vec<crate::tracking::InfluenceRecord>>,
    kind: BackendKind,
    precision: Precision,
    device: Device,
    pub stats: ExecutionStats,
    pub max_batch_elements: usize,
    /// Remaining engine allowance for this batch's graph and transfer buffers.
    pub max_memory_bytes: usize,
}
impl ExecutionContext {
    pub fn record_influence(
        &mut self,
        stage: &str,
        field: &str,
        recipient: u32,
        source: crate::donor::SourceRef,
        weight: f64,
    ) {
        if let Some(records) = &mut self.recorded_influences {
            records.push(crate::tracking::InfluenceRecord {
                stage: stage.into(),
                field: field.into(),
                recipient,
                source,
                weight,
            });
        }
    }

    pub fn record_field<T: Real>(
        &mut self,
        stage: &str,
        field: &str,
        version: u64,
        value: &crate::TensorBatch<T>,
    ) {
        self.record_field_with_coverage(stage, field, version, value, &[]);
    }
    pub fn record_field_with_coverage<T: Real>(
        &mut self,
        stage: &str,
        field: &str,
        version: u64,
        value: &crate::TensorBatch<T>,
        available: &[bool],
    ) {
        if let Some(records) = &mut self.recorded_fields {
            records.push(crate::tracking::FieldEvaluation {
                stage: stage.into(),
                field: field.into(),
                version,
                rows: value.rows(),
                item_shape: value.item_shape().to_vec(),
                values: value.values().iter().map(|x| x.to_f64()).collect(),
                available: if available.is_empty() {
                    vec![true; value.rows()]
                } else {
                    available.to_vec()
                },
            });
        }
    }
    pub fn record_noise<T: Real>(
        &mut self,
        stage: &str,
        request: crate::noise::NoiseRequest,
        innovation: &crate::TensorBatch<T>,
        factor: Option<Vec<f64>>,
    ) {
        if let Some(records) = &mut self.recorded_noise {
            records.push(crate::tracking::NoiseSnapshot {
                stage: stage.into(),
                step: request.step,
                stream: request.stream,
                substep: request.substep,
                rows: request.rows,
                dimension: request.dimension,
                sample: innovation.values().iter().map(|x| x.to_f64()).collect(),
                raw_innovation: None,
                factor,
                geometry: None,
            });
        }
    }

    pub(crate) fn trace_population<T: Real>(&mut self, stage: &str, p: &crate::Population<T>) {
        if let Some(stages) = &mut self.recorded_stages {
            stages.push(crate::tracking::StageSnapshot::capture(stage, p));
        }
        if let Some(trace) = &mut self.stage_trace
            && trace.len() < 16
        {
            trace.push(serde_json::json!({"stage":stage,"population":p}));
        }
    }
    pub async fn new(kind: BackendKind, precision: Precision) -> Result<Self> {
        if kind == BackendKind::Wgpu && precision != Precision::F32 {
            return Err(GasError::Capability(
                "WGPU/WebGPU profile requires f32; select CPU explicitly for f64".into(),
            ));
        }
        let device = guarded(async {Ok(match kind {
            #[cfg(feature = "cpu")]
            BackendKind::Cpu => Device::Cpu(Default::default()),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => {
                let device = Default::default();
                let setup=burn::backend::wgpu::init_setup_async::<
                    burn::backend::wgpu::graphics::AutoGraphicsApi,
                >(&device, Default::default())
                .await;
                if format!("{:?}",setup.adapter.get_info().device_type)=="Cpu" {return Err(GasError::Capability("WGPU selected a software CPU adapter; choose CPU explicitly rather than treating this as GPU execution".into()));}
                Device::Wgpu(device)
            }
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => {
                let device = Default::default();
                let dtype = if precision == Precision::F64 {
                    DType::F64
                } else {
                    DType::F32
                };
                if !burn::backend::Cuda::<f32, i32>::supports_dtype(&device, dtype) {
                    return Err(GasError::Capability(
                        "CUDA device lacks requested arithmetic dtype".into(),
                    ));
                }
                Device::Cuda(device)
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(GasError::Capability(format!(
                    "{kind:?} adapter was not compiled into this build"
                )));
            }
        })}).await?;
        let mut context = Self {
            kind,
            precision,
            device,
            stats: ExecutionStats::default(),
            stage_trace: None,
            recorded_stages: None,
            recorded_noise: None,
            recorded_fields: None,
            recorded_influences: None,
            max_batch_elements: 16_777_216,
            max_memory_bytes: crate::memory::DEFAULT_MEMORY_BYTES,
        };
        if kind != BackendKind::Cpu {
            match precision {
                Precision::F32 => context.probe::<f32>().await?,
                Precision::F64 => context.probe::<f64>().await?,
            }
        }
        Ok(context)
    }
    /// Verify the primitive family required by the built-in operators. A device
    /// dtype declaration alone is insufficient, especially for CUDA f64.
    async fn probe<T: Real>(&mut self) -> Result<()> {
        let data = TensorBatch::vectors(
            2,
            2,
            vec![
                T::from_f64(0.25),
                T::from_f64(0.5),
                T::from_f64(0.75),
                T::ONE,
            ],
        )?;
        let mut e = Expression::default();
        let x = e.input(0);
        let one = e.scalar(1.);
        let exp = e.unary(Unary::Exp, x);
        let log = e.unary(Unary::Log, exp);
        let square = e.unary(Unary::Square, log);
        let root = e.unary(Unary::Sqrt, square);
        let sin = e.unary(Unary::Sin, root);
        let cos = e.unary(Unary::Cos, root);
        let sum = e.binary(Binary::Add, sin, cos);
        let shifted = e.binary(Binary::Add, sum, one);
        let pow = e.unary(Unary::Power(0.5), shifted);
        let clamped = e.unary(Unary::Clamp(0., 2.), pow);
        let divided = e.binary(Binary::Divide, clamped, shifted);
        let gathered = e.push(Node::Gather {
            source: divided,
            indices: vec![1, 0],
        });
        let product = e.binary(Binary::Matmul, gathered, x);
        e.unary(Unary::SumRows, product);
        let output = self.evaluate(&e, &[data]).await?;
        if output.values().iter().any(|x| !x.is_finite()) {
            return Err(GasError::Capability(
                "required primitive probe produced nonfinite values".into(),
            ));
        }
        Ok(())
    }
}
impl ComputeBackend for ExecutionContext {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn precision(&self) -> Precision {
        self.precision
    }
    async fn evaluate<T: Real>(
        &mut self,
        expression: &Expression,
        inputs: &[TensorBatch<T>],
    ) -> Result<TensorBatch<T>> {
        if T::PRECISION != self.precision {
            return Err(GasError::Capability(
                "batch precision differs from execution policy".into(),
            ));
        }
        let shapes = expression.validate(inputs)?;
        let peak = shapes.iter().map(|s| s[0] * s[1]).max().unwrap_or(0);
        // Count all live graph nodes, not only the largest one. Reserve a
        // factor of three for host staging, adapter tensors and readback.
        let mut elements = 0;
        for shape in &shapes {
            elements = crate::memory::checked_add(
                elements,
                crate::memory::checked_mul(shape[0], shape[1])?,
            )?;
        }
        for input in inputs {
            elements = crate::memory::checked_add(elements, input.values().len())?;
        }
        let bytes = crate::memory::checked_mul(elements, std::mem::size_of::<T>() * 3)?;
        crate::memory::enforce(bytes, self.max_memory_bytes)?;
        if peak > self.max_batch_elements {
            return Err(GasError::Capability(
                "batch memory limit exceeded; reduce population or tile size".into(),
            ));
        }
        let result = guarded(async {
            match &self.device {
                #[cfg(feature = "cpu")]
                Device::Cpu(d) => execute::<burn::backend::Flex, T>(d, expression, inputs).await,
                #[cfg(feature = "wgpu")]
                Device::Wgpu(d) => {
                    execute::<burn::backend::Wgpu<f32, i32>, T>(d, expression, inputs).await
                }
                #[cfg(feature = "cuda")]
                Device::Cuda(d) => {
                    execute::<burn::backend::Cuda<f32, i32>, T>(d, expression, inputs).await
                }
            }
        })
        .await?;
        self.stats.evaluations += 1;
        self.stats.peak_batch_elements = self.stats.peak_batch_elements.max(peak);
        if self.kind != BackendKind::Cpu {
            let bytes = std::mem::size_of::<T>() as u64;
            self.stats.uploaded_bytes += inputs
                .iter()
                .map(|x| x.values().len() as u64 * bytes)
                .sum::<u64>();
            self.stats.uploaded_bytes += expression
                .nodes
                .iter()
                .map(|x| match x {
                    Node::Gather { indices, .. } => indices.len() as u64 * 4,
                    Node::Scalar(_) => bytes,
                    _ => 0,
                })
                .sum::<u64>();
            self.stats.downloaded_bytes += result.values().len() as u64 * bytes;
            self.stats.synchronizations += 1;
        }
        Ok(result)
    }
}

async fn execute<B: Backend, T: Real>(
    device: &B::Device,
    expression: &Expression,
    inputs: &[TensorBatch<T>],
) -> Result<TensorBatch<T>> {
    let dtype = if T::PRECISION == Precision::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let mut nodes: Vec<Tensor<B, 2>> = Vec::with_capacity(expression.nodes.len());
    for node in &expression.nodes {
        let tensor = match node {
            Node::Input(i) => {
                let t = &inputs[*i];
                let shape = [t.rows(), t.width()];
                let data = match T::PRECISION {
                    Precision::F32 => TensorData::new(
                        t.values()
                            .iter()
                            .map(|x| x.to_f64() as f32)
                            .collect::<Vec<_>>(),
                        shape,
                    ),
                    Precision::F64 => TensorData::new(
                        t.values().iter().map(|x| x.to_f64()).collect::<Vec<_>>(),
                        shape,
                    ),
                };
                Tensor::<B, 2>::from_data(data, (device, dtype))
            }
            Node::Scalar(x) => Tensor::full([1, 1], *x, (device, dtype)),
            Node::Unary(op, i) => {
                let x = nodes[*i].clone();
                match op {
                    Unary::Neg => -x,
                    Unary::Exp => x.exp(),
                    Unary::Log => x.log(),
                    Unary::Sqrt => x.sqrt(),
                    Unary::Sin => x.sin(),
                    Unary::Cos => x.cos(),
                    Unary::Abs => x.abs(),
                    Unary::Square => x.clone() * x,
                    Unary::SumRows => x.sum_dim(1),
                    Unary::Clamp(a, b) => x.clamp(*a, *b),
                    Unary::Power(p) => x.powf_scalar(*p),
                }
            }
            Node::Binary(op, i, j) => {
                let a = nodes[*i].clone();
                let b = nodes[*j].clone();
                match op {
                    Binary::Add => a + b,
                    Binary::Subtract => a - b,
                    Binary::Multiply => a * b,
                    Binary::Divide => a / b,
                    Binary::Matmul => a.matmul(b),
                }
            }
            Node::Columns { source, start, end } => {
                let x = nodes[*source].clone();
                let n = x.dims()[0];
                x.slice([0..n, *start..*end])
            }
            Node::Gather { source, indices } => {
                let data = TensorData::new(
                    indices.iter().map(|&i| i as i32).collect::<Vec<_>>(),
                    [indices.len()],
                );
                let index =
                    Tensor::<B, 1, burn::tensor::Int>::from_data(data, (device, DType::I32));
                nodes[*source].clone().select(0, index)
            }
            Node::ConcatColumns(ids) => {
                Tensor::cat(ids.iter().map(|&i| nodes[i].clone()).collect(), 1)
            }
        };
        nodes.push(tensor);
    }
    let output = nodes
        .pop()
        .ok_or_else(|| GasError::Shape("empty graph".into()))?;
    let [n, d] = output.dims();
    let data = output
        .into_data_async()
        .await
        .map_err(|e| GasError::Execution(format!("{e:?}")))?;
    let values = match T::PRECISION {
        Precision::F32 => data
            .as_slice::<f32>()
            .map_err(|e| GasError::Execution(format!("{e:?}")))?
            .iter()
            .map(|&x| T::from_f64(x as f64))
            .collect(),
        Precision::F64 => data
            .as_slice::<f64>()
            .map_err(|e| GasError::Execution(format!("{e:?}")))?
            .iter()
            .map(|&x| T::from_f64(x))
            .collect(),
    };
    TensorBatch::vectors(n, d, values)
}
