(sec-algorithmic-gas-architecture)=
# Architecture of Algorithmic Gas

**Status: engineering specification with an initial independent Rust implementation.**
The workspace is `algorithmic-gas/`; its `README.md` documents the implemented API,
build commands and limitations. This chapter defines the target contracts;
signature sketches are not declarations of the current Rust API. See
{ref}`sec-algorithmic-gas-implementation-status` for implementation status.
Algorithmic Gas is a modular Rust engine derived from Euclidean Gas. Existing
Atari, robotics, Fractal Gas, and Euclidean Gas implementations remain unchanged.
“Must” marks a requirement; “may” marks an optional, explicitly configured extension.

Algorithmic Gas owns its API; private Burn adapters provide tensor operations.
Derivatives use explicit providers, with optional automatic-differentiation adapters.
Observation and reward extractors independently consume shared inputs and declared
algorithm state. Their outputs are a tensor observation and one scalar reward per walker.

Separate `distance_donors` and `cloning_donors` instances use the same `DonorModule`
interface. Each defaults to one donor and supports multiple donors; future cloning
extensions can combine several donors. Current eligible walkers are the default
source pool, with optional historical snapshots. Self, kernel, and topology rules
apply; ancestry does not restrict selection by default. Lineage records provenance.

Configurations define different algorithms. The fixed models in
{doc}`../convergence_program/02_euclidean_gas` and
{doc}`../1_the_algorithm/02_fractal_gas_latent` have specific hypotheses, boundary
laws, and sampling distributions. Applying their convergence results requires
verifying that the selected configuration satisfies those conditions.

(sec-algorithmic-gas-components)=
## 1. Component architecture and preparation

Algorithmic Gas separates comparison, donor selection, fitness, cloning, and
kinetics into replaceable components. The engine validates their field, topology,
precision, and execution requirements during preparation. It owns the public API;
Burn remains behind private numerical adapters. A component is usable only when
its requirements are satisfied by the selected data schema and backend.

### 1.1. Public contracts and notation

:::{prf:definition} Algorithmic Gas execution contract
:label: def-algorithmic-gas-execution-contract

A prepared run is a tuple
$\mathcal R=(\mathcal C,\mathcal E,\mathcal I,\mathcal P,\mathcal S,\mathcal Q)$:
resolved component configuration, execution context, admitted input context,
numerical population with observations and raw rewards, optional opaque state
store, and persistent operator/random state. Each step
reads a versioned population snapshot and commits one consistent successor.
No operator may change the sampling law, computation precision, device placement,
or mathematical regularizers without an explicit configuration transition.

Let $N\geq1$ be the fixed number of allocated walker slots, $k$ the current number
of eligible slots, $d$ a selected numerical feature dimension, $m$ a requested pair
count, $K_{\rm dist}$ and $K_{\rm clone}$ the independently configured donor
capacities (both default to one), and $r$ an innovation dimension.
`P` denotes the run's selected `f32` or
`f64` computation precision. Slot indices and Boolean masks are not floating-point
data. A batch's leading dimension is its slot or explicitly declared request axis.
:::

Rust blocks are **signature sketches**, not a compilable crate. Generic bounds,
auxiliary types, and error variants remain to be finalized. `B` is an Algorithmic
Gas backend; `FloatBatch<B>` uses precision `P`; `IndexBatch<B>` and `Mask<B>` use
integer and Boolean storage. `Result<T>` carries a structured `GasError`.

The following rules apply to **every public interface**, including extension
providers:

| Contract dimension | Required declaration and behavior |
|---|---|
| Inputs and outputs | Named fields, ranks, axis order, units, validity masks, and source stage/version; no implicit reshaping or interpretation of images as positions |
| Ownership | `&` inputs are immutable borrows; returned batches own buffers or documented immutable shared handles; `&mut` destinations are exclusively borrowed and cannot alias sources |
| Precision | Floating calculations, constants, factors, reductions, and outputs use `P`; ingestion converts only according to an explicit conversion policy |
| Execution | A `ComponentDescriptor` lists operations, permitted placements, derivative/synchronization needs, supported precisions, and persistent state |
| Randomness | Stochastic calls consume a named addressable stream; draws are not taken from a shared, scheduling-dependent global generator |
| Failure | Schema, unit, shape, dtype, topology, nonfinite-input, stale-version, allocation, unsupported-capability, and domain failures are typed errors, not silent fallback |
| Lifecycle | Preparation checks static compatibility; calls check dynamic invariants; queued failures are surfaced before a successor is published |
| Extensibility | Components provide stable identifiers, schema versions, configuration serialization, and cloning/checkpoint rules for their persistent state |

### 1.2. Component map

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 22, 'rankSpacing': 25}}}%%
flowchart LR
    Builder[GasBuilder] --> Gas[AlgorithmicGas]
    Gas --> Data["Input / observations / reward<br/>Opaque StateStore"]
    Gas --> Selection["distance_donors<br/>cloning_donors"]
    Gas --> Dynamics["Clone transforms<br/>KineticOperator"]
    Dynamics --> Providers["Derivatives / NoiseSource<br/>DomainAdapter"]
    Gas --> Context["ExecutionContext<br/>ComputeBackend"]
    Context --> Burn["Private Burn<br/>adapters"]
    Gas --> History["HistorySink<br/>Checkpoint"]
```

| Public component family | Responsibility | Primary dependency or extension point |
|---|---|---|
| `AlgorithmicGas` / `GasBuilder` | Assemble, prepare, initialize, step, checkpoint | Validated `ComponentSet` and execution request |
| `ComputeBackend` / `ExecutionContext` | Tensor operations, device, precision, scratch, synchronization | Private CPU, CUDA, or WGPU adapter |
| `InputProvider` / `InputBatch` / `DerivedFieldProvider` | Admit external/domain inputs and share declared intermediate evaluations | Versioned input and algorithm-state dependencies |
| `Population` / `ObservationBatch` / `RewardBatch` | First-class observations, scalar raw rewards, and metadata | Schema, immutable typed views, and source provenance |
| `StateStore` | Optional opaque per-slot snapshots and simultaneous gather | Domain-owned snapshot representation |
| `ObservationExtractor` / `RewardExtractor` | Independently derive one tensor observation and one scalar reward per walker | Shared input context; identity or custom extraction |
| `DomainAdapter` / `DomainRecombiner` | Restore, transition, synchronize, and optionally recombine domain state | Simulator or numerical domain; explicit recombination capability |
| `AlgorithmicDistance` | Compare selected pairs or tiles | Explicit observation view and optional periodic domain |
| `InteractionKernel` / `CompanionSampler` | Pair weights; joint companion distribution | Comparison convention and topology |
| `DonorPoolProvider` / `DonorModule` | Freeze eligible current/historical sources; draw through two independent role instances | `distance_donors` and `cloning_donors`; no default ancestry filter |
| `CompanionReducer` | Reduce several scalar pair comparisons to one diversity measurement | Masked mean, weighted mean, or another declared rule |
| `BoundaryPolicy` | Classify validity and optionally repair | Shared domain description and external flags |
| `ObjectiveOrientation` / `FitnessPipeline` | Maximize/minimize raw reward; standardize, map, and combine measurements | Independent objective direction and channel providers |
| `CloneProposal` / `CloneDecision` / `CloneTransform` | Construct candidate replacements, accept them, and copy or recombine immutable sources | Generalized `ClonePlan`, multi-donor lineage, and topology certificate |
| `KineticOperator` | Advance selected numerical or environment states | Drift, derivatives, diffusion, independent noise |
| `InnovationLaw` / `NoiseGeometry` / `NoiseSource` | Base innovations, spatial factors, factored noise batches | Distribution moments and factor providers |
| `HistorySink` / `Checkpoint` | Stage-aware observations, lineage, replay state | Storage and domain serialization adapters |

### 1.3. Assembly and execution lifecycle

The assembly API accepts a complete component graph. `ComponentSet<B>` contains
input/derived-field providers, observation and reward extractors, objective
orientation, two distinct donor-module instances, diversity reducer, channel
pipelines, boundary policy, clone proposal/decision/transform, kinetic operator,
domain adapters, and recording policy. Immutable distance/kernel implementations
may be shared, but role-specific configuration, mutable sampler state, and random
namespaces are independent. There is no single global donor-count setting.

```rust
pub trait GasBuilder<B: ComputeBackend>: Sized {
    fn components(self, components: ComponentSet<B>) -> Self;
    fn distance_donors(self, module: DonorModuleConfig<B>) -> Self;
    fn cloning_donors(self, module: DonorModuleConfig<B>) -> Self;
    fn execution(self, request: ExecutionRequest) -> Self;
    fn policies(self, policies: RunPolicies) -> Self;
    async fn prepare(self) -> Result<PreparedGas<B>>;
}

pub trait AlgorithmicGas<B: ComputeBackend> {
    async fn initialize(&mut self, initial: InitialPopulation<B>) -> Result<StepReport>;
    async fn step(&mut self) -> Result<StepReport>;
    async fn step_with_input(&mut self, input: InputBatch<B>) -> Result<StepReport>;
    async fn run_steps(&mut self, budget: StepBudget) -> Result<BatchReport>;
    fn cancellation_handle(&self) -> CancellationHandle;
    async fn checkpoint(&mut self) -> Result<Checkpoint>;
    async fn restore(&mut self, checkpoint: Checkpoint) -> Result<RestoreReport>;
}

pub trait CancellationHandle {
    fn request_cancel(&self);
}
```

| Method | Lifecycle contract |
|---|---|
| `prepare` | Own configuration; validate schemas, units, topology, and required device operations; specialize kernels; reserve bounded memory. Return the resolved configuration without advancing the population. Reject detectable incompatibilities. |
| `initialize` | Consume initial `[N,...]` input and optional `N` snapshots; run extractors and commit the first version. Pre-extracted imports require matching provenance. Missing required fields, reward semantics, or state are errors. |
| `step_with_input` | Consume an external batch for the requested step. |
| `step` | Acquire input through the configured provider/domain adapter. Missing external input returns `InputRequired`. Return counts, versions, stage summaries, and `Completed`, `Extinct`, `Cancelled`, or a typed failure. |
| `run_steps` | Execute within a step-count limit and cooperative time budget. |

Only one step may mutate a run at a time; separate runs may execute independently.
A cloneable cancellation handle, obtained before stepping, uses native shared
flags or worker messages without requiring all backend objects to be `Send` or
`Sync`. Cancellation takes effect at documented safe points, not during an
indivisible GPU dispatch.

Commit atomically publishes the population, domain snapshots, persistent operator
state, and logical step/random counters. The cancellation policy either completes
the bounded step or discards staged changes. Snapshot-capable domains restore the
last commit after failure. An adapter that cannot roll back external side effects
must finish its safe unit or return `NonRecoverableDomainFailure` and invalidate
the run. Only committed states are exposed. A snapshot without domain restoration
support is diagnostic, not replayable.

(sec-algorithmic-gas-data)=
## 2. Data ownership and domain consistency

Numerical observations support comparison and computation; opaque snapshots
support domain restoration and transitions. An observation need not contain
enough information to reconstruct the domain state.

Cloning must preserve observation–snapshot consistency. A numerical transform
requires a corresponding domain edit when snapshots are authoritative. Tensor
storage alone does not make images or embeddings physical coordinates.

### 2.1. Shared input, observations, and first-class rewards

`InputBatch` supplies shared external data and declared, immutable algorithm-state
views. `ObservationExtractor` produces numerical observations; `RewardExtractor`
independently produces one raw scalar per walker. Either can consume the same
fields or declared shared evaluations.

Reward direction is explicit: maximize or minimize. Extraction dependencies must
be versioned and acyclic. Opaque snapshots remain accessible only to domain
interfaces.

:::{prf:definition} Input and extraction contract
:label: def-algorithmic-gas-input-extraction

`InputBatch<B>` is an immutable collection of named external/domain fields, with
slot IDs, batch size, field shapes/dtypes/units, stage, version, and origin. It may
contain declared views of an identified algorithm-state snapshot, including
coordinates, velocities, and previous-step quantities. Per-walker fields have
leading axis `[N,...]`; shared/global fields are explicitly tagged and are never
mistaken for walker axes.

An `ObservationExtractor` derives one vector/tensor observation per walker,
possibly structured into named tensor views. A `RewardExtractor` independently
derives exactly one raw scalar per walker. Its output is `RewardBatch<B>` with
values `[N]` in `P`, validity `[N]`, units, and dependency provenance. Neither
extractor is required to consume the other's output. A declared dependency is
permitted if it fits the acyclic evaluation schedule.
:::

| Object | Public numerical contract | Ownership and restrictions |
|---|---|---|
| `InputBatch` | Named raw fields, function results, RAM/images, and optional algorithm-state views | Owned buffers or immutable shared handles retained through evaluation; no hidden live mutable view |
| `ObservationBatch` | `[N,...]` numerical tensor or schema of tensor fields; selected pair comparison returns a scalar | Preserve tensor structure; no implicit image flattening or physical-coordinate interpretation |
| `RewardBatch` | Raw `[N]` scalar objective plus validity and provenance | Equal status to observations in population access, gather, history, and checkpoints; not replaced by fitness |
| `ExtractionContext` | Declared source population version, stage, eligible mask, and derived-field views | Read-only numerical state access; opaque snapshots remain domain-only |

An extractor may take input directly as its output through `IdentityObservation`
or `IdentityReward`. One input tensor can feed both extractors without copying
immutable storage unnecessarily. Arbitrary reward intermediate vectors must be
explicitly reduced by the reward provider to `[N]`; a global scalar requires an
explicit per-walker allocation or broadcast rule. Variable-sized observations
require declared padding/masks or an encoding into a fixed prepared schema.

:::{prf:definition} Population schema and observation views
:label: def-algorithmic-gas-population

`Population<B>` owns an `ObservationBatch<B>`, a first-class `RewardBatch<B>`,
per-slot metadata, and a committed version. Each observation field has a stable
name, shape `[N, ...]`, storage dtype,
semantic role, units, and provenance. Numerical views select named fields and
their declared transformations; a view never changes the underlying meaning of
an observation. `StateStore` is a separate, optional owner and is not an
observation field. Numerical dynamics fields, such as velocity, can be available
in the population/input context without belonging to the distance's selected
observation view. Reward is scalar regardless of that view's rank or dimension.
:::

| Observation class | Fields and shapes | Permitted interpretation |
|---|---|---|
| Vector | `features: [N, d]` in `P` | Numerical features; positions only if the schema declares coordinates and units |
| Position | `position: [N, d_x]` in `P` | Coordinates in a declared domain |
| Phase space | `position: [N, d_x]`, `velocity: [N, d_v]` in `P` | Phase distance allows declared dimensions; BAOAB/restitution require compatible $d_x=d_v$ |
| Image | `pixels: [N, C, H, W]` or explicitly tagged channel-last | Raw storage may be `u8`; an explicit adapter produces normalized `P` features or a declared pixel comparison view |
| Named tensor fields | A schema-checked map of `[N, ...]` fields | Multimodal observations, embeddings, auxiliary features; operators request only the fields they need |

Metadata includes separate `invalid`, `out_of_bounds`, `terminated`, and `truncated`
masks `[N]`; slot and lineage identifiers; domain snapshot versions; and optional
action/time fields with declared semantics. Raw objective values live in
`RewardBatch`, separate from metadata. Slot identity is stable;
lineage identity records replacement. A cumulative reward belongs to the donor's
trajectory on cloning; historical recipient records are not rewritten.
Fitness, companion distances, and gradients are version-tagged caches.

The numerical access and gather contracts are:

```rust
pub trait ObservationBatch<B: ComputeBackend> {
    fn schema(&self) -> &ObservationSchema;
    fn view(&self, request: &ViewRequest) -> Result<ObservationView<'_, B>>;
    fn gather_into(&self, source: &IndexBatch<B>, dst: &mut Self) -> Result<()>;
}

pub trait Population<B: ComputeBackend> {
    fn snapshot(&self) -> PopulationView<'_, B>;
    fn observations(&self) -> ObservationView<'_, B>;
    fn rewards(&self) -> RewardView<'_, B>;
    fn gather_into(&self, source: &IndexBatch<B>, dst: &mut Self) -> Result<()>;
}

pub trait RewardBatch<B: ComputeBackend> {
    fn view(&self) -> RewardView<'_, B>;
    fn gather_into(&self, source: &IndexBatch<B>, dst: &mut Self) -> Result<()>;
}
```

Gather takes `M` checked indices and produces `[M,...]` fields in index order,
preserving dtype. Population gather applies the metadata-copy policy while
retaining recipient slot identity. Source aliasing, missing fields, incompatible
dimensions, invalid indices, and unsupported conversions are errors. Reward
gather preserves raw values and provenance; reuse requires the refresh checks
below. Single-donor copying uses gather; recombination uses an explicit operator.

### 2.2. Extraction APIs and dependency-aware evaluation

`ObservationAdapter` and `RewardSource` are adapter roles implementing
`ObservationExtractor` and `RewardExtractor`, respectively. Each output has one
authoritative extraction pipeline.

```rust
pub trait InputProvider<B: ComputeBackend> {
    fn descriptor(&self) -> InputDescriptor;
    async fn acquire(&mut self, request: InputRequest) -> Result<InputBatch<B>>;
}

pub trait DerivedFieldProvider<B: ComputeBackend> {
    fn descriptor(&self) -> ExtractionDescriptor;
    fn evaluate(&self, input: &InputBatch<B>, context: &ExtractionContext<'_, B>,
        cx: &mut ExecutionContext<B>) -> Result<DerivedFieldBatch<B>>;
}

pub trait ObservationExtractor<B: ComputeBackend> {
    fn descriptor(&self) -> ExtractionDescriptor;
    fn extract(&self, input: &InputBatch<B>, context: &ExtractionContext<'_, B>,
        cx: &mut ExecutionContext<B>) -> Result<ObservationBatch<B>>;
}

pub trait RewardExtractor<B: ComputeBackend> {
    fn descriptor(&self) -> ExtractionDescriptor;
    fn extract(&self, input: &InputBatch<B>, context: &ExtractionContext<'_, B>,
        cx: &mut ExecutionContext<B>) -> Result<RewardBatch<B>>;
}
```

`acquire` returns an owned/versioned batch for specified slots and step/stage.
Admission checks slot identity/order, completeness, source version, and any domain
consistency token. Misaligned, duplicate, missing, or stale slot data fails before
the extractors run; reordering requires an explicit checked index map. External
data must be retained or archived if replay requires it. A seed alone cannot
reproduce an externally supplied sequence of inputs.

Extraction methods borrow inputs and context, return owned batches or immutable
shared buffer handles, and use `P` for numerical computation. Their descriptors
declare input fields, output schema/units, local versus population/global
dependence, supported placements, differentiability, and persistent state.
Nonfinite outputs carry invalid flags or fail under the configured policy;
wrong output ranks, unsupported conversions, absent fields, or cycles are typed
errors. A host-only extractor remains host-only even when selection runs on a GPU;
its transfers and synchronization must be explicitly planned.

Shared fields are evaluated once per valid cache key. Supported extraction patterns include:

| Input | Observation and reward extraction |
|---|---|
| Function evaluation | Reuse $f(x_i)$ for observations and a reward such as $f(x_i)+c\lVert v_i\rVert^2$. |
| Atari RAM/images | Extract positions and an explicitly encoded level, RGB pixels, or an embedding. Level encoding and its distance weight are configured. |
| Population geometry | Return one Einstein-action contribution per walker. The provider defines the allocation: local contribution, density, marginal contribution, or another rule. A broadcast global scalar gives no reward discrimination under global standardization. Physical and convergence claims require separate validation. |

Cache keys include input/state versions, eligible masks, and declared donor or
statistic dependencies. Independent extractors may run in parallel; dependencies
set execution order. Rewards may use observations, later-stage donor batches, or
versioned previous-step fitness, but cannot depend on the fitness they supply.
A change to one walker can invalidate every population-dependent reward.

Reward-driven changes to eligibility are resolved before donor draws. The default
nonfinite reward policy is `RewardEvaluationFailed`; an optional exclusion policy
records reward invalidity separately from observation invalidity and must define
a bounded re-evaluation schedule if the reward itself depends on the eligible
set. The engine must not silently iterate a mask-dependent reward to a fixed point.

### 2.3. Domain state and transitions

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 22, 'rankSpacing': 22}}}%%
flowchart TD
    Inputs["External / domain inputs<br/>Declared algorithm state"] --> Obs["ObservationExtractor<br/>ObservationBatch"]
    Inputs --> Reward["RewardExtractor<br/>RewardBatch"]
    Obs --> Selection["Distance donors / fitness<br/>Cloning donors"]
    Reward --> Selection
    Selection --> Plan["ClonePlan<br/>One or several donors"]
    Store["DomainAdapter<br/>Opaque StateStore"] --> State["Gather or supported<br/>domain recombination"]
    Plan --> State
    Plan --> Numeric["Numerical copy<br/>or recombination"]
    Numeric --> Next["Refresh input / observations / reward<br/>Consistent successor"]
    State --> Next
```

The state API exposes copying and serialization, not inspection by numerical
operators. A state handle may refer to host memory, a simulator-owned arena,
device-resident snapshots, or immutable shared storage. Mutable simulator states
must be independent after cloning, including repeated donors; copy-on-write is
allowed if that independence is preserved.

```rust
pub trait StateStore {
    fn len(&self) -> usize;
    fn gather_into(&self, source: &StateIndexPlan, dst: &mut Self) -> Result<()>;
    fn checkpoint(&self) -> Result<StateArchive>;
    fn restore(&mut self, archive: StateArchive) -> Result<()>;
}

pub trait DomainAdapter<B: ComputeBackend> {
    fn descriptor(&self) -> ComponentDescriptor;
    async fn transition(&mut self, request: TransitionRequest<'_, B>)
        -> Result<DomainTransition<B>>;
    async fn commit_edit(&mut self, request: DomainEdit<'_, B>)
        -> Result<DomainTransition<B>>;
    async fn refresh(&mut self, slots: &IndexBatch<B>)
        -> Result<DomainTransition<B>>;
}
```

`StateIndexPlan` is the same checked source map as numerical gather, represented
at the store's declared placement. Historical pools expose a composite immutable
state view whose rows resolve frame/slot/version references; they do not reinterpret
archived slot IDs as current slots. Moving GPU indices to a host store is an
explicit synchronization and transfer. `StateArchive` owns versioned snapshot
bytes or portable domain references; unsupported serialization is a typed error.
`transition` takes eligible slots, actions/controls, current observation/input versions,
and the corresponding state handles. It restores and advances the simulator and
returns fresh input fields (including raw domain observations and reward signals),
all termination flags, and updated state versions. The configured extractors
derive the authoritative observation and raw reward batches from these fields;
pre-extracted fast paths must satisfy the same provenance contract.
Only the domain adapter may interpret opaque snapshots. States remain outside
the numerical selection/fitness calculation even when essential for transitions.

### 2.4. Edits, refreshes, and consistency barriers

`DomainEdit` names target slots and coordinate/velocity changes. `commit_edit`
updates the domain and returns fresh inputs, observations, and snapshots, or
returns `UnsupportedDomainEdit`. Images and embeddings generally lack an inverse
coordinate edit. Such domains support literal cloning and transitions; jitter
or periodic repair requires an adapter that can apply the corresponding domain edit.

Pure numerical domains use a state-free adapter: numerical coordinates are
authoritative. Snapshot-based domains treat snapshots as authoritative and
re-encode observations after transitions or edits. A domain with independently
editable coordinates declares an atomic synchronization rule. Each returned
`DomainTransition` contains a consistency token tying numerical observations to
the saved state. `refresh` re-observes those states without advancing domain time.

Operator state is classified separately: per-lineage state is donor-gathered;
per-slot state is retained or reset according to its declared policy; shared state
is advanced once per stage. Colored-noise memory, recurrent-policy memory,
trajectory rewards, retained per-lineage input fields, and cached simulator
controls must all choose a category. Global external inputs are not donor-gathered.
Unspecified operator-state behavior is a preparation error. Random stream slot
addresses are not donor-copied, so cloned walkers do not automatically receive
identical future innovations.

A clone or kinetic change must refresh every affected extractor dependency.
For literal trajectory copying, raw historical reward and its source stage may
be copied. For coordinate changes or multi-donor recombination, a derived reward
must be recomputed from authoritative inputs; averaging donor rewards is insufficient.
If the required external function/domain evaluator
is unavailable, the operation returns `RefreshRequired` without publishing stale
values as current. `PostKinetic` history may explicitly mark an unevaluated reward
invalid/deferred; the next enabled reward-fitness stage requires a fresh value.

(sec-algorithmic-gas-geometry)=
## 3. Distances, kernels, and companion laws

Distance defines comparison values, the kernel converts them into weights, and
the sampling law defines dependencies between donor choices. Equal weights do
not make independent draws, permutations, and mutual matching equivalent.

Fisher–Yates followed by adjacent pairing produces uniform mutual matching, not
independent draws. The broad-Gaussian limit produces uniform independent sampling
only under the independent sampling law. Sequential Gaussian greedy matching
defines another law; it is not interchangeable with an ideal weighted matching
distribution.

### 3.1. Comparison values and units

:::{prf:definition} Comparison convention
:label: def-algorithmic-gas-comparison

An `AlgorithmicDistance` evaluates a declared comparison $q_{ij}\geq0$ from a
selected observation view. Its descriptor specifies `Distance`,
`SquaredDistance`, or `Dissimilarity`, symmetry, units, and whether metric axioms
hold. `ComparisonBatch` carries that convention; an interaction kernel must
explicitly accept it. The name of the interface does not assert the triangle
inequality for every implementation.
:::

| Variant | Default output | View, scaling, and edge cases |
|---|---|---|
| `PositionEuclidean` | $d_{ij}=\lVert x_i-x_j\rVert_2$ (`Distance`) | Position `[N,d_x]`; optional declared coordinate scales; squared-output specialization must be tagged |
| `PhaseSpaceDistance` | $d_{ij}=\sqrt{\lVert x_i-x_j\rVert^2+\lambda_{\rm alg}\lVert v_i-v_j\rVert^2}$ (`Distance`) | Position and velocity fields; $\lambda_{\rm alg}\geq0$ has units $[x]^2/[v]^2$, or is dimensionless after explicit nondimensionalization |
| `CosineDissimilarity` | $q_{ij}=1-(u_i\cdot u_j)/(\lVert u_i\rVert\lVert u_j\rVert)$ (`Dissimilarity`) | Explicit `[N,d]` feature view; dimensionless, symmetric, generally not a metric |

The phase-space convention corresponds to
{prf:ref}`def-algorithmic-distance-metric`. With physical units, a zero velocity
weight yields a pseudometric on phase space; it does not compare velocities.
An implementation may omit that unused field only through a declared position-only
specialization. Unequal feature units require explicit normalization, not a
dtype-dependent default scale.

Cosine defines two zero vectors to have dissimilarity `0`, exactly one zero vector
to have dissimilarity `1`, and otherwise clamps the computed cosine to `[-1,1]`
before subtraction. Zero detection uses exact zero by default and a scaled norm
calculation to avoid avoidable overflow/underflow. A configured near-zero threshold
changes the comparison model and must be recorded. The zero convention is not
smooth; derivative-dependent uses must declare a smooth alternative or reject
zero-norm inputs. Invalid/nonfinite observations are excluded before evaluation.

```rust
pub trait AlgorithmicDistance<B: ComputeBackend> {
    fn descriptor(&self) -> DistanceDescriptor;
    fn pairs(&self, queries: &ObservationView<'_, B>, sources: &ObservationView<'_, B>,
        pairs: &PairBatch<B>,
        cx: &mut ExecutionContext<B>) -> Result<ComparisonBatch<B>>;
    fn tile(&self, queries: &ObservationView<'_, B>, sources: &ObservationView<'_, B>,
        rows: &IndexBatch<B>,
        cols: &IndexBatch<B>, cx: &mut ExecutionContext<B>)
        -> Result<ComparisonTile<B>>;
}
```

`pairs` borrows `[m,2]` query/source row indices and returns `[m]` values; `tile`
returns `[b_r,b_c]` for requested query/source sets. Queries have current leading
axis `[N,...]`; sources have the frozen donor pool's `[M,...]` axis. In the default
current-round pool these can alias one immutable observation batch. Both expose
a borrowed `ComparisonView` with their shape and convention for kernel evaluation.
Both are read-only in the query population and donor pool and
use `P`. Implementations must support bounded requests without constructing an
entire $N\times N$ matrix. Incompatible views, indices, units, or output convention
are errors. Periodic comparisons use the domain object defined in
{ref}`sec-algorithmic-gas-fitness-boundaries`.

### 3.2. Kernels and donor selection

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 22, 'rankSpacing': 22}}}%%
flowchart TD
    View[Selected observation view] --> Dist[Distance or dissimilarity]
    Dist --> Conv[Explicit comparison convention]
    Conv --> Kernel[Kernel: pair log-weights]
    Eligible[Eligible candidates and self policy] --> Law[Joint sampling law]
    Kernel --> Law
    Law --> IID[Independent directed draws]
    Law --> Perm[Permutation assignment]
    Law --> Match[Disjoint mutual matching]
    IID --> Result["DonorBatch: N by K<br/>Law and topology"]
    Perm --> Result
    Match --> Result
```

`UniformKernel` assigns weight one to eligible edges. The canonical Gaussian
kernel consumes distance $d$ and uses
$w_{ij}=\exp[-d_{ij}^2/(2\epsilon^2)]$, with $\epsilon>0$ in distance units.
For `SquaredDistance`, the exponent is $-q_{ij}/(2\epsilon^2)$, without squaring
again. Cosine requires a named convention: `ExponentialDissimilarity` uses
$\exp(-q_{ij}/\tau)$, $\tau>0$ dimensionless; `GaussianOfDissimilarity` uses
$\exp[-q_{ij}^2/(2\epsilon^2)]$. They are different configurations.

```rust
pub trait InteractionKernel<B: ComputeBackend> {
    fn descriptor(&self) -> KernelDescriptor;
    fn log_weights(&self, values: &ComparisonView<'_, B>, eligible: &Mask<B>,
        cx: &mut ExecutionContext<B>) -> Result<LogWeightBatch<B>>;
}

pub trait CompanionSampler<B: ComputeBackend> {
    fn descriptor(&self) -> SamplerDescriptor;
    fn sample(&self, request: CompanionRequest<'_, B>,
        rng: &mut RandomStream, cx: &mut ExecutionContext<B>)
        -> Result<DonorBatch<B>>;
}

pub trait DonorModule<B: ComputeBackend> {
    fn descriptor(&self) -> DonorModuleDescriptor;
    fn draw(&self, request: DonorRequest<'_, B>, rng: &mut RandomStream,
        cx: &mut ExecutionContext<B>) -> Result<DonorBatch<B>>;
}

pub trait DonorPoolProvider<B: ComputeBackend> {
    fn descriptor(&self) -> DonorPoolDescriptor;
    fn freeze(&self, current: &PopulationView<'_, B>, archive: &DonorArchiveView<'_, B>,
        cx: &mut ExecutionContext<B>) -> Result<DonorPool<B>>;
}
```

Kernel output has the comparison shape in `P`; ineligible edges have explicit
masked/negative-infinite log-weight. Sampling uses stable normalization or an
equivalent validated categorical method. All eligible Gaussian weights are
mathematically positive; numerical underflow must not silently change the law to
uniform. Nonfinite distances, invalid bandwidths, or a mathematically zero-support row
cause a typed error unless a named support-fallback policy is configured.

`CompanionRequest` borrows a current population version, a frozen donor pool,
eligible query/source sets, a distance/kernel provider, donor capacity, and
self/odd/replacement policies.
`DonorModule` combines geometry, kernel, and sampler in one role-neutral API.
`DonorRequest` adds the role and stage-specific context. The required instances are:

| Instance | Consumer | Independent configuration |
|---|---|---|
| `distance_donors` | Pair-distance evaluation and diversity reduction | $K_{\rm dist}=1$ by default; source pool, sampling geometry/view, kernel, law, replacement/self policies, and its own random namespace |
| `cloning_donors` | Clone proposal construction, acceptance, and copy/recombination | $K_{\rm clone}=1$ by default; its own source pool, geometry/view, kernel, law, policies, and random namespace |

The distance role compares donors without copying them. The two instances may
share immutable configuration and weight tiles when geometry/input dependencies
match. Mutable sampler state and random streams remain separate; sharing draws
would change the algorithm and requires an explicit variant.

:::{prf:definition} Donor pools and ancestry-neutral eligibility
:label: def-algorithmic-gas-donor-pool

`DonorPool<B>` is a read-only, versioned collection of $M$ source records frozen
for a selection/clone transaction. A record is identified by source frame/step,
slot identity and generation, and snapshot version. Its numerical observation,
raw reward provenance, input fields, and optional opaque state refer to that
same record. A compact integer pool-row index resolves that identity; equal slot
numbers in different iterations do not identify the same snapshot.

The default `CurrentEligible` pool uses the current round and admits every
eligible walker regardless of ancestry or lineage label. Kernel weighting and
explicit self/topology rules may restrict edges, but genealogical relationships
are not an implicit filter. Optional `HistoricalWindow` and
`CurrentPlusHistoricalWindow` policies select archived earlier frames. An
ancestry filter is permitted only as an explicitly named, non-default experiment.
:::

`freeze` returns retained immutable source handles, checked identity maps, and
an eligibility mask; composite views can avoid copying frames. Each role configures
retention, frame range/stride, capacity, and missing-frame behavior. Missing
required snapshots, unsupported schema/precision conversions, or insufficient
memory are errors. Default historical donors must be valid and alive at capture,
neither terminated nor truncated, and compatible with current domain/extractor
policies. Policy changes may require revalidation.

Historical uniform sampling is uniform over eligible source records by default
once that optional pool is selected. Equal weighting of frames, age decay, and
deduplication across frames are separate policies. The same lineage or slot may
appear in several frames without any ancestry restriction. Default self-exclusion
removes only the exact current query snapshot; excluding its own earlier versions
would be another explicit policy. Mutual matching, pair restitution, and legacy
permutation variants require a compatible current-round pool by default; archived
states are read-only donors, not partners whose old velocities can be updated.

Historical distance donors require compatible observations. Cloning also requires
the input, operator memory, and domain snapshots needed for a consistent successor;
images alone are insufficient. Donor scores require re-evaluation in the current
frozen context or an explicit historical-score comparison policy. Unavailable
re-evaluation or incompatible score units/stages is an error.

Default revival and extinction use current alive walkers. Historical donor pools
do not enable revival after extinction; that requires a separate archive-restart policy.

:::{prf:definition} Batched donor contract
:label: def-algorithmic-gas-donor-batch

A `DonorBatch<B>` contains checked integer pool-row donor indices `[N,K]`, valid-edge
mask `[N,K]`, per-row valid counts `[N]`, role, frozen-pool/query versions, and joint-law
descriptor. Optional topology data includes matching-round IDs, pair lists, and
unmatched flags. Optional sampling probabilities and consumer weights are
separately named: sampling probability is not automatically an averaging or
recombination weight. Floating fields use `P`; index and mask fields retain their
own types.

Every valid entry references an eligible source under the configured self policy.
Padded entries are never evaluated or treated as zero-distance donors. Rows for
dead queries are invalid for normal selection; revival uses a separate request.
For each certified mutual matching round $a$, $c_a(c_a(i))=i$ on alive slots
after resolving pool-row indices to current slot identities.
No such certificate is inferred for the union of several rounds or a general
directed multi-donor graph.
:::

The public shape remains `[N,1]` in the default case. A checked `single()` view
may expose `[N]` only when exactly one valid donor is required per active row.
Kernels specialize this path without changing the mathematical contract. Outputs
own index buffers and immutable metadata; input lifetimes, dtype checks, support
errors, and memory-limit failures follow the common interface rules.

### 3.3. Supported joint laws and costs

For the default current-round pools, unless overridden by a compatibility profile,
alive queries choose among alive candidates, exclude themselves when $k>1$, and
use self when $k=1$. Historical pools use the explicit record-level eligibility
and self rules above, never an inferred ancestry restriction. Mutual matching
allows exactly one self/unmatched slot when $k$ is odd. These rules apply separately
to `distance_donors` and `cloning_donors`, with separate random namespaces. The
table specifies the base $K=1$ current-pool laws; the multi-donor extensions below state how
those laws are repeated or generalized.

| Sampler | Joint law and implementation contract | Arithmetic / dependency / additional memory |
|---|---|---|
| `UniformIndependent` | Each alive query independently chooses a uniform allowed candidate, with replacement across queries | $O(k)$ draws; rows parallel; $O(k)$ indices |
| `UniformMutualMatching` | Uniform Fisher–Yates permutation of alive IDs; adjacent IDs become mutual pairs; last ID is self if odd | CPU/WASM reference $O(k)$ work and sequential shuffle dependence; $O(k)$ indices |
| `IndependentGaussian` | Independently sample each row with $p_{ij}=w_{ij}/\sum_{\ell\in C_i}w_{i\ell}$ | Dense exact $O(k^2d)$ comparisons; rows/tiles parallel with row reductions; bounded tile scratch plus $O(k)$ outputs |
| `SequentialGaussianGreedyMatching` | Choose the first remaining slot in a declared order, draw a partner from remaining slots proportional to Gaussian weights, remove both, repeat | $O(k^2d)$ comparisons; $O(k)$ dependent pairing rounds; bounded tiles and $O(k)$ bookkeeping |
| `LegacyPermutationCompanions` | Compatibility law: all-alive random permutation permits self and is not generally mutual; with deaths the legacy helper samples alive donors with replacement | $O(N)$ indices/draws on the CPU reference; law and masking differ from the defaults |

The Fisher–Yates algorithm draws each swap index uniformly from its inclusive
remaining range, with unbiased integer sampling. Adjacent pairing of a uniform
permutation produces uniform perfect matching when $k$ is even; for odd $k$ it
also makes the unmatched walker uniform. This is the paired construction used in
{prf:ref}`cor-sm-physics-paired-cloning`. GPU implementations are specified
separately in {ref}`sec-algorithmic-gas-execution`.

For greedy Gaussian matching, default pivot order is ascending alive slot ID;
an independent random permutation of that order is an explicit alternative.
The final unmatched walker self-pairs. This is the sequential law of
{prf:ref}`def-greedy-pairing-algorithm`, not exact sampling from an ideal global
matching distribution proportional to the product of edge weights.
Changing pivot order can change the law.

At fixed finite distances and identical eligibility/self rules, the
$\epsilon\to\infty$ limit of **independent** Gaussian rows is uniform independent
sampling. This does not produce uniform mutual matching: its joint dependence
is different. A kernel alone cannot select between those two laws. A permutation
without replacement is a third law and can contain cycles longer than two.

Legacy compatibility targets the helper in `src/fragile/fractalai/fractalai.py`
and additionally requires matching dead-slot behavior, fitness statistics, operation
order, and random implementation. Its all-dead fallback is not a default here.

For $K>1$, a sampler declares one of the following multiplicity laws:

| Multiplicity law | Semantics and scarce-candidate behavior |
|---|---|
| `IndependentWithReplacement` | $K$ independent draws from each row's candidate law; duplicates are valid repeated samples and are not silently deduplicated |
| `SequentialWithoutReplacement` | Draw proportionally to weights on the remaining candidates, then remove the chosen candidate; this ordered weighted law is explicit, not an unspecified weighted subset law |
| `IndependentMatchingRounds` | Generate $K$ separate mutual matchings, each with a round-specific stream and topology certificate; donors may repeat across rounds |
| `ExplicitAllEligible` | Deterministically enumerate allowed candidates for a full-neighborhood reduction; stream/chunk edges when necessary rather than requiring a dense allocation |

Without replacement, default `UseAvailable` returns at most
$\min(K,|C_i|)$ valid entries and masks padding; `RequireCount` fails instead.
With replacement, a singleton produces repeated self entries if $K>1$ under the
default singleton policy, all with zero raw separation. Each matching round uses
its own odd-population self rule. An all-eligible singleton uses one self entry.
No algorithm treats missing candidates as actual walkers. The legacy compatibility
profile remains $K=1$ unless a separately named repeated-law extension is selected.

Matching rounds can be generated independently, but a union of disjoint matchings
is usually an overlapping graph. An operator requiring disjoint pairs must select
one globally identified round, validate a disjoint submatching, or declare a new
multi-round update schedule; per-walker arbitrary round selection is not sufficient.

Revival samples a uniform alive donor independently for each dead slot by default,
using `revival_donors`. Repeated donors are legal. There is no alive donor at
$k=0$, so the run halts explicitly as `Extinct`; it does not silently restart.
At $k=1$, the surviving walker's raw self-separation is zero. Its diversity
measurement is then the configured regularized separation, defined below.
The alternative theoretical convention that absorbs at $k<2$ is a distinct
`RequireTwoAlive` policy, not this default.

### 3.4. Multiple comparisons and diversity reduction

`AlgorithmicDistance` returns one scalar per valid donor edge. `CompanionReducer`
converts those edge values into one diversity comparison per walker, excluding
padded entries.

Mean distance is not generally distance to the mean observation. Reduction also
does not change topology: a multi-donor graph remains incompatible with pairwise
restitution unless a disjoint matching or supported update schedule is explicitly
selected.

Valid `[N,K_dist]` donor edges form a compact `[m,2]` pair request.
`EdgeComparisonBatch` returns logical `[N,K_dist]` values, mask, convention, and
version, with equivalent streamed chunks for all-eligible or large-$K$ cases.
Measurement and sampling may use different configured distances; this choice is recorded.

```rust
pub trait CompanionReducer<B: ComputeBackend> {
    fn descriptor(&self) -> ReductionDescriptor;
    fn reduce(&self, edges: &EdgeComparisonBatch<B>, weights: Option<&EdgeWeights<B>>,
        cx: &mut ExecutionContext<B>) -> Result<ReducedComparisonBatch<B>>;
}
```

`reduce` borrows scalar edge comparisons and optional declared consumer weights,
returns one `[N]` value and validity mask in `P`, and records the convention of
that reduced value. The initial reducer is `MeanDistance`, with a zero-overhead
single-entry specialization. Additional reducers include weighted mean, minimum,
maximum, and a named smooth aggregation. Mean distance requires conversion of
squared-distance outputs back to distance first; mean squared distance is a
different explicitly named reducer. Cosine values retain their dissimilarity
meaning unless an explicit transform changes it.

For valid entries $A_i$ and nonnegative consumer weights $w_{ia}$,

$$
\bar d_i=\frac{1}{|A_i|}\sum_{a\in A_i}d_{ia},\qquad
\bar d_i^{(w)}=\frac{\sum_{a\in A_i}w_{ia}d_{ia}}
                         {\sum_{a\in A_i}w_{ia}}.
$$

Repeated with-replacement entries count with their multiplicity. Weights must be
finite with positive valid-row sum; padding and invalid values are excluded
before arithmetic. An empty active row is `NoValidDonors`, not zero diversity;
dead rows remain masked. The initial regularization order is **reduce first,
then apply the diversity measurement regularizer** in
{ref}`sec-algorithmic-gas-fitness-boundaries`. Averaging individually regularized
distances is a separate selectable order, since generally

$$
\frac1K\sum_a\sqrt{d_{ia}^2+\varepsilon_{\rm dist}^2}
\ne\sqrt{\left(\frac1K\sum_a d_{ia}\right)^2+\varepsilon_{\rm dist}^2}.
$$

For independent draws from a fixed eligible population with finite conditional
distance variance, the variance of their mean is $\operatorname{Var}(d\mid
\text{snapshot})/K$. Without-replacement draws or coupled laws have their own
covariance terms. Reduced sampling noise changes the selection process and does
not establish better exploration or preserve single-companion convergence claims.
For all-eligible comparisons, the arithmetic can become quadratic even if edge
storage and reduction are streamed.

Reducers declare smoothness and whether their weights depend on current
observations. Differentiating a weighted mean includes those dependencies unless
the target explicitly freezes them. Minimum/maximum have nonsmooth ties; they
cannot silently satisfy Hessian-based requirements through a nominal scalar
output. A differentiable alternative or an explicitly permitted derivative
convention must be selected.

(sec-algorithmic-gas-fitness-boundaries)=
## 4. Boundary policies and fitness construction

Boundary policies preserve separate causes of ineligibility and define supported
repairs. Periodic repair and periodic distance must use the same domain geometry.

Fitness construction separates measurement, standardization, positive mapping,
and channel combination. Reproducing a legacy pipeline requires matching its
statistics, eligibility rules, and degeneracy handling as well as its scalar map.

### 4.1. Validity signals and domain policies

```rust
pub trait BoundaryPolicy<B: ComputeBackend> {
    fn descriptor(&self) -> BoundaryDescriptor;
    fn evaluate(&self, input: BoundaryInput<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<BoundaryReport<B>>;
    fn repair_plan(&self, input: BoundaryInput<'_, B>, report: &BoundaryReport<B>)
        -> Result<Option<DomainEdit<B>>>;
}
```

`BoundaryInput` contains a versioned observation view, domain description, and
external termination/truncation flags. `BoundaryReport` returns distinct `[N]`
masks for invalid observations, raw out-of-bounds status, termination, truncation,
successful repair, and final eligibility, together with reason codes. Evaluation
is read-only; repair returns an edit plan rather than mutating observations.
The engine commits supported edits through the domain adapter and re-evaluates.

| Policy | Numerical/domain semantics | Failure or composition rule |
|---|---|---|
| `Unbounded` | No geometric boundary; required numerical fields must still be finite and schema-valid | Exclude nonfinite observations |
| `AbsorbingBox` | Coordinate-wise closed intervals $a_q\leq x_q\leq b_q$; outside means ineligible | Finite $a_q<b_q$ required; no automatic clipping |
| `PeriodicBox` | Repair to $[a_q,b_q)$ by $a_q+((x_q-a_q)\bmod \ell_q)$, $\ell_q=b_q-a_q$ | Requires consistent domain edit; finite positive lengths; not an absorbing death after successful repair |
| `ExternalTermination` | Preserve `terminated` and `truncated` separately | Default excludes either flag; alternate truncation treatment must be configured and recorded |
| `ComposedBoundary` | Union of mortality/invalidity causes after a declared repair sequence | Reject conflicting coordinate domains or ambiguous repair order |

For periodic distance, the same immutable `PeriodicDomain` supplies
$\delta_q=((x_{iq}-x_{jq}+\ell_q/2)\bmod\ell_q)-\ell_q/2$.
At a half-period tie this formula fixes the sign convention; its squared norm is
unambiguous. Wrapping and minimum-image comparison cannot carry independent box
lengths. Periodic comparisons are nonsmooth on the cut locus; smoothness-dependent
providers must account for that limitation.

Default eligibility is
$\neg\mathrm{invalid}\land\neg\mathrm{unrepaired\_oob}
\land\neg\mathrm{terminated}\land\neg\mathrm{truncated}$.
Raw causes remain available for diagnostics even after a repair. NaNs cannot be
repaired by modulo arithmetic. Domain edits that invalidate other derived fields
must refresh them before eligibility or fitness is recomputed. Boundary detection
at discrete checkpoints does not claim exact continuous-time first-exit detection.

### 4.2. Measurement and standardization

```rust
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

pub trait ObjectiveOrientation<B: ComputeBackend> {
    fn orient(&self, raw: &RewardBatch<B>, direction: ObjectiveDirection,
        cx: &mut ExecutionContext<B>) -> Result<MeasurementBatch<B>>;
}

pub trait DiversityMeasurement<B: ComputeBackend> {
    fn measure(&self, comparisons: &ReducedComparisonBatch<B>, cx: &mut ExecutionContext<B>)
        -> Result<MeasurementBatch<B>>;
}

pub trait Standardizer<B: ComputeBackend> {
    fn standardize(&self, values: &MeasurementBatch<B>, input: StatisticsInput<'_, B>,
        cx: &mut ExecutionContext<B>) -> Result<StandardizedBatch<B>>;
}

pub trait PositiveMap<B: ComputeBackend> {
    fn map(&self, z: &StandardizedBatch<B>, cx: &mut ExecutionContext<B>)
        -> Result<PositiveBatch<B>>;
}

pub trait ChannelCombiner<B: ComputeBackend> {
    fn combine(&self, channels: &[PositiveBatch<B>], cx: &mut ExecutionContext<B>)
        -> Result<FitnessBatch<B>>;
}

pub trait FitnessPipeline<B: ComputeBackend> {
    fn evaluate(&self, input: FitnessInput<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<FitnessEvaluation<B>>;
}
```

Measurements are `[N]` in `P`, with units, validity, and stage/version.
`FitnessInput` borrows fresh raw rewards and reduced diversity comparisons; it
does not evaluate raw reward. Extractors define the objective and its dependencies,
which may extend beyond the selected observation. Simulator rewards declare
whether they describe a transition or cumulative trajectory and retain their
source stage. Negative raw rewards are valid.

:::{prf:definition} Raw reward and objective orientation
:label: def-algorithmic-gas-reward-orientation

The reward extractor returns one scalar $r_i$ per walker. `ObjectiveDirection`
is a run-level configuration, defaulting to `Maximize`. The standard oriented
measurement is

$$
y_i=\begin{cases}r_i,&\mathrm{Maximize},\\-r_i,&\mathrm{Minimize}.\end{cases}
$$

Standardization, positive mapping, and reward/diversity combination consume $y$;
history and checkpoints retain the original $r$ and the direction. Minimization
does not negate positive fitness or reverse the clone rule a second time.
One may equivalently extract $-U$ and maximize, or extract $U$ and minimize, but
must not apply both sign reversals. Direction changes create a recorded
configuration transition and invalidate oriented fitness/derivative caches.
:::

`orient` borrows `[N]` raw values and validity/provenance, returns `[N]` oriented
measurement without mutating the reward batch, and preserves its units. Missing
or nonfinite required rewards fail under the declared reward-invalidity policy.
Custom combinations of several intermediate objective terms belong inside the
reward extractor and still return one scalar; vector-valued Pareto selection is
not implied by this contract. Derivative targets declare whether they differentiate
raw reward, oriented reward, positive fitness, or a separate potential.

The default diversity measurement for the reduced distance $d_i$ is
$h_i=\sqrt{d_i^2+\varepsilon_{\rm dist}^2}$,
$\varepsilon_{\rm dist}>0$ in distance units. Thus a singleton has raw distance
zero and measured separation $\varepsilon_{\rm dist}$, not a fabricated nonzero
pair distance. A `SquaredDistance` input is not squared again. Dissimilarity
measurements require their own explicit transform, such as the same formula
with a dimensionless $q_i$. Invalid slots have a validity mask and harmless
storage values; they do not participate in statistics.

:::{prf:definition} Global and local regularized standardization
:label: def-algorithmic-gas-standardization

For a measurement $y$ and eligible set $\mathcal A$ of size $k>0$, global
alive-only population moments are

$$
\mu=\frac1k\sum_{j\in\mathcal A}y_j,\qquad
s^2=\frac1k\sum_{j\in\mathcal A}(y_j-\mu)^2,\qquad
z_i=\frac{y_i-\mu}{\sqrt{s^2+\sigma_{\min}^2}},\quad i\in\mathcal A.
$$

The positive model regularizer $\sigma_{\min}$ has measurement units. Dead slots
receive a masked zero score. Constant alive measurements, including a singleton,
produce $z_i=0$.

For local weights $a_{ij}\geq0$, let
$C_i=\mathcal A\setminus\{i\}$ by default and
$\omega_{ij}=a_{ij}/\sum_{\ell\in C_i}a_{i\ell}$.
Then

$$
\mu_i=\sum_{j\in C_i}\omega_{ij}y_j,\qquad
s_i^2=\sum_{j\in C_i}\omega_{ij}(y_j-\mu_i)^2,\qquad
z_i=\frac{y_i-\mu_i}{\sqrt{s_i^2+\sigma_{\min}^2}}.
$$

The neighborhood distance, kernel, bandwidth, and self policy are separate
standardizer configuration. An empty or zero-support neighborhood uses the
global rule by default and records that fallback. A strict-error alternative
must be selected explicitly. Numerical underflow is not a mathematical
zero-support neighborhood and must be handled by stable normalization.
:::

Statistics consume only gathered/masked valid values; multiplying a NaN by zero
is not a valid masking implementation. Global moments use a stable batched
reduction; local moments use tiled weighted reductions with no mandatory dense
matrix. Small negative variance caused by rounding may be clamped within a
declared numerical tolerance; a larger violation is an error. Standardized
outputs are dimensionless and carry the moments, mask, and provenance needed by
diagnostics and declared derivative providers.

### 4.3. Positive maps, combination, and legacy fidelity

:::{prf:definition} Positive fitness channels
:label: def-algorithmic-gas-positive-fitness

The documented logistic map is

$$
g_A(z)=\frac{A}{1+e^{-z}},\qquad A>0.
$$

Its symmetry is $g_A(z)+g_A(-z)=A$; it is not an odd function.
The default positivity policy is the separately configured additive floor
$\widetilde g(z)=g(z)+\eta$, $\eta>0$. It is not an implicit dtype epsilon or a
hard clipping operation.

The legacy asymmetric scalar map, when applied to an already standardized value,
is

$$
g_{\rm legacy}(z)=
\begin{cases}
e^z,&z\leq0,\\
1+\log(1+z),&z>0.
\end{cases}
$$

Given mapped reward and diversity $R_i,D_i>0$, the default combiner is

$$
F_i=R_i^\alpha D_i^\beta,\qquad \alpha,\beta\geq0.
$$

A disabled fitness channel contributes one and is not evaluated; both exponents zero
give constant fitness one. Enabled channels and exponents are recorded
independently. Clone acceptance uses these pre-clone fitness values.
:::

Disabling the reward contribution does not erase `RewardBatch` or its independent
use by diagnostics, dynamics, or donor providers. Extraction may be deferred only
when no configured consumer requires it; deferred entries are marked unavailable,
never filled with fabricated current rewards. The same dependency rule applies
to optional diversity evaluation.

`PositiveMap` consumes standardized values only. The existing
`asymmetric_rescale` in `src/fragile/fractalai/fractalai.py` includes its own
standardization: it uses `Tensor.std()` with the sample-standard-deviation
convention and returns ones for zero or nonfinite standard deviation. A legacy
compatibility pipeline therefore requires a `LegacyStandardizer` with the same
input population, degrees-of-freedom and degeneracy policy, as well as the map.
Choosing the asymmetric map after the new alive-only population standardizer
does **not** reproduce the old function. Exact compatibility also selects no
additive floor where the original function had none, and records the resulting
underflow risk and failure handling.

The logistic map is smooth. The asymmetric map is $C^1$ but not $C^2$ at zero:
its one-sided second derivatives are $1$ and $-1$. Hessian-dependent variants
must reject that composition at the nonsmooth point or explicitly choose a
different smooth map. Hard masks, cosine zero rules, neighbor truncation, and
boundary changes impose additional smoothness restrictions. Smoothness of the
scalar map alone is not smoothness of sampled, population-dependent fitness.

Combination may use log-space arithmetic for numerical stability, but must
preserve the specified positive-fitness and acceptance semantics. Nonfinite
enabled-channel output or unrepresentable results produce `NumericalFailure`,
not an unrecorded clipping rule. Additive floors, measurement regularizers,
kernel widths, and clone denominator regularizers are mathematical parameters;
they remain separate from numerical tolerances.

(sec-algorithmic-gas-dynamics)=
## 5. Cloning, kinetic operators, and independent noise

Cloning copies or recombines immutable source records into a separate destination.
Kinetics advances the resulting eligible population. This separation prevents
execution order from changing source values during chains, swaps, or repeated-donor
updates.

`NoiseSource` supplies innovations $\eta=L\xi$ independently of the consuming
operator. For conditionally centered unit-covariance innovations and a factor
fixed before sampling, the covariance is $LL^\top$. The kinetic integrator or
clone transform supplies temporal scaling or jump amplitude; the noise source
does not apply an implicit $\sqrt{\Delta t}$.

### 5.1. Decisions and simultaneous clone transforms

`distance_donors` and `cloning_donors` are independently configured `DonorModule`
instances. Both default to one donor and support multiple donors.

Selecting one candidate from several remains single-donor cloning. True multi-donor
cloning combines contributions through explicit proposal, acceptance, and transform
rules. Opaque states have no generic averaging operation: domain-backed recombination
requires a compatible `DomainRecombiner` or fails preparation.

:::{prf:definition} Immutable clone plan
:label: def-algorithmic-gas-clone-plan

A `ClonePlan` is defined against one immutable pre-clone query population and a
frozen donor pool. It records recipient slots, proposed and effective source
references `[N,K_clone]`, donor masks/counts, any operator-defined coefficients, acceptance probabilities
and decisions `[N]`, revival decisions, and certified topology. Each recipient
has exactly one resolved update: `Keep`, `Copy(donor)`, or
`Recombine(donors, operator, parameters)`. Several donors can contribute to one
recombined successor; the plan is not universally a one-index source map.

The default has one donor and uses `Copy` or `Keep`. A checked single-source map
is available for copy-only plans. `Keep` reads the current recipient directly;
the source-map adapter resolves both these current reads and pool-source copies,
including historical records. Revival defaults to one pre-clone alive donor
and a separate copy rule. Every source read, including chains, repeated donors,
and recombination, resolves against the immutable current snapshot or frozen
donor pool, never the destination. Current sources are from
the current round; historical sources retain their own archived frame/version.
An old frame is never overwritten, and ancestry never determines eligibility
unless an explicit non-default filter was selected.
All successor fields and domain effects are assembled in separate destination
storage and committed consistently.
:::

The default one-donor `RelativeFitnessDecision` uses pre-clone fitness and a separate
`clone_acceptance` stream:

$$
s_i=\frac{F_{c(i)}-F_i}{F_i+\varepsilon_{\rm clone}},\qquad
p_i=\operatorname{clip}\!\left(\frac{s_i}{p_{\rm sat}},0,1\right),\qquad
b_i=\mathbf1\{u_i<p_i\},\quad u_i\sim\operatorname{Uniform}[0,1).
$$

Here $\varepsilon_{\rm clone}>0$ has fitness units, and $p_{\rm sat}>0$ is a
dimensionless saturation threshold. The existing Euclidean implementation's
`p_max` acts as this divisor/threshold; it is not an upper bound smaller than one
on the acceptance probability. Self-companions have zero acceptance. Dead slots
are revived by unconditional donor copy when an alive donor exists; their invalid
fitness is never used in this formula.

For a current-pool copy, $F_{c(i)}$ is the selected source's pre-clone fitness.
A historical pool-row index cannot index the current `[N]` fitness batch. Such a
copy requires a `ProposalFitnessProvider` to supply its compatible source score
under the declared historical-score policy, with the same validity, units, and
comparison-context checks as other proposal scores.

```rust
pub trait CloneProposal<B: ComputeBackend> {
    fn descriptor(&self) -> CloneProposalDescriptor;
    fn propose(&self, source: &CloneSourceView<'_, B>, donors: &DonorBatch<B>,
        rng: &mut RandomStream, cx: &mut ExecutionContext<B>)
        -> Result<ProposedCloneBatch<B>>;
}

pub trait CloneDecision<B: ComputeBackend> {
    fn decide(&self, source: &PopulationView<'_, B>, fitness: &FitnessBatch<B>,
        proposals: &ProposedCloneBatch<B>, revival: &RevivalDonors<B>,
        proposal_fitness: Option<&ProposalFitnessBatch<B>>,
        rng: &mut RandomStream, cx: &mut ExecutionContext<B>) -> Result<ClonePlan<B>>;
}

pub trait ProposalFitnessProvider<B: ComputeBackend> {
    fn descriptor(&self) -> ProposalFitnessDescriptor;
    fn evaluate(&self, proposals: &ProposedCloneBatch<B>,
        context: &ProposalEvaluationContext<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<ProposalFitnessBatch<B>>;
}

pub trait CloneTransform<B: ComputeBackend> {
    fn descriptor(&self) -> CloneTransformDescriptor;
    fn apply(&self, source: &CloneSourceView<'_, B>, plan: &ClonePlan<B>,
        destination: &mut PopulationBuffer<B>, rng: &mut RandomStream,
        cx: &mut ExecutionContext<B>) -> Result<CloneEffects<B>>;
}

pub trait DomainRecombiner<B: ComputeBackend> {
    fn descriptor(&self) -> DomainRecombinationDescriptor;
    async fn recombine(&mut self, source: &DomainSnapshotView, plan: &ClonePlan<B>,
        cx: &mut ExecutionContext<B>) -> Result<DomainTransition<B>>;
}
```

`CloneSourceView` contains the immutable current recipient snapshot and frozen
donor pool. Source references include frame/version, not only current slot indices.

| Method | Input and output contract |
|---|---|
| `propose` | Read cloning donors and immutable numerical inputs. Return owned donor lists, proposal kind, topology, coefficients, and optional tentative fields with provenance; commit nothing. |
| `decide` | Read `[N]` fitness, proposals, and source mask. Return a versioned plan without inspecting opaque state. |
| `apply` | Write a fresh `[N,...]` destination. Return changed-field/slot masks, lineage events, cache invalidations, and required domain edits. |

The engine applies copy maps through `StateStore::gather_into` and recombinations
through `DomainRecombiner`, committing numerical and domain updates together.
Stale versions, invalid donors, incompatible transforms, and source aliasing are
errors; the last committed numerical state remains intact on failure.

The architecture distinguishes three cloning modes:

| Mode | Donor module and proposal contract | Acceptance / successor semantics |
|---|---|---|
| Default `SingleDonor` | $K_{\rm clone}=1$; `CopyProposal` | Existing relative-fitness decision and literal copy or compatible jitter/restitution |
| `SelectOneOfMany` | $K_{\rm clone}>1$; `SelectOneProposal` chooses one valid candidate | Uniform, weighted, or best-candidate choice is explicit and separately streamed; successor still has one copying donor |
| Future `MultiDonorRecombination` | $K_{\rm clone}>1$; a recombination proposal retains several contributing donors | Registered numerical/domain transform and acceptance semantics; not reduced to one donor by the API |

True recombination may define, for a numerical coordinate field, an offspring
$x_i^*=\sum_a\theta_{ia}x_{j_{ia}}$ with nonnegative coefficients summing to one,
or a different explicitly named operation. The transform specifies all written
fields, coefficient constraints, whether the recipient also contributes, and
what happens when fewer donors are available. It cannot assume that images,
levels, velocities, cumulative rewards, or domain snapshots obey the same mixing
rule as a Euclidean position vector. Duplicate donor references retain their declared
multiplicity or are explicitly combined with summed coefficients.

The default acceptance formula applies directly only to one effective donor.
A multi-donor proposal must declare its proposal score/fitness and decision rule:
for example, a `ProposalFitnessProvider` evaluating tentative fields against a
named frozen population context, or a separately defined donor-fitness rule.
Evaluating an offspring objective is not generally equivalent to averaging donor
rewards or fitness. A registered proposal-fitness provider has the common provider
contract: input donor/proposal fields and frozen context, output `[N]` scores
with validity/units/version, required precision/placement, and typed failure.
Missing multi-donor acceptance semantics is a preparation error; the engine does
not silently reuse the first or best donor. Proposed evaluations have a separate
stage and RNG namespace and cannot mutate current fitness.

An explicitly selected `RelativeProposalFitness` decision may substitute positive
proposal fitness $F_i^*$ for $F_{c(i)}$ in the relative score, provided its units
and comparison context match $F_i$. This is a configured multi-donor policy,
not an assumed property of recombination. Invalid tentative states are marked
ineligible for acceptance before providers read unsupported fields; speculative
domain evaluation needs an isolated, rollback-capable candidate arena.

`DomainRecombiner` alone may interpret its borrowed immutable snapshot view. It
constructs valid destination snapshots and corresponding fresh input fields for
the accepted recombinations. Unsupported snapshot mixing, invalid domain state,
or unavailable input/reward refresh fails explicitly. Choosing one snapshot while
averaging unrelated observations is not a valid consistency rule. A donor-choice
state policy is allowed only if the resulting numerical fields are
consistent with that state, or the domain can commit the required edits.

Multi-donor lineage records every contributing source frame/slot and operator-defined
coefficient/role, separately from the candidate list and acceptance decision.
Per-lineage operator memory and raw input fields require explicit recombination,
donor-selection, or reset rules; single-donor gather is not a universal default
for them. Raw rewards are refreshed under their dependency contracts. The initial
implementation milestone enables only `SingleDonor`, but its public batches,
plans, checkpoints, and extension registry must admit the other modes without a
schema redesign. Unsupported registered combinations fail capability checks.

| Transform | Copied or transformed fields | Preconditions and exclusions |
|---|---|---|
| `LiteralCopy` | All declared donor-owned observation fields, snapshots, trajectory metadata, and operator memory | General observations; recipient slot ID retained and donor provenance recorded |
| `PositionJitter` | Literal copy followed by a configured position innovation on accepted alive recipients | Declared coordinates; injected noise source; compatible domain edit; velocity unchanged unless another transform explicitly changes it |
| `PairwiseRestitution` | On each accepted disjoint alive pair, update both velocities from their pre-clone pair values; copy accepted recipient positions | Compatible phase-space fields; mutual matching certificate; donor-side velocity write is explicit |
| `RevivalCopy` | Literal pre-clone donor copy into each dead recipient | Independent revival map; no pairwise restitution; optional extra revival transform must be separately configured |
| Future `MultiDonorRecombine` | Explicit multi-donor numerical construction and fresh derived observations/reward | Supported field semantics, proposal/acceptance rules, lineage/operator-state policy, and domain recombiner where opaque states are present |

Default single-donor transform composition is donor gather, optional alive-recipient position
jitter, then pairwise velocity restitution computed from the immutable source.
Composition records write sets and order. Two transforms writing the same field
without a declared composition rule are rejected.

For equal-mass velocities of an accepted alive pair $(i,j)$,

$$
\bar v_{ij}=\frac{v_i+v_j}{2},\qquad
v_i'=\bar v_{ij}+a(v_i-\bar v_{ij}),\qquad
v_j'=\bar v_{ij}+a(v_j-\bar v_{ij}),\qquad 0\leq a\leq1.
$$

This preserves $v_i+v_j$ for that pair. With the specified relative-fitness
decision, at most one direction in a mutual pair can have positive acceptance;
equal fitness means neither. A custom decision that accepts both directions
must provide its own unambiguous pair contract. Disjointness prevents competing
velocity writes; directed companions, arbitrary permutations, true multi-donor
recombination, and overlapping unions of matching rounds are incompatible with
this transform and fail preparation unless a separately validated disjoint update
schedule is selected. Selecting one globally identified matching round is a
single-pair variant; sequential rounds are a different kinetic/clone schedule
with intermediate-state semantics, not simultaneous updates from one snapshot.
This pair conservation statement does
not cover revival, literal copying outside the pair rule, boundary impulses, or
later Langevin dynamics.

Chains, swaps, and repeated donors are legal for literal copying. For the source
map `[1,2,2]`, destination slot `0` receives original slot `1`, not its newly
copied value from slot `2`. A dead slot copying a donor whose velocity is changed
by restitution receives that donor's **pre-clone** snapshot by default. Post-pair
revival would be a different, explicitly named two-stage transform.

### 5.2. Noise laws and spatial factors

:::{prf:definition} Factored noise source
:label: def-algorithmic-gas-noise

An `InnovationLaw` generates centered innovations
$\xi_i\in\mathbb R^r$. A `NoiseGeometry` supplies factors
$L_i\in\mathbb R^{d\times r}$. A `NoiseSource` composes them as

$$
\eta_i=L_i\xi_i.
$$

If, conditionally on the source's current inputs, $\mathbb E[\xi_i]=0$ and
$\operatorname{Cov}(\xi_i)=I_r$, and $L_i$ is determined before the current draw,
then

$$
\mathbb E[\eta_i\mid\text{current inputs}]=0,\qquad
\operatorname{Cov}(\eta_i\mid\text{current inputs})=L_iL_i^\top.
$$

Default innovations are independent across walkers and calls. Noise sources do
not multiply by $\sqrt{\Delta t}$: temporal scaling belongs to the consuming
kinetic integrator or clone transform.
:::

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 22, 'rankSpacing': 22}}}%%
flowchart TD
    Law[InnovationLaw] --> Xi[Centered innovations: N by r]
    Inputs[Declared current observation fields] --> Geometry[NoiseGeometry]
    Geometry --> Factor[Factor L: d by r or N by d by r]
    Xi --> Source[NoiseSource: eta equals L xi]
    Factor --> Source
    Source --> Jump["Direct jump<br/>amplitude"]
    Source --> Brownian["Brownian<br/>square-root time"]
    Source --> Thermostat["Langevin<br/>finite-step thermostat"]
    Jump --> Update[Kinetic update]
    Brownian --> Update
    Thermostat --> Update
```

The noise API borrows a versioned input context and writes an owned innovation
batch at the requested placement and precision. `NoiseRequest` declares slots,
output dimension, field target/units, stage/substep, and stream namespace.

```rust
pub trait InnovationLaw<B: ComputeBackend> {
    fn descriptor(&self) -> InnovationDescriptor;
    fn sample(&self, shape: InnovationShape, rng: &mut RandomStream,
        cx: &mut ExecutionContext<B>) -> Result<InnovationBatch<B>>;
}

pub trait NoiseGeometry<B: ComputeBackend> {
    fn descriptor(&self) -> NoiseGeometryDescriptor;
    fn factor(&self, input: NoiseInput<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<NoiseFactor<B>>;
}

pub trait NoiseSource<B: ComputeBackend> {
    fn descriptor(&self) -> NoiseDescriptor;
    fn sample(&self, request: NoiseRequest<'_, B>, rng: &mut RandomStream,
        cx: &mut ExecutionContext<B>) -> Result<NoiseBatch<B>>;
}
```

`InnovationBatch` has shape `[n,r]` for `n` requested slots; `NoiseBatch` has
`[n,d]` plus moment/coupling descriptors. Stateful adapters require explicit mutable
state access. Geometry uses declared current inputs, independently of the draw
being scaled. Broadcast shapes are checked during preparation and on dynamic changes.

| Geometry | Factor representation | Conditional covariance and validation |
|---|---|---|
| Isotropic | Global scalar $\sigma$ or per-walker `[n,1]`; implicit $L_i=\sigma_i I_d$ | $\sigma_i^2I_d$; $r=d$, finite $\sigma_i\geq0$ |
| Diagonal | Global `[d]` or per-walker `[n,d]` diagonal | $\operatorname{diag}(\ell_{iq}^2)$; $r=d$, finite factors; signed entries permitted as factors |
| Full factor | Global `[d,d]` or per-walker `[n,d,d]` | $L_iL_i^\top$; need not be symmetric or triangular; finite entries |
| Low-rank factor | Global `[d,r]` or per-walker `[n,d,r]`, $1\leq r<d$ | Rank at most $r$; semidefinite covariance is allowed and is not an ellipticity guarantee |
| Observation-dependent | A factor provider returning any representation above | Required view, stage, units, freshness, and placement declared; evaluate before sampling |

A matrix supplied as $L$ is a **factor**, not a covariance. A separate
`CovarianceFactorProvider` may accept $C$, check symmetry/positive semidefiniteness,
and produce $L$ with $LL^\top=C$ under declared numerical tolerance. Its method
(for example positive-definite Cholesky or rank-revealing factorization), required
operations, and rank handling are capabilities. Adding a diagonal regularizer
changes the model and is never done silently. A negative variance, unsupported
factorization, incompatible rank, or unrepresentable factor is an error.

| Innovation law | Standardization and moments | Use restrictions |
|---|---|---|
| Gaussian, default | Independent $\mathcal N(0,1)$ components | Unit covariance; suitable for the Gaussian thermostat below |
| Standardized uniform | Independent $\operatorname{Uniform}[-\sqrt3,\sqrt3]$ components | Zero mean and unit variance, but not a Gaussian increment |
| Finite-variance Student | Independent $t_\nu\sqrt{(\nu-2)/\nu}$ components, $\nu>2$ | Unit covariance; for $2<\nu\leq4$ fourth moments need not exist, affecting covariance-test uncertainty |

Distributions with no finite covariance require a different moment descriptor;
they cannot claim the covariance contract or be accepted by an integrator that
requires it. Isotropic covariance is weaker than rotational invariance: independent
uniform or Student components with scalar geometry have covariance proportional
to $I$ but need not be spherically distributed. A rotationally invariant
non-Gaussian law must specify its joint distribution explicitly.

Cross-walker or temporally correlated noise is an extension with an explicit
coupling descriptor. Such a source must provide the relevant cross-covariances
or state-transition law, declare whether per-lineage noise memory is copied on
cloning, and checkpoint all persistent state. Independent-stream guarantees do
not apply to an intentionally coupled source. Operator streams for position
jitter and kinetic noise are distinct even when they share a noise configuration.

### 5.3. Kinetic operators and derivative providers

```rust
pub trait KineticOperator<B: ComputeBackend> {
    fn descriptor(&self) -> KineticDescriptor;
    async fn advance(&self, input: KineticInput<'_, B>, providers: &ProviderSet<B>,
        noise: &dyn NoiseSource<B>, control: &mut StepControl<B>)
        -> Result<KineticResult<B>>;
}

pub trait GradientProvider<B: ComputeBackend> {
    fn descriptor(&self) -> DerivativeDescriptor;
    fn evaluate(&self, input: DerivativeInput<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<GradientBatch<B>>;
}

pub trait HessianProvider<B: ComputeBackend> {
    fn descriptor(&self) -> DerivativeDescriptor;
    fn apply(&self, input: DerivativeInput<'_, B>, vectors: &FloatBatch<B>,
        cx: &mut ExecutionContext<B>) -> Result<FloatBatch<B>>;
}

pub trait DriftProvider<B: ComputeBackend> {
    fn evaluate(&self, input: DynamicsInput<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<DriftBatch<B>>;
}

pub trait DiffusionProvider<B: ComputeBackend> {
    fn factor(&self, input: DynamicsInput<'_, B>, cx: &mut ExecutionContext<B>)
        -> Result<NoiseFactor<B>>;
}
```

`KineticInput` borrows the post-clone population, eligible slot set, step duration,
and domain context. `StepControl` owns scratch/random streams and the boundary
checkpoint callback; it gates every declared substep before another provider
reads the result. `KineticResult` owns successor fields/domain effects and
substep reports. The interface sketch permits runtime selection; prepared
numerical inner loops specialize the chosen provider/noise types, rather than
using per-walker dynamic dispatch. A noise-free environment transition declares
that it does not consume the injected source.

| Provider | Mathematical output and shape | Required declaration / failure |
|---|---|---|
| `GradientProvider` | $\nabla_q U$ or a named scalar target, `[n,d_q]` | Differentiated field $q$, units $[U]/[q]$, target sign, stage, and population dependence; not automatically force |
| `HessianProvider` | Hessian-vector product in the selected coordinates; `[n,d_q]` for walker-local scalar targets | Walker-local versus full-population coupling, vector layout, symmetry/smoothness assumptions; dense `[n,d_q,d_q]` materialization is optional |
| `DriftProvider` | Drift $b$ of the named SDE/update, `[n,d]` | Units of target field per time and declared Itô/other convention |
| `DiffusionProvider` | SDE factor $B$, `[d,r]` or `[n,d,r]` | Units of target field per square-root time; factor, not covariance; may implement `NoiseGeometry` through a units-checked adapter |

For a scalar depending on the full population, derivatives must declare whether
the gradient is of one target $U_i$ with respect to its own field, all targets
with respect to all fields, or an aggregate objective. A local `[n,d]` interface
must not silently discard off-walker dependence. A full-population Hessian-vector
product acts on the declared flattened population degrees of freedom without
materializing a four-axis Hessian.

Analytic functions, finite differences with recorded perturbation rules, and
optional Burn automatic-differentiation adapters can implement these contracts.
A frozen-companion sampled fitness $\widetilde F(\cdot\mid c)$ is a different
derivative target from expected fitness $\overline F=\mathbb E_c[F]$, whose
derivative can include derivatives of the sampling law. The target, frozen
variables, companion version, eligible set, and differentiation scope must be
declared. No provider is required to differentiate opaque simulator state or
discrete cloning decisions. This distinction agrees with the fitness separation
in {prf:ref}`def-latent-fractal-gas-fitness`.

The initial kinetic variants have the following precise semantics:

| Variant | Update and requirements | Noise/time contract |
|---|---|---|
| `DirectStochasticJump` | Selected coordinates $x'=x+a\eta$, or declared drift-plus-jump $x'=x+h b(x)+a\eta$; no velocity required | `a` is a configured amplitude; no implicit $\sqrt h$ |
| `EulerMaruyama` extension | Itô $x'=x+h b(x)+\sqrt h\,B(x)\xi$ | Factor and drift evaluated at the start of the substep; covariance $hBB^\top$ |
| `UnderdampedBAOAB` | Compatible position/velocity fields, force/gradient provider, scalar friction, independent Gaussian innovations | Finite-step thermostat scaling below; not the Euler noise amplitude |
| `EnvironmentTransition` | Actions/controls and domain adapter restore/advance snapshots, return observation/reward/flags | External simulator owns its time convention and randomness unless explicitly injected; host/device placement declared |

The BAOAB reference targets, with unit mass,

$$
dx=v\,dt,\qquad dv=-\nabla U(x)\,dt-\gamma v\,dt+B(x)\,dW_t,
\qquad\gamma\geq0.
$$

For step $h>0$: **B** is a half force kick;
**A** is a half position drift; **O** is the thermostat; then another **A** and
another **B**. With $B$ evaluated and frozen at the O-substep inputs, set

$$
v\leftarrow a_h v+c_h\eta,\qquad
a_h=e^{-\gamma h},\qquad
c_h=\begin{cases}
\sqrt{(1-e^{-2\gamma h})/(2\gamma)},&\gamma>0,\\
\sqrt h,&\gamma=0,
\end{cases}
\qquad \eta=B\xi.
$$

The Gaussian O update has conditional covariance $c_h^2BB^\top$ for this frozen
subproblem. Stable evaluation near $\gamma h=0$ is a numerical requirement.
For $B=\sqrt{2\gamma T}\,I$ and $\gamma>0$, it reduces to covariance
$T(1-e^{-2\gamma h})I$. General anisotropic $B$, position-dependent geometry,
nonconservative drift, cloning, and boundaries do not automatically preserve a
Gibbs law. Replacing Gaussian innovations by uniform or Student innovations
defines a different discrete stochastic operator; it is not the exact Gaussian
OU thermostat and inherits no Gaussian equilibrium claim. Matrix-valued friction
requires an additional thermostat provider solving the corresponding covariance
problem, not reuse of this scalar formula.

The second B kick evaluates force at the new position. Refresh derivatives after
any change to their input fields, eligible masks, reward/statistic dependencies,
or frozen companions. Gather, jitter, repair, and kinetic substeps can invalidate
caches without changing buffer shapes. Pre-clone fitness remains stage-tagged
diagnostic data, not a substitute for current derivative inputs.

### 5.4. Step stages and boundary checkpoints

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 20, 'rankSpacing': 12, 'padding': 10}}}%%
flowchart TD
    B0["1. Admit input / extract<br/>Validate / repair"]
    B0 -->|No donor| Halt[Explicit extinction halt]
    B0 -->|Alive donors| Fitness["2. Raw reward / direction<br/>Distance donors / reduce / fitness"]
    Fitness --> Plan["3. Cloning donors<br/>Propose / decide / plan"]
    Plan --> Clone["4. Copy or recombine<br/>Refresh dependencies"]
    Clone --> B1["5. Boundary check"]
    B1 -->|Eligible slots| Kinetic["6. Kinetics + boundary"]
    B1 -->|None eligible| Record[Record and commit]
    Kinetic -->|Next substep if eligible| Kinetic
    Kinetic -->|Done or extinct| Record
    Record -->|Next step if alive| B0
```

The default stage contract is normative:

| Stage | Population measured or modified | Required action and recorded meaning |
|---|---|---|
| `ValidatedCurrent` | Admitted external/domain input and current algorithm-state view after permitted repair | Check identity/version; extract observations; classify all causes; refresh repaired dependencies; halt if no eligible donors |
| `PreCloneFitness` | Immutable validated current input/population snapshot | Extract raw reward, orient maximize/minimize, finalize configured reward validity; draw `distance_donors`; compare valid edges, reduce, regularize, standardize, map, and combine |
| `CloneProposal` / `CloneDecision` | Same immutable snapshot and fitness; isolated tentative proposal fields | Independently draw `cloning_donors` and revival donors; construct one- or multi-donor proposals, evaluate declared proposal scores if required, accept and construct `ClonePlan` |
| `PostClone` | Fresh copied/recombined/transformed destination | Synchronize opaque state, input fields, observations, and required reward refreshes; invalidate derived caches; retain pre-clone measurements only with their old stage labels |
| `PostCloneBoundary` | Refreshed post-clone state | Repair/classify; only eligible walkers enter kinetics; derivative providers now see their declared post-clone inputs |
| `KineticSubstep(j)` | Output of the named substep | Check required fields and boundary signals before the next provider call; repair consistently or mark slot ineligible and stop its remaining substeps |
| `PostKinetic` | Final resulting state, with parked ineligible slots | Final check; refresh input/extractor dependencies required by recorded outputs; record raw reward and validity/stage, not stale fitness; commit; fresh final fitness is optional with its own diagnostic stream |

For BAOAB, validation after each B, A, and O stage prevents an invalid velocity
or out-of-domain position from being passed to the next stage's derivative
provider. A successfully periodic-repaired walker may continue. An absorbed,
terminated, truncated, or invalid walker is parked and cannot continue substeps.
Repair-induced boundary effects are recorded with the substep index. These are
discrete checks; detecting an unobserved crossing between endpoints requires an
explicit event-aware kinetic variant.

Revival occurs once, at the default cloning stage. A walker newly killed by
jitter or kinetics waits until the next step's revival stage; it is not revived
and advanced again within the same step. If all walkers become ineligible after
cloning or a substep, remaining kinetics is skipped, the resulting stage is
recorded, and the run reports `Extinct`. A singleton may self-compare and supply
revival donors under the default policy. `RequireTwoAlive` changes this stopping
rule and is recorded as a different theoretical profile.

Post-clone fitness is evaluated only when requested, with a declared scope and
stream. Raw reward and observation refreshes follow the dependency graph; even
literal gather can invalidate population-dependent rewards. Copied historical
rewards retain their trajectory stage until fresh domain values are available.

This is the default dependency order. Reward variants depending on a donor batch
declare a later extraction stage and a compatible schedule; donor selection that
already requires that same reward is rejected as a cycle. Boundary or reward
exclusions that leave no alive donors use the same explicit extinction policy.

(sec-algorithmic-gas-execution)=
## 6. Parallel execution, precision, and deployment

Preparation resolves the requested operators, precision, and placement against
backend capabilities. Unsupported combinations fail explicitly; precision
reductions and device transfers are never implicit.

WASM CPU execution supports `f32` and `f64`; `wasm32` describes addressing, not
floating-point precision. This design's browser WebGPU profile uses `f32`, so
browser `f64` requires explicit WASM CPU selection.

CPU Fisher–Yates has sequential dependencies. GPU and hybrid implementations
therefore need separate cost contracts covering work, memory, synchronization,
and transfers.

### 6.1. Backend facade and numerical policies

```rust
pub trait ComputeBackend {
    type Real: RealScalar;
    type FloatBuffer;
    type IndexBuffer;
    type BoolBuffer;
    type Device;
    fn capabilities(&self, device: &Self::Device) -> CapabilitySet;
    fn allocate(&self, request: AllocationRequest) -> Result<BufferHandle<Self>>;
    fn submit(&self, operation: PreparedOperation<Self>) -> Result<CompletionToken>;
}

pub trait ExecutionContext<B: ComputeBackend> {
    fn precision(&self) -> Precision;
    fn placement(&self) -> Placement;
    fn require(&self, requirements: &CapabilitySet) -> Result<()>;
    fn scratch(&mut self, request: ScratchRequest) -> Result<ScratchLease<B>>;
    async fn synchronize(&mut self, token: CompletionToken) -> Result<()>;
    async fn transfer(&mut self, request: TransferRequest<B>) -> Result<TransferResult<B>>;
}
```

The facade owns public buffer/view/index abstractions; Burn tensors, device
handles, and autodiff graphs remain private. The operation set must cover shaped
allocation, explicit conversion, pointwise arithmetic/comparison, masked
selection, gather, reductions, prefix scans/compaction, matrix products, tiled
comparisons, and declared random primitives. Scatter operations require unique
destination ownership or an explicit reduction law; they cannot substitute for
simultaneous cloning by racing writes.

`allocate` consumes a checked shape/dtype/placement request and returns owned
storage, failing on limit/overflow/allocation errors. `submit` borrows the buffers
through a prepared operation whose lifetime extends until its completion token;
outputs may not be read or reused early. `scratch` returns a scoped lease from
the reusable arena, with no aliasing of live inputs. `synchronize` reports queued
errors and completion. `transfer` is an explicit copy/move request with source,
destination, byte count, and synchronization metadata; it never silently changes
precision. Buffer shapes and byte sizes are checked with overflow-safe arithmetic.

The initial private adapters target Burn's CPU Flex, CUDA, and WGPU routes.
Burn documents multiple interchangeable compute backends and WebAssembly support.
[Burn backend overview](https://burn.dev/docs/burn/) The Flex source declares
`f32` and `f64` arithmetic and storage support; its runtime dtype dispatch must
be configured explicitly by the adapter. A phantom generic parameter is not
sufficient evidence that an operation executes in `f64`.
[Burn Flex source](https://raw.githubusercontent.com/tracel-ai/burn/main/crates/burn-flex/src/backend.rs)

| Execution profile | Required precision contract | Preparation gate |
|---|---|---|
| Native CPU | `f32` and `f64` | Pinned CPU adapter implements every requested operation in the selected precision |
| Browser WASM CPU | `f32` and `f64` | WASM-compatible build of the CPU adapter, supported imports, memory and operator checks |
| Native CUDA | `f32`; `f64` only with full required support | Both device and adapter/kernel operation capabilities checked; hardware support alone is insufficient |
| Native WGPU | `f32`; additional precision only if explicitly supported | Adapter, shader representation, and every operation must support it |
| Browser WebGPU | `f32` | Runtime WebGPU availability, limits, required features; `f64` requires explicit selection of WASM CPU |

Each custom operator must satisfy the selected profile. Browser WebGPU cannot use
native Vulkan's optional `SHADER_F64`: wgpu documents that feature for native
SPIR-V/Vulkan execution. [wgpu precision features](https://wgpu.rs/doc/wgpu/struct.Features.html#associatedconstant.SHADER_F64)
WebAssembly itself defines both `f32` and `f64` number types; `wasm32` concerns
addressing, not a ban on double precision.
[WebAssembly numeric types](https://webassembly.github.io/spec/core/syntax/types.html#number-types)

`NumericalPolicy` defaults to `f32` and selects one computation precision for a
run. It governs observation/reward extraction, objective orientation, donor-edge
reductions, recombination coefficients, noise factors, scalar parameters, and
derivative providers as well as numerical population buffers. Integer images,
masks, and simulator internals retain separate types. External simulators declare
their own precision. Mixed-precision computation is outside the initial contract.
Import/restore conversions must be explicit and recorded as a new execution
configuration, not an identical replay.

Model parameters such as $\sigma_{\min}$, $\varepsilon_{\rm dist}$,
$\varepsilon_{\rm clone}$, $\eta$, and bandwidths are serialized independently
of dtype. Numerical tolerances describe rounding/error checks. Switching to
`f64` must not silently shrink smoothing, noise, or boundary scales. A parameter
that is positive in the saved specification but rounds to zero in the selected
precision causes a preparation error when positivity is required.

Logical `SlotIndex` uses a checked portable integer representation, normally
`u32`, with counts bounded by device/adapter limits. A private adapter using
signed `i32` indices enforces its smaller range. No universal `int64` tensor is
required, and indices are never encoded in floats. Long step counters, random
addresses, and lineage IDs may use pairs of 32-bit words on browser GPUs and
lossless host serialization; they are not JavaScript floating-point counters.

### 6.2. Parallelism, random streams, and transfer costs

Runtime component configuration is resolved during preparation into a compiled
or statically specialized execution graph. Rust generics and backend kernel
specialization handle numerical loops; configuration enums or type-erased service
objects live outside per-walker loops. Arbitrary Rust closures and host trait
objects do not automatically compile into GPU shaders. A custom component must
provide supported tensor operations or a registered compiled device kernel;
otherwise it is host-only and must be selected as such.

The population uses batched, field-wise storage with contiguous slot axes where
appropriate. A step reuses source/destination buffers, masks, indices, and tile
scratch. Exact dense Gaussian selection can stream candidate tiles and reduce
weights without persistent $N\times N$ storage; avoiding that storage does not
remove its quadratic arithmetic. Exact local standardization has the same
qualification. Approximate neighbors, sparse support, or approximate matching
are separately named algorithm variants, with accuracy and sampling-law changes
recorded.

Prepared donor capacities $K_{\rm dist}$ and $K_{\rm clone}$ are independent.
Fixed-capacity `[N,K]` buffers plus masks are the portable default; capacity
changes require explicit re-preparation and memory checks. A streamed all-eligible
view avoids requiring a resident quadratic donor array. Specialize $K=1$ draws,
comparison/reduction, and copy-only plans; do not impose unused multi-donor
arithmetic on the default run. Multi-donor kernels assign one destination owner
per recipient or use a specified reduction, never racing contributions from
donor edges. Large gathers can be tiled instead of materializing `[N,K,d]`.

Exact dense Gaussian weights can be reused across several draws when the law
and source context permit it. The sampler must report how it reuses streamed
weights or categorical preprocessing; $K$ draws are neither automatically free
nor necessarily $K$ complete quadratic distance passes. Two donor-role instances
may share compatible immutable comparison work, but their draws remain independent.

With $N$ current queries and $M$ pooled source records, exact dense comparison
work is $O(NMd)$; $N^2$ is the current-round special case. Historical retention
and source materialization have explicit byte/transfer budgets. Read-only frame
handles and compact pool-row indices allow partial device residency. Archive
access failures must not change the configured candidate distribution.

| Operation | Arithmetic and parallel dependence | Working memory / synchronization / transfers |
|---|---|---|
| Input/extraction | Provider-dependent; independent branches can run in parallel | Shared derived-field cache; declare global reductions, raw-input staging, and device transfers |
| Selected pair distances and mean | $O(NK_{\rm dist}d)$ comparisons plus $O(NK_{\rm dist})$ reduction; edges parallel | $O(NK_{\rm dist})$ outputs or streamed chunks; single-companion specialization |
| Global moments and fitness | $O(N)$ per channel; reductions then pointwise work | Bounded reduction buffers; device-side stage dependencies |
| Dense independent Gaussian | $O(N^2d)$; independent rows and parallel candidate tiles | Tile scratch, row state, index outputs; no mandatory host readback |
| CPU/WASM Fisher–Yates | $O(k)$ shuffle work; sequential swap chain | $O(k)$ ID storage; native CPU or worker-local execution |
| Multiple matching rounds | $O(Kk)$ CPU shuffle work; rounds independent, each shuffle sequential | $O(NK)$ donor IDs if resident; round identities retained; no union-level disjointness claim |
| Device permutation/matching | Separate registered law-preserving primitive | Must report its actual work, dependency depth, scratch, launches, and limits; no automatic $O(k)$ parallel claim |
| Explicit hybrid Fisher–Yates | CPU shuffle plus adjacent pairing | Upload companion indices; if alive IDs are only on device, also download/compact them with the required synchronization |
| Clone gather / restitution | $O(Nd)$ plus snapshot bytes; slots or disjoint pairs parallel | Separate destination storage; host snapshot stores may require index readback |
| Future multi-donor recombination | For a weighted numerical mix, $O(NK_{\rm clone}d)$ plus explicit proposal/domain evaluation cost | Donor graph and coefficients; tiled source reads; include reward refresh, domain synthesis, and multi-donor lineage costs |
| Diagonal / full noise | $O(Nd)$ / $O(Ndr)$ factor application | Factor storage may dominate; shared factors need not be replicated per walker |
| Environment transitions | Domain-dependent; concurrency declared by adapter | Include feature uploads, action/donor readbacks, and simulator-state copy costs |

A GPU permutation primitive must establish its uniform law, including treatment
of finite random-key ties if using a key-based method. A sort of finite random
keys with fixed-index tie-breaking is not automatically an exact uniform
permutation. Until a verified device primitive exists, users choose explicit
hybrid Fisher–Yates or receive `UnsupportedSamplerOnDevice`; the sequential CPU
shuffle is not advertised as a parallel GPU algorithm. Hybrid mode transfers
indices and eligibility information, not full floating observation arrays solely
for shuffling. Its benchmark still includes those transfers and host work.

The run keeps numerical intermediates on their declared device. Scalar status
readbacks are batched at commit/cancellation/reporting boundaries; per-walker
`.item()` operations or JavaScript dispatch loops are not part of the performance
contract. Actual deaths can require control synchronization, which must be
counted rather than hidden. Parallel reductions can change rounding order, so
determinism settings record reduction strategy, fusion, and backend versions.

Randomness is addressed by

$$
(\text{seed},\text{run},\text{step},\text{operator},\text{role},\text{slot},
\text{donor or round},\text{substep},\text{draw},\text{attempt}).
$$

The RNG algorithm/version and integer-to-real conversion are part of the resolved
configuration. Uniform integer choices avoid modulo bias. Rejection samplers use
per-address attempt counters, not a variable-length global stream that perturbs
other walkers. Permutation swaps use swap positions in a dedicated namespace;
pair operations use canonical pair IDs. Initialization, `distance_donors`,
`cloning_donors`, candidate choice, proposal construction/evaluation, acceptance,
revival, jitter, thermostat, extraction randomness, and diagnostics have distinct
namespaces. $K=1$ uses donor/round address zero; changing the diversity capacity
must not consume the cloning namespace. A law such as sampling without replacement
still has intentional within-row dependencies, which scheduling must preserve.
Workgroup ordering, thread scheduling, and tile traversal must not determine
which random address belongs to a logical draw. A global Burn seed alone does
not establish this contract; private adapters need the addressed random primitive.

Replay is guaranteed only for a specifically supported execution configuration,
including its RNG, arithmetic/reduction modes, device/kernel versions, and
checkpointable domain. Cross-platform bitwise-identical trajectories are not
promised: rounding can change discrete acceptance decisions. Validation separates
fixed-input numerical tolerance checks from distributional comparisons over
independent runs.

### 6.3. Native and browser runtimes

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 22, 'rankSpacing': 25}}}%%
flowchart TD
    Config[Resolved configuration] --> Native[Native Rust]
    Config --> UI[Browser UI]
    UI -->|Messages / snapshots| Worker["Web Worker<br/>Rust WASM"]
    Native --> CPU["CPU<br/>f32 / f64"]
    Native -->|Device transfers| NGPU["CUDA / WGPU<br/>Capability-gated"]
    Worker --> WCPU["WASM CPU<br/>f32 / f64"]
    Worker -->|Uploads / readbacks| WebGPU["WebGPU<br/>f32"]
    Worker -.->|Hybrid index exchange| WebGPU
    Worker --> Storage["Browser<br/>storage"]
    Native --> Files["Native<br/>storage"]
```

The browser package uses a thin JavaScript/WASM binding and a Web Worker owning
one run. Its public control contract is asynchronous initialization, admitted
versioned input batches/`step_with_input`, bounded `run_steps`, cancellation,
observation and raw-reward snapshots, full checkpoints, and restore.
Messages contain request IDs and schema/versioned payloads, not public Burn
objects. ArrayBuffer transfers/copies are explicit; full populations are not
sent to the UI after every substep. UI rendering consumes a configured sampling
of observations, raw rewards/objective direction, and scalar diagnostics.
Input messages carry slot IDs, request/stage versions, and declared transfer
ownership; detached or stale input buffers cannot be reused after admission.

Initialization detects WebGPU in the actual worker/browser context, obtains an
adapter/device, and checks storage-buffer limits, binding/workgroup limits,
required operations, and the full planned working set. WebGPU requires a secure
context; a missing API/device, device loss, unsupported shader operation, or
allocation failure returns a typed capability/runtime error.
[WebGPU API requirements](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
There is no silent CPU fallback and no automatic `f64` to `f32` conversion.
The user may explicitly prepare a WASM CPU run instead.

The CPU build must obey the imports and limitations of the Rust WASM target;
native filesystem access and native threading APIs cannot simply be assumed to
work in a browser. [Rust WASM target](https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html)
Optional threaded WASM requires a compatible atomics/shared-memory build, a worker
pool, `SharedArrayBuffer`, and a secure cross-origin-isolated deployment, normally
using suitable COOP/COEP headers. An unthreaded worker build remains an explicit
supported route when these requirements are absent.
[Shared memory deployment requirements](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements)
Threaded mode must be requested or selected by a recorded deployment policy;
WASM execution is not automatically multithreaded.

Preparation budgets WASM linear memory, GPU buffers, double-buffered populations,
opaque snapshots, and checkpoint staging. Limits are detected, not hardcoded from
a desktop GPU assumption. Cancellation yields between bounded dispatch batches;
a busy worker must return to its event loop to receive control messages in the
non-shared-memory deployment. A time budget is cooperative, not a hard deadline
that can interrupt a submitted kernel or simulator call.

### 6.4. History and replay checkpoints

```rust
pub trait HistorySink<B: ComputeBackend> {
    async fn record(&mut self, record: StageRecord<'_, B>) -> Result<RecordReceipt>;
    async fn flush(&mut self) -> Result<()>;
}

pub trait CheckpointCodec<B: ComputeBackend> {
    async fn capture(&self, run: &CommittedRunView<'_, B>) -> Result<Checkpoint>;
    async fn restore(&self, checkpoint: Checkpoint, target: &ExecutionRequest)
        -> Result<RestoredRun<B>>;
}
```

`HistorySink::record` borrows a stage-tagged view until its returned receipt
confirms safe buffer reuse; an asynchronous sink must retain an immutable handle
or copy selected fields. Recorded arrays have shape `[n_recorded,N,...]`, or an
explicit sampled-slot axis with slot IDs. Backpressure policy is configured:
block at a safe point, drop allowed diagnostics with counters, or fail; never
silently drop mandatory lineage/checkpoint records. Transfer/storage failures
are surfaced. History observation records alone are not sufficient for replay.

`Checkpoint` is an owned, versioned artifact containing:

| State class | Required contents |
|---|---|
| Identity and configuration | Format version; all component identifiers/configs; extractor dependencies; objective direction; both donor-module configs/cardinalities/laws; reducer and regularization order; clone proposal/decision/recombination policies; mathematical and execution parameters |
| Input and extraction | Admitted external/domain input or immutable archived references with integrity checks; slot/stage alignment; input-stream cursor; derived-field provenance and persistent extractor state |
| Historical donor pools | Frame/slot/generation/version identities, source eligibility, retention/source-selection policies, and the numerical/domain archive needed to resolve requested donors; checksums alone do not replace missing state |
| Population | First-class observations and raw `[N]` rewards with shapes/dtypes/units, validity and source versions; masks/reason codes; slot and lineage IDs; actions/time where required |
| Donor and lineage records | Both role identities, retained `[N,K]` pool-row draws/masks/topology, resolved current/historical source identities, and actual contributing donors/coefficients; provenance is not an ancestry filter |
| Opaque domain | Snapshot archive, domain version, RNG and simulator state needed to resume, observation consistency tokens |
| Stochastic/operator state | Root seed, stream schema and counters, step/substep position, persistent noise/policy/provider memory with cloning categories |
| Replay environment | Reduction/fusion modes, kernel/RNG versions, recorded domain determinism limits, and required device operations |
| Integrity | Lengths, checksums, and compatibility metadata; restored content is validated before execution |

The initial checkpoint boundary is a fully committed step, not an arbitrary
partially executed BAOAB stage. `capture` synchronizes necessary buffers and
serializes only after observation/state consistency is established. `restore`
consumes an archive, validates formats and capabilities, and prepares all caches
afresh unless their validity is proven. Incompatible dtype/backend restoration
requires explicit migration mode and yields a new execution identity, not an
exact replay claim. No dynamic code embedded in an archive is executed.

Native storage adapters may use files; browser adapters may use IndexedDB or a
download/upload artifact. Browser storage quotas, permission failures, eviction,
and unavailable simulator serialization are explicit errors or limitations.
All public `capture`/`restore` paths use the same checkpoint contract. A future
adapter to the existing Python `RunHistory`/`FractalSet` formats must preserve
stage/lineage semantics and label any information those formats cannot express;
it is not required to expose Burn tensors or modify those existing classes.

(sec-algorithmic-gas-validation)=
## 7. Worked configurations and validation criteria

The reference configurations cover position-only dynamics, phase-space Langevin
dynamics, and opaque simulator state. Validation separates interface contracts,
fixed-input numerical agreement, and sampling distributions.

Matching seeds do not establish a correct distribution. Conversely, floating-point
differences can change later clone decisions without establishing an implementation
error. Tests therefore check ownership and topology directly, numerical results
with justified tolerances, and stochastic behavior with appropriate statistical
uncertainty.

### 7.1. Position-only Gaussian jumps

This configuration is a complete numerical reference profile, not executable
configuration syntax or a claim of convergence. Coordinates and reward are
nondimensional. All omitted optional transforms/providers are disabled.

| Component | Selected configuration |
|---|---|
| Execution and initialization | Native CPU `f32`; $N=1024$, position `[N,2]`; seed `17`; independent uniform initialization on $[-1,1]^2$ |
| Input and extractors | Input `position: [N,2]`; identity position observation; raw reward extractor $r_i=U(x_i)=\lVert x_i\rVert^2/2$; state-free numerical domain, no velocity or opaque state |
| Boundary | Absorbing closed box $[-5,5]^2$, finite checks, default external-flag policy |
| Distance and kernel | Position Euclidean distance; Gaussian width $\epsilon=1$ |
| Donor modules | Distinct `distance_donors` and `cloning_donors`, each `CurrentEligible`, Gaussian independent with replacement, $K=1$, separate streams; no ancestry filter, self only for singleton |
| Reward orientation/diversity | `Minimize` raw $U$, giving oriented $-U$ at `PreCloneFitness`; `MeanDistance` on $K_{\rm dist}=1$, then $h_i=\sqrt{d_i^2+10^{-6}}$ |
| Standardization/map/combine | Global alive population moments per channel; $\sigma_{\min}=10^{-3}$ in each channel's units; logistic $A=2$, additive $\eta=10^{-6}$; $\alpha=\beta=1$ |
| Clone proposal/decision/transform | `CopyProposal`, relative acceptance $p_{\rm sat}=1$, $\varepsilon_{\rm clone}=10^{-6}$; literal copy; uniform pre-clone revival donors; no clone jitter |
| Kinetic/noise | Direct jump $x'=x+0.05\xi$, Gaussian law, isotropic geometry $L=I_2$; no drift, derivatives, or implicit time scaling |
| Stopping and history | Default extinction halt/singleton policy; at most 1000 steps; lineage every step, post-kinetic positions every 10 steps, checkpoint every 100 committed steps |

For equal measurements, each mapped channel equals $1+10^{-6}$ and the fitness
is $(1+10^{-6})^2$. Equal-fitness companions cannot trigger accepted alive clones.
The jump covariance is $0.05^2I_2=0.0025I_2$ per step, independently of any
displayed wall-clock interval. A jump outside the box is recorded as ineligible
and awaits the next normal revival stage. Switching this profile to `f64`
changes the numerical policy, not $0.05$ or any model regularizer.

A controlled multi-companion variant changes only `distance_donors` to
$K_{\rm dist}=4$, retaining its independent with-replacement law and mean-then-
regularize order. `cloning_donors` stays at $K_{\rm clone}=1$. If a walker's
four raw distances are $1,2,2,3$, its mean is $2$ and its diversity measurement
is $\sqrt{4+10^{-6}}$; the repeated distance is a valid sample, not padding.
This variant leaves direct-jump covariance unchanged but changes the sampled
fitness process and comparison cost. Raw objective history still reports $U$,
not the negated oriented score.

### 7.2. Phase-space Langevin with anisotropic noise and mutual pairs

| Component | Selected configuration |
|---|---|
| Execution and initialization | Native CPU `f64`; $N=512$, position/velocity `[N,2]`; seed `23`; positions uniform on $[-1,1]^2$, velocities zero |
| Domain/boundary | Nondimensional, state-free phase space; unbounded with finite checks |
| Input and extractors | Named position/velocity fields; phase-space observation extractor; raw $r_i=U(x_i)$ from a shared analytic evaluation; `Minimize`; velocity is available for alternative reward extractors |
| Distance and kernels | Phase-space distance, $\lambda_{\rm alg}=1$; uniform interaction kernels |
| Donor modules | Distinct `distance_donors` and `cloning_donors`, both `CurrentEligible`, uniform mutual matching via CPU Fisher–Yates, $K=1$, separate streams; odd last slot self, no ancestry filter; uniform independent revival donors |
| Fitness | Oriented score $-U(x)$, $U(x)=\lVert x\rVert^2/2$; single-entry mean then regularized distance $\varepsilon_{\rm dist}=10^{-3}$; global statistics, $\sigma_{\min}=10^{-3}$; logistic $A=2$, $\eta=10^{-6}$; $\alpha=\beta=1$ |
| Cloning | `CopyProposal`, relative acceptance $p_{\rm sat}=1$, $\varepsilon_{\rm clone}=10^{-6}$; accepted-recipient position jitter $0.01\xi$ with its own isotropic Gaussian source; pairwise restitution $a=0.8$; revival is literal pre-clone copy |
| Kinetic and derivatives | BAOAB with $h=0.01$, scalar $\gamma=1$, analytic gradient $\nabla U=x$; new gradient at each B input; no Hessian required |
| Kinetic noise | Independent Gaussian innovations and global full factor $B=\left(\begin{smallmatrix}0.2&0\\0.1&0.3\end{smallmatrix}\right)$, injected through the diffusion/noise-geometry adapter |
| Stopping and history | Default extinction/singleton rules; at most 1000 steps; lineage every step, positions/velocities and pre-clone fitness every 10, checkpoint every 100; debug substep flags enabled |

Here

$$
BB^\top=\begin{pmatrix}0.04&0.02\\0.02&0.10\end{pmatrix},\qquad
\operatorname{Cov}(\Delta v_{\rm O}\mid\text{O inputs})
=\frac{1-e^{-0.02}}2
\begin{pmatrix}0.04&0.02\\0.02&0.10\end{pmatrix}.
$$

This is anisotropic and correlated across coordinate directions, but independent
across walkers. The matrix is a diffusion factor, so the thermostat supplies
the time scaling exactly once. Clone jitter is a distinct source and stream.
Replacing matching by directed independent companions while retaining restitution
must fail compatibility checks. Running on a GPU requires an eligible precision
profile and either a validated device matching primitive or explicitly selected
hybrid Fisher–Yates. This anisotropic cloning process makes no automatic
thermal-equilibrium assertion.

A future recombination profile can select two current-round donors, disable
restitution/jitter, and mix positions and velocities with equal coefficients.
Donor positions $(0,0)$ and $(2,0)$ produce $(1,0)$, with objective $U=0.5$ rather
than the mean donor objective $1$. Evaluate the proposed inputs and configure
proposal fitness, acceptance, memory/metadata policies, and a domain recombiner
if states are present. Missing capabilities fail preparation. Record both donor
sources without ancestry restrictions.

### 7.3. Opaque simulator states with numerical features

| Component | Selected configuration |
|---|---|
| Execution and initialization | Browser worker, explicit WASM CPU `f32`; $N=256$, seed `31`; snapshot-capable simulator initializes independent episodes with recorded seeds |
| Input and observation extractor | Raw image `[N,3,84,84]` in `u8`, optional RAM, score, termination flags; version-pinned deterministic encoder returns `[N,128]` in `f32`; distance selects features only |
| Raw reward extractor | Cumulative episode reward from the shared input, initialized to zero and copied as trajectory data; `[N]` output with `Maximize` |
| Domain and state | Host/worker `StateStore` with complete simulator snapshots and RNG; adapter restores, advances, saves, and encodes; no coordinate-edit capability |
| Boundary | Finite feature validation plus external termination/truncation, both excluded; no geometric repair |
| Distance/kernel/donor modules | Cosine with the specified zero rules; `ExponentialDissimilarity` $\tau=1$; distinct `distance_donors` and `cloning_donors`, `CurrentEligible`, independent directed weighted draws, each $K=1$, separate streams; no ancestry filter, singleton self |
| Fitness | Raw reward oriented for maximization; single-entry dissimilarity reduction, then $\sqrt{q_i^2+10^{-6}}$; global alive moments, $\sigma_{\min}=10^{-3}$; logistic $A=2$, additive $\eta=10^{-6}$; reward exponent $1$, diversity exponent $0.5$ |
| Cloning | `CopyProposal`, relative acceptance $p_{\rm sat}=1$, $\varepsilon_{\rm clone}=10^{-6}$; literal numerical/state copy; independent uniform revival donors; no jitter or restitution |
| Kinetic/control/noise | One simulator transition per gas step; version-pinned deterministic feature-to-action policy, lowest-index action on ties; external simulator owns transition randomness, no numerical noise source consumed |
| Recording and stopping | Default extinction/singleton rules; 1000-step maximum; lineage/flags/reward every step, sampled images/features every 20 steps, full checkpoint every 100; browser storage adapter explicitly selected |

The environment-transition provider set includes the action policy's input/output
schema (`[n,128]` to a declared integer action `[n]` here), version, precision,
placement, failure behavior, and optional recurrent state category. All simulator
imports must be browser-compatible; compiling the engine to WASM does not port
a native-only simulator.

If slot `7` terminates and slot `12` remains alive, revival copies slot `12`'s
snapshot, feature/image observation, cumulative reward, and per-lineage policy
memory from one source version. Subsequent simulator transitions restore separate
mutable states for the two slots. Selection cannot inspect timers or RNG state
inside those snapshots. If simulator RNG is donor-copied and the policy is
deterministic, identical continuation is possible; an explicit domain reseeding
or stochastic-control policy is needed if independent divergence is desired.

Adding Euclidean jitter to these image features fails `UnsupportedDomainEdit`;
changing an embedding does not move the simulator. A future WebGPU selection
profile must explicitly account for encoded-feature uploads and action/donor
readbacks. Simulator state remains at the adapter's declared placement rather
than being implicitly transferred to the GPU.

A RAM-based variant replaces only observation extraction: it returns selected
positions and a declared encoding of level, while reward still comes from score
or another independent scalar computation. Its distance must specify how level
and position are compared. A historical-cloning variant additionally configures
a retained snapshot window and compatible score re-evaluation; an earlier saved
walker from any lineage is eligible under that policy. A collection of old RGB
frames without full simulator snapshots is insufficient for cloning.

### 7.4. Acceptance matrix and publication verification

The following are requirements for the implementation test suite. Some are covered
by the initial Rust tests; they are not all completed acceptance evidence.

| Test family | Required acceptance evidence |
|---|---|
| Component contracts | Reject wrong view, rank, dtype, units, missing velocity, stale versions, unsupported operation, or invalid index before corrupting a destination |
| Input and extraction | Shared input produces independent tensor observations and scalar `[N]` rewards; identity extraction, RAM features, RGB tensors, function/velocity and population-dependent rewards; wrong shapes, slot alignment, and dependency cycles fail |
| Shared evaluation/cache | Expensive intermediate evaluated once per valid key; refresh after cloning, recombination, repair, or velocity change; global dependencies invalidate all affected slots; unavailable refresh cannot be labelled current |
| Objective direction | Raw history retained; minimizing $U$ matches maximizing $-U$ under fixed comparable inputs, with exactly one sign reversal; disabled reward fitness does not erase raw reward access |
| Donor-role independence | Two instances can use different pools, kernels, laws, and capacities; changing $K_{\rm dist}$ does not consume cloning RNG; $K=1$ preserves the specified baseline law |
| Current/historical eligibility | Donors from unrelated lineages remain eligible by default; resolve frame/slot/generation identities; test archive retention, exact-snapshot self rules, historical-score re-evaluation, and rejection of missing domain snapshots |
| Multi-companion reduction | `[N,K]` masks/counts, replacement duplicates, scarce candidates, repeated matching rounds, weighted mean validation, mean distance versus distance to mean, and regularization-order distinction |
| Matching structure | Check eligibility, self/odd/singleton rules, disjointness/involution certificates, and legacy permutation cycles separately |
| Matching distributions | Enumerate small-population expected laws; test uniform matching frequencies, independent row probabilities and dependence, broad-Gaussian limit, pivot-dependent greedy law, and separate diversity/clone streams |
| Simultaneous cloning | Chains, swaps, repeated donors, mixed alive/dead recipients, donor-side restitution, separate destination buffers, and matching numerical/opaque source versions |
| Multi-donor cloning | Distinguish selecting one candidate from genuine multi-donor contribution; frozen current/historical source reads, fresh offspring reward, complete donor provenance, coefficient/memory policies, incompatible restitution rejection, and unsupported opaque recombination errors |
| Boundaries and timing | All four causes preserved; periodic wrap and minimum-image agreement; clone-jitter deaths, substep deaths, no provider calls on parked slots, default extinction and theoretical $k<2$ stopping distinguished |
| Fitness degeneracy | Constant/one-alive populations, excluded NaNs, large magnitudes, zero-support local fallback, disabled exponents, additive floors, and exact legacy sample-statistics degeneracy |
| Cosine | Two zero vectors, one zero vector, collinear/opposite vectors, rounding bounds, declared near-zero policy, and derivative restrictions |
| Noise geometry | Empirical conditional mean/covariance for scalar, per-walker, diagonal, full, low-rank, and state-dependent factors; reject malformed covariance inputs and inappropriate moment claims |
| Noise distribution/coupling | Distinguish isotropic covariance from rotational invariance; check cross-walker independence, separate streams, stateful clone memory, and heavy-tail-appropriate uncertainty |
| Langevin scaling | Isolated Gaussian O-stage covariance for several $h,\gamma$ including $\gamma\to0$; no double $\sqrt h$; fresh second B gradient; non-Gaussian thermostat labelled as a different operator |
| Precision propagation | Inspect intermediate/output dtypes, constants, reductions, factors and derivatives; fixed-parameter `f32`/`f64` comparisons; reject unintended narrowing or unrepresentable regularizers |
| Replay and cancellation | Same supported execution resumes from a committed checkpoint; independent streams unaffected by optional diagnostics; domain/operator memory restored; bounded cancellation exposes no partial successor |
| Capability failures | Browser GPU `f64`, unsupported device matching, missing factorization, native-only domain imports, insufficient memory, unavailable shared memory, device loss, and storage quota failures produce explicit outcomes |

Distribution tests must state sample sizes, statistical error control, and seed
sets; a single matching sequence is not proof of the intended law. Covariance
tests condition on fixed inputs/factors and use uncertainty methods appropriate
to the innovation moments. Numerical tolerances are justified per operation and
precision, not chosen to force agreement of chaotic long trajectories.
Convergence or invariant-measure claims require separate proofs under the exact
resolved algorithm and domain hypotheses.

Performance acceptance sweeps $N$, source-pool size $M$, $K_{\rm dist}$,
$K_{\rm clone}$, feature dimension, tile size, noise rank, precision, alive
fraction, and platform. Report warm-up/compilation separately
from steady-state steps, and report end-to-end latency as well as device timings.
For each configuration measure arithmetic scaling, dependency depth/launches,
peak persistent/scratch/checkpoint memory, synchronization count, uploaded and
downloaded bytes, host simulator/shuffle time, and recording overhead. Dense
Gaussian selectors should exhibit quadratic comparison work without mandatory
quadratic storage; uniform CPU matching should exhibit linear shuffle work.
No universal GPU speedup is an acceptance criterion.

The documentation publication gate is separate from the engine tests:

1. Both `docs/_toc.yml` and `docs/myst.yml` register this file once, as the
   standalone **Architecture** chapter immediately after Algorithms and
   Foundations, with matching publication order.
2. The chapter has seven labelled major sections, resolved document/proof links,
   six Mermaid diagrams, and signature-only API sketches. Existing Python and
   C++ algorithm sources remain unchanged; new tests live in the independent
   Rust workspace and browser lab.
3. Run the repository's documentation publication checks in
   `tests/docs/test_book_tools.py`, and build the Theory site with the book's
   configured extensions and navigation.
4. Inspect the page and diagrams in Full and Expert Mode. All chapter content,
   including explanations, definitions, API contracts, configurations, validation
   criteria, and diagrams, must remain visible in both modes.

(sec-algorithmic-gas-implementation-status)=
### 7.5. Initial implementation status

`algorithmic-gas/` is a separate Cargo workspace with engine, analytic benchmark
and WASM crates. `fractal-gas-web/web/euclidean-gas/` is its separate browser lab.
Neither replaces the existing Python engines or C++ Optimization Lab.

| Area | Initial implementation |
|---|---|
| Data and extraction | Typed tensor observations, scalar raw rewards, named external inputs, shared derivation followed by independent extractors, opaque snapshots and domain reconciliation |
| Modules | Independent donor configurations/streams; Euclidean, phase-space and cosine comparisons; uniform/Gaussian kernels; independent, Fisher–Yates, greedy and legacy sampling; multiple distance companions; single-donor cloning |
| Dynamics | Global/local fitness, boundaries, simultaneous copying, jitter/restitution, direct/Brownian jumps, BAOAB, independent Gaussian/uniform noise with isotropic/diagonal/full/low-rank factors |
| Extensions | Batch-level `GasOperators` hooks plus extraction, reward, domain, distance, noise and derivative interfaces; optional derivative providers are not all used by built-in integrators |
| Precision and execution | Native/WASM CPU `f32` and `f64`; feature-gated CUDA and WGPU adapters; browser WebGPU `f32`; explicit precision/capability failures |
| Recording | Versioned committed-step reports and CBOR checkpoints with population, configuration, donor archive, source identities and provider IDs |
| Browser | Worker execution, bounded stepping/pause, configuration controls, 2D projection, convergence chart, walker inspector, JSON export, binary/IndexedDB checkpoint restore |

The accelerator path is **host-orchestrated**, not device-resident: Burn executes
batched objective/gradient evaluation, pair reductions, noise factors and kinetic
arithmetic. Sampling control, fitness statistics/maps, cloning and RNG currently
run on the host. Uploads, readbacks and synchronization are counted. This does
not satisfy the target of persistent device populations, reusable scratch buffers
or index-only hybrid Fisher–Yates transfers. No universal GPU speedup is claimed.

Historical cloning currently requires global statistics and no external per-slot
input batch. Unsupported source alignment, local historical scoring, historical
mutual pairing and true multi-donor combination fail explicitly. Custom providers
must be stateless for replay or keep persistent per-walker state in domain
snapshots; generic custom-operator serialization is not yet implemented.

The Rust and WASM test suites exercise implemented contracts. GPU compilation
checks are not GPU runtime or numerical-conformance tests. Remaining acceptance
work includes device-resident execution, total-memory planning, accelerator
runtime/distribution comparisons, performance sweeps, cached dependency graphs,
threaded WASM and the explicitly optional extension families above. The workspace
guide records these limits alongside runnable commands.
