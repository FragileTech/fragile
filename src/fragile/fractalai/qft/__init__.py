"""QFT utilities plus simulation and analysis pipelines for Fractal Gas runs."""

from fragile.fractalai.qft.smoc_pipeline import (
    AggregatedCorrelator,
    ChannelProjector,
    CorrelatorComputer,
    CorrelatorConfig,
    MassExtractionConfig,
    MassExtractor,
    ProjectorConfig,
    SimulationConfig,
    SMoCPipelineConfig,
    SMoCPipelineResult,
    SMoCSimulator,
    aggregate_correlators,
    compute_smoc_correlators_from_history,
    run_smoc_pipeline,
)

from fragile.fractalai.qft.mass_correlator_plots import (
    STANDARD_CHANNELS,
    ChannelCorrelatorResult as ChannelCorrelatorResultLegacy,
    ChannelDefinition,
    MassCorrelatorComputer,
    MassCorrelatorConfig,
    MassCorrelatorPlotter,
    build_mass_correlator_dashboard,
    compute_all_channel_correlators,
    save_mass_correlator_plots,
)

from fragile.fractalai.qft.correlator_channels import (
    # Config and results
    ChannelConfig,
    ChannelCorrelatorResult,
    # AIC extractor
    ConvolutionalAICExtractor,
    # FFT correlator
    compute_correlator_fft,
    compute_effective_mass_torch,
    # Base classes
    ChannelCorrelator,
    BilinearChannelCorrelator,
    TrilinearChannelCorrelator,
    GaugeChannelCorrelator,
    # Concrete channels
    ScalarChannel,
    PseudoscalarChannel,
    VectorChannel,
    AxialVectorChannel,
    TensorChannel,
    NucleonChannel,
    GlueballChannel,
    # Registry and factory
    CHANNEL_REGISTRY,
    compute_all_channels,
    get_channel_class,
)
from fragile.fractalai.qft.voronoi_time_slices import (
    compute_time_sliced_voronoi,
    TimeSliceResult,
    TimeSlicedVoronoi,
)
from fragile.fractalai.qft.higgs_observables import (
    HiggsConfig,
    HiggsObservables,
    compute_higgs_observables,
    compute_emergent_metric,
    compute_centroid_displacement,
    compute_geodesic_distances,
    compute_higgs_action,
)
from fragile.fractalai.qft.higgs_plotting import (
    build_all_higgs_plots,
    build_metric_tensor_heatmap,
    build_centroid_vector_field,
    build_ricci_scalar_distribution,
    build_geodesic_distance_scatter,
    build_higgs_action_summary,
    build_volume_vs_curvature_scatter,
    build_scalar_field_map,
    build_metric_eigenvalues_distribution,
)
from fragile.fractalai.qft.radial_channels import (
    RadialChannelBundle,
    RadialChannelConfig,
    RadialChannelOutput,
    compute_radial_channels,
)
from fragile.fractalai.qft.quantum_gravity import (
    QuantumGravityConfig,
    QuantumGravityObservables,
    QuantumGravityTimeSeries,
    compute_quantum_gravity_observables,
    compute_quantum_gravity_time_evolution,
)
from fragile.fractalai.qft.quantum_gravity_plotting import (
    build_all_gravity_plots,
    build_all_quantum_gravity_time_series_plots,
)

__all__ = [
    # SMoC pipeline
    "AggregatedCorrelator",
    "ChannelProjector",
    "CorrelatorComputer",
    "CorrelatorConfig",
    "MassExtractionConfig",
    "MassExtractor",
    "ProjectorConfig",
    "SimulationConfig",
    "SMoCPipelineConfig",
    "SMoCPipelineResult",
    "SMoCSimulator",
    "aggregate_correlators",
    "compute_smoc_correlators_from_history",
    "run_smoc_pipeline",
    # Mass correlator plots (legacy)
    "STANDARD_CHANNELS",
    "ChannelCorrelatorResultLegacy",
    "ChannelDefinition",
    "MassCorrelatorComputer",
    "MassCorrelatorConfig",
    "MassCorrelatorPlotter",
    "build_mass_correlator_dashboard",
    "compute_all_channel_correlators",
    "save_mass_correlator_plots",
    # Correlator channels (new vectorized)
    "ChannelConfig",
    "ChannelCorrelatorResult",
    "ConvolutionalAICExtractor",
    "compute_correlator_fft",
    "compute_effective_mass_torch",
    "ChannelCorrelator",
    "BilinearChannelCorrelator",
    "TrilinearChannelCorrelator",
    "GaugeChannelCorrelator",
    "ScalarChannel",
    "PseudoscalarChannel",
    "VectorChannel",
    "AxialVectorChannel",
    "TensorChannel",
    "NucleonChannel",
    "GlueballChannel",
    "CHANNEL_REGISTRY",
    "compute_all_channels",
    "get_channel_class",
    "compute_time_sliced_voronoi",
    "TimeSliceResult",
    "TimeSlicedVoronoi",
    # Higgs field observables
    "HiggsConfig",
    "HiggsObservables",
    "compute_higgs_observables",
    "compute_emergent_metric",
    "compute_centroid_displacement",
    "compute_geodesic_distances",
    "compute_higgs_action",
    "build_all_higgs_plots",
    "build_metric_tensor_heatmap",
    "build_centroid_vector_field",
    "build_ricci_scalar_distribution",
    "build_geodesic_distance_scatter",
    "build_higgs_action_summary",
    "build_volume_vs_curvature_scatter",
    "build_scalar_field_map",
    "build_metric_eigenvalues_distribution",
    # Radial channel correlators
    "RadialChannelBundle",
    "RadialChannelConfig",
    "RadialChannelOutput",
    "compute_radial_channels",
    # Quantum gravity observables
    "QuantumGravityConfig",
    "QuantumGravityObservables",
    "QuantumGravityTimeSeries",
    "compute_quantum_gravity_observables",
    "compute_quantum_gravity_time_evolution",
    "build_all_gravity_plots",
    "build_all_quantum_gravity_time_series_plots",
]
