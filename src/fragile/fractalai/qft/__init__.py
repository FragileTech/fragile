"""QFT utilities plus simulation and analysis pipelines for Fractal Gas runs."""

from fragile.fractalai.qft.aggregation import (
    # Main aggregation function
    aggregate_time_series,
    # Data structures
    AggregatedTimeSeries,
    # Time series binning
    bin_by_euclidean_time,
    # Gamma matrices
    build_gamma_matrices,
    compute_all_operator_series,
    compute_axial_vector_operators,
    # Component functions
    compute_color_states_batch,
    compute_glueball_operators,
    compute_nucleon_operators,
    compute_pseudoscalar_operators,
    # Channel-specific operator functions
    compute_scalar_operators,
    compute_tensor_operators,
    compute_vector_operators,
    estimate_ell0,
    OperatorTimeSeries,
)
from fragile.fractalai.qft.correlator_channels import (
    AxialVectorChannel,
    BilinearChannelCorrelator,
    bootstrap_correlator_error,
    # Registry and factory
    CHANNEL_REGISTRY,
    # Config and results
    ChannelConfig,
    # Base classes
    ChannelCorrelator,
    ChannelCorrelatorResult,
    compute_all_channels,
    compute_all_correlators,
    compute_channel_correlator,
    # FFT correlator
    compute_correlator_fft,
    compute_effective_mass_torch,
    # AIC extractor
    ConvolutionalAICExtractor,
    CorrelatorConfig as CorrelatorAnalysisConfig,  # Avoid conflict with SMoC pipeline
    # Pure function API (new)
    extract_mass_aic,
    extract_mass_linear,
    GaugeChannelCorrelator,
    get_channel_class,
    GlueballChannel,
    NucleonChannel,
    PseudoscalarChannel,
    # Concrete channels
    ScalarChannel,
    TensorChannel,
    TrilinearChannelCorrelator,
    VectorChannel,
)
from fragile.fractalai.qft.dirac_spectrum import (
    build_antisymmetric_kernel,
    classify_walkers,
    compute_dirac_spectrum,
    DiracSpectrumConfig,
    DiracSpectrumResult,
    SectorSpectrum,
)
from fragile.fractalai.qft.dirac_spectrum_plotting import (
    build_all_dirac_plots,
)
from fragile.fractalai.qft.einstein_equations import (
    compute_einstein_test,
    EinsteinConfig,
    EinsteinTestResult,
)
from fragile.fractalai.qft.einstein_equations_plotting import (
    build_all_einstein_plots,
)
from fragile.fractalai.qft.electroweak_observables import (
    antisymmetric_singular_values,
    build_phase_space_antisymmetric_kernel,
    compute_fitness_gap_distribution,
    compute_higgs_vev_from_positions,
    compute_weighted_electroweak_ops_vectorized,
    pack_neighbor_lists,
    pack_neighbors_from_edges,
    PackedNeighbors,
    predict_yukawa_mass_from_fitness,
)
from fragile.fractalai.qft.higgs_observables import (
    compute_centroid_displacement,
    compute_emergent_metric,
    compute_geodesic_distances,
    compute_higgs_action,
    compute_higgs_observables,
    HiggsConfig,
    HiggsObservables,
)
from fragile.fractalai.qft.higgs_plotting import (
    build_all_higgs_plots,
    build_centroid_vector_field,
    build_geodesic_distance_scatter,
    build_higgs_action_summary,
    build_metric_eigenvalues_distribution,
    build_metric_tensor_heatmap,
    build_ricci_scalar_distribution,
    build_scalar_field_map,
    build_volume_vs_curvature_scatter,
)
from fragile.fractalai.qft.isospin_channels import (
    compute_isospin_channels,
    ISOSPIN_CHANNEL_SPLITTINGS,
    ISOSPIN_MASS_RATIOS,
    IsospinChannelResult,
)
from fragile.fractalai.qft.mass_correlator_plots import (
    build_mass_correlator_dashboard,
    ChannelCorrelatorResult as ChannelCorrelatorResultLegacy,
    ChannelDefinition,
    compute_all_channel_correlators,
    MassCorrelatorComputer,
    MassCorrelatorConfig,
    MassCorrelatorPlotter,
    save_mass_correlator_plots,
    STANDARD_CHANNELS,
)
from fragile.fractalai.qft.neighbor_analysis import (
    # Individual neighbor methods
    compute_companion_batch,
    compute_full_neighbor_matrix,
    # Neighbor computation functions
    compute_neighbor_topology,
    compute_neighbors_auto,
    compute_recorded_neighbors_batch,
)
from fragile.fractalai.qft.quantum_gravity import (
    compute_quantum_gravity_observables,
    compute_quantum_gravity_time_evolution,
    QuantumGravityConfig,
    QuantumGravityObservables,
    QuantumGravityTimeSeries,
)
from fragile.fractalai.qft.quantum_gravity_plotting import (
    build_all_gravity_plots,
    build_all_quantum_gravity_time_series_plots,
)
from fragile.fractalai.qft.radial_channels import (
    compute_radial_channels,
    RadialChannelBundle,
    RadialChannelConfig,
    RadialChannelOutput,
)
from fragile.fractalai.qft.smoc_pipeline import (
    aggregate_correlators,
    AggregatedCorrelator,
    ChannelProjector,
    compute_smoc_correlators_from_history,
    CorrelatorComputer,
    CorrelatorConfig,
    MassExtractionConfig,
    MassExtractor,
    ProjectorConfig,
    run_smoc_pipeline,
    SimulationConfig,
    SMoCPipelineConfig,
    SMoCPipelineResult,
    SMoCSimulator,
)
from fragile.fractalai.qft.voronoi_time_slices import (
    compute_time_sliced_voronoi,
    TimeSlicedVoronoi,
    TimeSliceResult,
)


__all__ = [
    "CHANNEL_REGISTRY",
    # Isospin channel correlators
    "ISOSPIN_CHANNEL_SPLITTINGS",
    "ISOSPIN_MASS_RATIOS",
    # Mass correlator plots (legacy)
    "STANDARD_CHANNELS",
    # SMoC pipeline
    "AggregatedCorrelator",
    # Time series aggregation (new)
    "AggregatedTimeSeries",
    "AxialVectorChannel",
    "BilinearChannelCorrelator",
    # Correlator channels (new vectorized)
    "ChannelConfig",
    "ChannelCorrelator",
    "ChannelCorrelatorResult",
    "ChannelCorrelatorResultLegacy",
    "ChannelDefinition",
    "ChannelProjector",
    "ConvolutionalAICExtractor",
    "CorrelatorAnalysisConfig",
    "CorrelatorComputer",
    "CorrelatorConfig",
    # Dirac spectrum analysis
    "DiracSpectrumConfig",
    "DiracSpectrumResult",
    # Einstein equation verification
    "EinsteinConfig",
    "EinsteinTestResult",
    "GaugeChannelCorrelator",
    "GlueballChannel",
    # Higgs field observables
    "HiggsConfig",
    "HiggsObservables",
    "IsospinChannelResult",
    "MassCorrelatorComputer",
    "MassCorrelatorConfig",
    "MassCorrelatorPlotter",
    "MassExtractionConfig",
    "MassExtractor",
    "NucleonChannel",
    "OperatorTimeSeries",
    # Vectorized electroweak observables
    "PackedNeighbors",
    "ProjectorConfig",
    "PseudoscalarChannel",
    # Quantum gravity observables
    "QuantumGravityConfig",
    "QuantumGravityObservables",
    "QuantumGravityTimeSeries",
    # Radial channel correlators
    "RadialChannelBundle",
    "RadialChannelConfig",
    "RadialChannelOutput",
    "SMoCPipelineConfig",
    "SMoCPipelineResult",
    "SMoCSimulator",
    "ScalarChannel",
    "SectorSpectrum",
    "SimulationConfig",
    "TensorChannel",
    "TimeSliceResult",
    "TimeSlicedVoronoi",
    "TrilinearChannelCorrelator",
    "VectorChannel",
    "aggregate_correlators",
    "aggregate_time_series",
    "antisymmetric_singular_values",
    "bin_by_euclidean_time",
    "bootstrap_correlator_error",
    "build_all_dirac_plots",
    "build_all_einstein_plots",
    "build_all_gravity_plots",
    "build_all_higgs_plots",
    "build_all_quantum_gravity_time_series_plots",
    "build_antisymmetric_kernel",
    "build_centroid_vector_field",
    "build_gamma_matrices",
    "build_geodesic_distance_scatter",
    "build_higgs_action_summary",
    "build_mass_correlator_dashboard",
    "build_metric_eigenvalues_distribution",
    "build_metric_tensor_heatmap",
    "build_phase_space_antisymmetric_kernel",
    "build_ricci_scalar_distribution",
    "build_scalar_field_map",
    "build_volume_vs_curvature_scatter",
    "classify_walkers",
    "compute_all_channel_correlators",
    "compute_all_channels",
    "compute_all_correlators",
    "compute_all_operator_series",
    "compute_axial_vector_operators",
    "compute_centroid_displacement",
    "compute_channel_correlator",
    "compute_color_states_batch",
    "compute_companion_batch",
    "compute_correlator_fft",
    "compute_dirac_spectrum",
    "compute_effective_mass_torch",
    "compute_einstein_test",
    "compute_emergent_metric",
    "compute_fitness_gap_distribution",
    "compute_full_neighbor_matrix",
    "compute_geodesic_distances",
    "compute_glueball_operators",
    "compute_higgs_action",
    "compute_higgs_observables",
    "compute_higgs_vev_from_positions",
    "compute_isospin_channels",
    # Neighbor analysis
    "compute_neighbor_topology",
    "compute_neighbors_auto",
    "compute_nucleon_operators",
    "compute_pseudoscalar_operators",
    "compute_quantum_gravity_observables",
    "compute_quantum_gravity_time_evolution",
    "compute_radial_channels",
    "compute_recorded_neighbors_batch",
    "compute_scalar_operators",
    "compute_smoc_correlators_from_history",
    "compute_tensor_operators",
    "compute_time_sliced_voronoi",
    "compute_vector_operators",
    "compute_weighted_electroweak_ops_vectorized",
    "estimate_ell0",
    "extract_mass_aic",
    "extract_mass_linear",
    "get_channel_class",
    "pack_neighbor_lists",
    "pack_neighbors_from_edges",
    "predict_yukawa_mass_from_fitness",
    "run_smoc_pipeline",
    "save_mass_correlator_plots",
]
