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

from fragile.fractalai.qft.neighbor_analysis import (
    # Neighbor computation functions
    compute_neighbor_topology,
    compute_full_neighbor_matrix,
    # Individual neighbor methods
    compute_companion_batch,
    compute_recorded_neighbors_batch,
    compute_neighbors_auto,
)

from fragile.fractalai.qft.aggregation import (
    # Data structures
    AggregatedTimeSeries,
    OperatorTimeSeries,
    # Main aggregation function
    aggregate_time_series,
    compute_all_operator_series,
    # Time series binning
    bin_by_euclidean_time,
    # Component functions
    compute_color_states_batch,
    estimate_ell0,
    # Gamma matrices
    build_gamma_matrices,
    # Channel-specific operator functions
    compute_scalar_operators,
    compute_pseudoscalar_operators,
    compute_vector_operators,
    compute_axial_vector_operators,
    compute_tensor_operators,
    compute_nucleon_operators,
    compute_glueball_operators,
)

from fragile.fractalai.qft.correlator_channels import (
    # Config and results
    ChannelConfig,
    CorrelatorConfig as CorrelatorAnalysisConfig,  # Avoid conflict with SMoC pipeline
    ChannelCorrelatorResult,
    # AIC extractor
    ConvolutionalAICExtractor,
    # FFT correlator
    compute_correlator_fft,
    compute_effective_mass_torch,
    bootstrap_correlator_error,
    # Pure function API (new)
    extract_mass_aic,
    extract_mass_linear,
    compute_channel_correlator,
    compute_all_correlators,
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
from fragile.fractalai.qft.electroweak_observables import (
    PackedNeighbors,
    pack_neighbor_lists,
    pack_neighbors_from_edges,
    compute_weighted_electroweak_ops_vectorized,
    compute_higgs_vev_from_positions,
    compute_fitness_gap_distribution,
    predict_yukawa_mass_from_fitness,
    build_phase_space_antisymmetric_kernel,
    antisymmetric_singular_values,
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
from fragile.fractalai.qft.dirac_spectrum import (
    DiracSpectrumConfig,
    DiracSpectrumResult,
    SectorSpectrum,
    compute_dirac_spectrum,
    build_antisymmetric_kernel,
    classify_walkers,
)
from fragile.fractalai.qft.dirac_spectrum_plotting import (
    build_all_dirac_plots,
)
from fragile.fractalai.qft.isospin_channels import (
    ISOSPIN_CHANNEL_SPLITTINGS,
    ISOSPIN_MASS_RATIOS,
    IsospinChannelResult,
    compute_isospin_channels,
)
from fragile.fractalai.qft.einstein_equations import (
    EinsteinConfig,
    EinsteinTestResult,
    compute_einstein_test,
)
from fragile.fractalai.qft.einstein_equations_plotting import (
    build_all_einstein_plots,
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
    # Neighbor analysis
    "compute_neighbor_topology",
    "compute_full_neighbor_matrix",
    "compute_companion_batch",
    "compute_recorded_neighbors_batch",
    "compute_neighbors_auto",
    # Time series aggregation (new)
    "AggregatedTimeSeries",
    "OperatorTimeSeries",
    "aggregate_time_series",
    "compute_all_operator_series",
    "bin_by_euclidean_time",
    "compute_color_states_batch",
    "estimate_ell0",
    "build_gamma_matrices",
    "compute_scalar_operators",
    "compute_pseudoscalar_operators",
    "compute_vector_operators",
    "compute_axial_vector_operators",
    "compute_tensor_operators",
    "compute_nucleon_operators",
    "compute_glueball_operators",
    # Correlator channels (new vectorized)
    "ChannelConfig",
    "CorrelatorAnalysisConfig",
    "ChannelCorrelatorResult",
    "ConvolutionalAICExtractor",
    "compute_correlator_fft",
    "compute_effective_mass_torch",
    "bootstrap_correlator_error",
    "extract_mass_aic",
    "extract_mass_linear",
    "compute_channel_correlator",
    "compute_all_correlators",
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
    # Vectorized electroweak observables
    "PackedNeighbors",
    "pack_neighbor_lists",
    "pack_neighbors_from_edges",
    "compute_weighted_electroweak_ops_vectorized",
    "compute_higgs_vev_from_positions",
    "compute_fitness_gap_distribution",
    "predict_yukawa_mass_from_fitness",
    "build_phase_space_antisymmetric_kernel",
    "antisymmetric_singular_values",
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
    # Dirac spectrum analysis
    "DiracSpectrumConfig",
    "DiracSpectrumResult",
    "SectorSpectrum",
    "compute_dirac_spectrum",
    "build_antisymmetric_kernel",
    "classify_walkers",
    "build_all_dirac_plots",
    # Isospin channel correlators
    "ISOSPIN_CHANNEL_SPLITTINGS",
    "ISOSPIN_MASS_RATIOS",
    "IsospinChannelResult",
    "compute_isospin_channels",
    # Einstein equation verification
    "EinsteinConfig",
    "EinsteinTestResult",
    "compute_einstein_test",
    "build_all_einstein_plots",
]
