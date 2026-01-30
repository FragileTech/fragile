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
]
