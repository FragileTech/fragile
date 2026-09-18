//! Delaunay/Voronoi tessellation of the swarm and the geometry estimated on it.
//!
//! The stage is a pipeline of independently replaceable components: a
//! tessellator produces the neighbor graph, a metric estimator assigns a
//! metric to every walker, a volume element and edge weightings are derived
//! from it, curvature estimators read all of the above, and a reward
//! allocation turns curvature and volume into one action contribution per
//! walker. Geometry never consumes engine random addresses.
pub mod curvature;
pub mod degenerate;
pub mod delaunay2;
pub mod delaunay3;
pub mod domain;
pub mod forces;
pub mod frame;
pub mod graph;
pub mod hessian;
pub mod linalg;
pub mod mesh;
pub mod metric;
pub mod par;
pub mod pipeline;
pub mod presets;
pub mod proxies;
pub mod regge;
pub mod reward;
pub mod ricci;
pub mod sites;
pub mod stage;
pub mod volume;
pub mod voronoi;
pub mod weights;

pub use curvature::{
    CurvatureContext, CurvatureEstimator, CurvatureKind, CurvatureOutput, CurvatureSpec,
};
pub use degenerate::{DegeneracyPolicy, DuplicatePolicy, FailurePolicy, Tessellation};
pub use domain::TessellationDomain;
pub use frame::{EdgeLengths, GeometryFrame};
pub use graph::NeighborGraph;
pub use mesh::{SiteMesh, Tessellator, TessellatorKind};
pub use metric::{MetricEstimator, MetricField, MetricKind};
pub use par::Parallelism;
pub use pipeline::{GeometryPipelineConfig, TessellationGeometry};
pub use regge::ReggeLengths;
pub use reward::{GeometryReward, RewardAllocation, RewardAllocationKind, ZeroPotential};
pub use sites::{Projection, SiteSet};
pub use stage::{GeometrySchedule, GeometryStageConfig, GraphSnapshot};
pub use volume::{VolumeElement, VolumeKind};
pub use voronoi::{VoronoiCells, VoronoiCellsConfig};
pub use weights::{EdgeContext, EdgeWeighting, WeightMode, WeightSpec};
