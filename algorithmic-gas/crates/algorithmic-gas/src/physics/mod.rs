//! Finite-step algorithmic physics. Diagnostics never consume engine random addresses.
//!
//! A fitness Hessian, a diffusion metric, and a parameter Fisher metric are distinct
//! objects. These modules compute them without assuming a continuum or Einstein law.
pub mod balances;
pub mod closure;
pub mod evolution;
pub mod fields;
pub mod fitness;
pub mod geometry;
pub mod jet;
pub mod partvi;
pub mod path_action;
pub mod qft;
pub mod thermodynamics;
