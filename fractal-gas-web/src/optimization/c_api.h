#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All strings/pointers are borrowed until the next mutation of that handle.
// A snapshot is float64: 12 header words, then n rows of [x[d],v[d],U,
// fitness,alive,distance_companion,clone_companion,parent,cloned,leaf].
// Headers: version,n,d,has_velocity,iteration,evaluations,alive,cloned,
// current_best,best_so_far,mean,best_index. Non-finite U denotes invalidity.
// Best metrics follow config.objective; U remains the raw function value.
// FMC/Wave Jump append the committed position after config.walkers search rows.
// On a new search, search-row parent indices refer to that committed row.
const char* fgo_catalog(void);
// Actual numerical storage precision, independent of the double snapshot format.
// Static JSON; remains valid for the lifetime of the library.
const char* fgo_precision(void);
const char* fgo_error(void);
uint32_t fgo_create(const char* config);
int fgo_destroy(uint32_t handle);
const char* fgo_config(uint32_t handle);
// JSON status: finished, stop_reason (CMA), next_evaluations, next_population,
// budget_exhausted, and CMA generation/population/restarts/sigma. Read-only.
// The status string is borrowed until the next fgo_status call (any handle).
const char* fgo_status(uint32_t handle);
int fgo_step(uint32_t handle);
// Opt-in observational diagnostics; does not advance or reconfigure the optimizer.
int fgo_geometry_diagnostics(uint32_t handle, int enabled);
const char* fgo_export_basins(uint32_t handle);
int fgo_import_basins(uint32_t handle, const char* data);
// Updates are validated as a batch. Immutable fields are rejected; planner movement
// changes wait for the next search, while the budget takes effect immediately.
// Status contains effective_settings, pending_settings, settings_revision and boundary events.
// Preview validates and returns the merged requested config without changing the session.
int fgo_update_settings(uint32_t handle, const char* patch);
const char* fgo_preview_settings(uint32_t handle, const char* patch);
int fgo_set_population(uint32_t handle, int walkers, const char* removal_policy);
const double* fgo_snapshot(uint32_t handle);
int fgo_snapshot_size(uint32_t handle);
// Deterministic display sampling; noise returns its expectation without RNG
// draws.
int fgo_sample(uint32_t handle, const float* positions, int count,
               double* values);
// Double-coordinate display/fixture sampling; does not count evaluations.
int fgo_sample64(uint32_t handle, const double* positions, int count, double* values);
#ifdef __cplusplus
}
#endif
