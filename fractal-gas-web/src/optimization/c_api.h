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
const char* fgo_error(void);
uint32_t fgo_create(const char* config);
int fgo_destroy(uint32_t handle);
const char* fgo_config(uint32_t handle);
// JSON status: finished, stop_reason (CMA), next_evaluations, next_population,
// budget_exhausted, and CMA generation/population/restarts/sigma. Read-only.
// The status string is borrowed until the next fgo_status call (any handle).
const char* fgo_status(uint32_t handle);
int fgo_step(uint32_t handle);
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
