#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All strings/pointers are borrowed until the next mutation of that handle.
// A snapshot is float64: 12 header words, then n rows of [x[d],v[d],U,
// fitness,alive,distance_companion,clone_companion,parent,cloned,leaf].
// Headers: version,n,d,has_velocity,iteration,evaluations,alive,cloned,
// current_min,best_so_far,mean,best_index. Non-finite U denotes invalidity.
const char* fgo_catalog(void);
const char* fgo_error(void);
uint32_t fgo_create(const char* config);
int fgo_destroy(uint32_t handle);
const char* fgo_config(uint32_t handle);
int fgo_step(uint32_t handle);
const double* fgo_snapshot(uint32_t handle);
int fgo_snapshot_size(uint32_t handle);
// Deterministic display sampling; noise returns its expectation without RNG
// draws.
int fgo_sample(uint32_t handle, const float* positions, int count,
               double* values);
#ifdef __cplusplus
}
#endif
