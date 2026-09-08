#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// One caller owns a runtime. Independent runtimes may execute concurrently.
// Negative status / null pointer means failure; fgc_error describes it.
// Borrowed raw pointers last until the next mutating call; fgc_borrow is
// leased.
const char* fgc_error();
// 1..64 execution slots, including the caller; independent of state format.
void* fgc_create(const char* json, int worlds, int threads);
void fgc_destroy(void* p);
int fgc_reset(void* p, uint32_t lo, uint32_t hi);
uint32_t fgc_hash_lo(void* p);
uint32_t fgc_hash_hi(void* p);
int fgc_info(void* p, int field);
float fgc_action_bound(void* p, int channel, int upper);
int fgc_action_body(void* p, int channel);
const char* fgc_action_name(void* p, int channel);
float* fgc_actions(void* p);
int32_t* fgc_frames(void* p);
float* fgc_states(void* p);
float* fgc_metrics(void* p);
double* fgc_profile(void* p);
int fgc_profile_reset(void* p);
int fgc_inspect(void* p);
float* fgc_inspection(void* p);
float* fgc_results(void* p);
int fgc_step(void* p);
int fgc_get_states(void* p, float* out, size_t bytes);
int fgc_set_states(void* p, const float* in, size_t bytes);
int fgc_broadcast(void* p, const uint8_t* data, size_t size);
int fgc_gather(void* p, const int32_t* indices, size_t count);
size_t fgc_snapshot_size(void* p);
int fgc_serialize(void* p, uint8_t* out, size_t bytes);
int fgc_deserialize(void* p, const uint8_t* in, size_t bytes);
void* fgc_borrow(void* p);
float* fgc_batch_data(void* p);
void fgc_release(void* p);
int fgc_observe(void* p, float* out, size_t count);
int fgc_plan_begin(void* p, const char* settings, uint32_t seed);
int fgc_plan_advance(void* p);
float* fgc_plan_action(void* p);
const char* fgc_plan_result(void* p);
int fgc_plan_best_leaf(void* p);
int fgc_plan_common_ancestor(void* p);
int fgc_wave_step(void* p);
float* fgc_wave_states(void* p);
int fgc_tree_export(void* p);
uint32_t* fgc_tree_meta(void* p);
float* fgc_tree_values(void* p);
uint8_t* fgc_tree_root(void* p);
size_t fgc_tree_root_size(void* p);
size_t fgc_checkpoint_size(void* p);
int fgc_checkpoint_write(void* p, uint8_t* out, size_t capacity);
int fgc_checkpoint_restore(void* p, const uint8_t* data, size_t size);
int fgc_replay_node(void* p, uint32_t id);
float fgc_raycast(void* p, float x, float y, float dx, float dy, float distance);
#ifdef __cplusplus
}
#endif
