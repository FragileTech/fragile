#pragma once
#include "backends/snapshot.hpp"
namespace fg {
struct SnapshotGraphBackend : SnapshotBackend {
  using SnapshotBackend::SnapshotBackend;
  using Storage = std::vector<std::vector<char>>;
  using Info = WalkerInfo;
  using Action = int32_t;
  using Batch = WalkerState;
  using StoredState = std::vector<char>;
  std::vector<float> initial_observation;
  int n_actions() const { return env.n_actions(); }
  int action_dim() const { return 1; }
  int observation_dim() const { return int(initial_observation.size()); }
  ThreadPool* pool() { return env.worker_pool(); }
  bool has_visit_key() const { return env.has_visit_key(); }
  bool has_infos() const { return env.has_walker_info(); }
  VisitKey visit_key(const Info& w) const { return {w.visit_plane, w.visit_x, w.visit_y}; }
  std::vector<char> reset_root() {
    std::vector<char> root;
    env.reset(root, initial_observation);
    return root;
  }
  void grow(Storage& s, int n) { s.resize(n); }
  void broadcast(const std::vector<char>& root, Storage& s, int n) { s.assign(n, root); }
  bool best_candidate(const Storage& s, size_t i) const { return env.best_candidate(s[i]); }
  bool valid_slot(const Storage& s, size_t i) const { return !s[i].empty(); }
  void commit(Storage& from, size_t i, Storage& to, size_t j) { std::swap(to[j], from[i]); }
  StoredState save_state(const Storage& states, size_t i) const { return states[i]; }
  void compact(Storage& states, const std::vector<int32_t>& keep) {
    Storage next;
    next.reserve(keep.size());
    for (int32_t i : keep) next.push_back(std::move(states[static_cast<size_t>(i)]));
    states = std::move(next);
  }
};
}  // namespace fg
