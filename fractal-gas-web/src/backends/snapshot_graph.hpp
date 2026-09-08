#pragma once
#include "backends/snapshot.hpp"
namespace fg {
struct SnapshotGraphBackend : SnapshotBackend {
  using SnapshotBackend::SnapshotBackend;
  using Storage = std::vector<std::vector<char>>;
  using Info = WalkerInfo;
  using Action = int32_t;
  using Batch = WalkerState;
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
  bool valid_slot(const Storage& s, size_t i) const { return !s[i].empty(); }
  void commit(Storage& from, size_t i, Storage& to, size_t j) { std::swap(to[j], from[i]); }
};
}  // namespace fg
