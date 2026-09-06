// Optional history recorder, independent of the algorithm and environment
// storage.
#pragma once
#include <cstdint>
#include "control/checkpoint_io.hpp"
#include <unordered_map>
#include <vector>

namespace fg {
enum class RecordingMode { Off = 0, Pruned = 1, Full = 2 };
struct ExplorationNode {
  uint32_t id = 0, parent = 0, depth = 0, frames = 0, flags = 0, children = 0;
  float reward = 0, step_reward = 0, virtual_reward = 0;
};
class ExplorationTree {
 public:
  RecordingMode mode = RecordingMode::Off;
  size_t max_bytes = 128 * 1024 * 1024;
  std::vector<uint8_t> root_snapshot;
  uint64_t removed = 0;
  void reset(RecordingMode recording, size_t action_dim, size_t pose_dim);
  void reserve(size_t count);
  uint32_t append(uint32_t parent, uint32_t frames, const float* action,
                  const float* pose, float reward, float step_reward,
                  float fitness, uint32_t flags);
  size_t prune(const std::vector<uint32_t>& protected_ids);
  std::vector<uint32_t> branch(uint32_t leaf) const;
  const ExplorationNode& node(uint32_t id) const;
  const float* action(uint32_t id) const;
  void save_checkpoint(control::CheckpointWriter& out) const;
  void load_checkpoint(control::CheckpointReader& in);
  size_t size() const { return index_.size(); }
  size_t action_dim() const { return action_dim_; }
  size_t pose_dim() const { return pose_dim_; }
  // Compact export: metadata [id,parent,depth,frames,flags], values
  // [reward,step_reward,virtual_reward,pose...,action...].
  void export_data(std::vector<uint32_t>& metadata,
                   std::vector<float>& values) const;

 private:
  size_t action_dim_ = 0, pose_dim_ = 0;
  uint32_t next_id_ = 1;
  std::vector<ExplorationNode> nodes_;
  std::vector<float> actions_, poses_;
  std::vector<uint32_t> free_;
  std::unordered_map<uint32_t, uint32_t> index_;
};
}  // namespace fg
