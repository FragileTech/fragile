#include "exploration_tree.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace fg {
void ExplorationTree::reset(RecordingMode recording, size_t action_dim,
                            size_t pose_dim) {
  mode = recording;
  action_dim_ = action_dim;
  pose_dim_ = pose_dim;
  next_id_ = 1;
  removed = 0;
  nodes_.clear();
  actions_.clear();
  poses_.clear();
  free_.clear();
  index_.clear();
  root_snapshot.clear();
}
void ExplorationTree::reserve(size_t count) {
  if (mode == RecordingMode::Off) return;
  size_t needed =
      nodes_.size() + (count > free_.size() ? count - free_.size() : 0);
  size_t per_node =
      sizeof(ExplorationNode) + 4 * (action_dim_ + pose_dim_) + 48;
  if (needed > max_bytes / per_node ||
      count > std::numeric_limits<uint32_t>::max() - next_id_)
    throw std::runtime_error(
        "Exploration recording memory limit reached; export or reset the "
        "recording");
  // Geometric growth is checked against the memory budget as well.
  size_t capacity =
      std::min(max_bytes / per_node,
               std::max(needed, std::max(size_t(64), nodes_.capacity() * 2)));
  if (needed > nodes_.capacity()) {
    nodes_.reserve(capacity);
    actions_.reserve(capacity * action_dim_);
    poses_.reserve(capacity * pose_dim_);
    index_.reserve(capacity);
  }
}
uint32_t ExplorationTree::append(uint32_t parent, uint32_t frames,
                                 const float* a, const float* p, float reward,
                                 float step_reward, float fitness,
                                 uint32_t flags) {
  if (mode == RecordingMode::Off) return 0;
  uint32_t depth = 0;
  if (parent) {
    auto it = index_.find(parent);
    if (it == index_.end())
      throw std::logic_error("Missing exploration parent");
    depth = nodes_[it->second].depth + 1;
  }
  reserve(1);
  uint32_t slot;
  if (free_.empty()) {
    slot = uint32_t(nodes_.size());
    nodes_.emplace_back();
    actions_.resize(nodes_.size() * action_dim_);
    poses_.resize(nodes_.size() * pose_dim_);
  } else {
    slot = free_.back();
    free_.pop_back();
  }
  uint32_t id = next_id_++;
  nodes_[slot] = {id, parent, depth,       frames, flags,
                  0,  reward, step_reward, fitness};
  index_.emplace(id, slot);
  if (parent) ++nodes_[index_.at(parent)].children;
  if (action_dim_)
    std::copy_n(a, action_dim_, actions_.data() + slot * action_dim_);
  if (pose_dim_) std::copy_n(p, pose_dim_, poses_.data() + slot * pose_dim_);
  return id;
}
size_t ExplorationTree::prune(const std::vector<uint32_t>& protected_ids) {
  if (mode != RecordingMode::Pruned) return 0;
  std::unordered_set<uint32_t> pins(protected_ids.begin(), protected_ids.end());
  std::vector<uint32_t> leaves;
  for (const auto& n : nodes_)
    if (n.id && n.parent && !n.children && !pins.count(n.id))
      leaves.push_back(n.id);
  size_t count = 0;
  for (uint32_t id : leaves)
    while (id) {
      auto it = index_.find(id);
      if (it == index_.end()) break;
      auto& n = nodes_[it->second];
      if (!n.parent || n.children || pins.count(id) || pins.count(n.parent))
        break;
      uint32_t parent = n.parent;
      free_.push_back(it->second);
      n.id = 0;
      index_.erase(it);
      ++count;
      auto& pn = nodes_[index_.at(parent)];
      --pn.children;
      id = parent;
    }
  removed += count;
  return count;
}
const ExplorationNode& ExplorationTree::node(uint32_t id) const {
  return nodes_.at(index_.at(id));
}
const float* ExplorationTree::action(uint32_t id) const {
  return actions_.data() + index_.at(id) * action_dim_;
}
std::vector<uint32_t> ExplorationTree::branch(uint32_t leaf) const {
  std::vector<uint32_t> result;
  while (leaf) {
    result.push_back(leaf);
    leaf = node(leaf).parent;
  }
  std::reverse(result.begin(), result.end());
  return result;
}
void ExplorationTree::export_data(std::vector<uint32_t>& meta,
                                  std::vector<float>& values) const {
  meta.clear();
  values.clear();
  meta.reserve(size() * 5);
  values.reserve(size() * (3 + pose_dim_ + action_dim_));
  for (size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    if (!n.id) continue;
    meta.insert(meta.end(), {n.id, n.parent, n.depth, n.frames, n.flags});
    values.insert(values.end(), {n.reward, n.step_reward, n.virtual_reward});
    values.insert(values.end(), poses_.begin() + i * pose_dim_,
                  poses_.begin() + (i + 1) * pose_dim_);
    values.insert(values.end(), actions_.begin() + i * action_dim_,
                  actions_.begin() + (i + 1) * action_dim_);
  }
}
void ExplorationTree::save_checkpoint(control::CheckpointWriter& out) const {
  out.scalar(uint32_t(mode));
  out.scalar(uint64_t(max_bytes));
  out.scalar(uint64_t(action_dim_));
  out.scalar(uint64_t(pose_dim_));
  out.scalar(next_id_);
  out.scalar(removed);
  out.vector(root_snapshot);
  out.vector(nodes_);
  out.vector(actions_);
  out.vector(poses_);
  out.vector(free_);
}
void ExplorationTree::load_checkpoint(control::CheckpointReader& in) {
  const auto recording = in.scalar<uint32_t>();
  const auto budget = in.scalar<uint64_t>();
  const auto ad = in.scalar<uint64_t>(), pd = in.scalar<uint64_t>();
  if (recording > 2 || ad > 8192 || pd > 8192 || budget > 128 * 1024 * 1024)
    throw std::invalid_argument("Invalid checkpoint tree dimensions");
  reset(static_cast<RecordingMode>(recording), ad, pd);
  max_bytes = budget;
  next_id_ = in.scalar<uint32_t>();
  removed = in.scalar<uint64_t>();
  root_snapshot = in.vector<uint8_t>();
  nodes_ = in.vector<ExplorationNode>();
  actions_ = in.vector<float>();
  poses_ = in.vector<float>();
  free_ = in.vector<uint32_t>();
  if (actions_.size() != nodes_.size() * ad ||
      poses_.size() != nodes_.size() * pd)
    throw std::invalid_argument("Invalid checkpoint tree storage");
  std::vector<uint32_t> children(nodes_.size()), slots(nodes_.size());
  for (uint32_t i = 0; i < nodes_.size(); ++i)
    if (nodes_[i].id) {
      if (nodes_[i].id >= next_id_ || !index_.emplace(nodes_[i].id, i).second)
        throw std::invalid_argument("Duplicate checkpoint node");
    }
  for (auto slot : free_)
    if (slot >= nodes_.size() || nodes_[slot].id || slots[slot]++)
      throw std::invalid_argument("Invalid free tree slot");
  for (size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    if (!n.id) {
      if (!slots[i]) throw std::invalid_argument("Missing free tree slot");
      continue;
    }
    if (n.parent) {
      auto p = index_.find(n.parent);
      if (p == index_.end() || n.parent >= n.id ||
          nodes_[p->second].depth + 1 != n.depth)
        throw std::invalid_argument("Invalid checkpoint ancestry");
      ++children[p->second];
    }
  }
  for (size_t i = 0; i < nodes_.size(); ++i)
    if (nodes_[i].id && children[i] != nodes_[i].children)
      throw std::invalid_argument("Invalid checkpoint child count");
}
}  // namespace fg
