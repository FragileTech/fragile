#include "visit_grid.hpp"

#include <algorithm>

namespace fg {

VisitGrid::VisitGrid(int32_t block_size, float erase_coef, float clip_max)
    : block_(block_size < 1 ? 1 : block_size),
      erase_coef_(erase_coef),
      clip_max_(clip_max) {}

void VisitGrid::reset() { blocks_.clear(); }

uint64_t VisitGrid::block_id(const VisitKey& key, int32_t block) {
  const uint64_t plane = static_cast<uint64_t>(static_cast<uint32_t>(key.plane));
  const uint64_t bx = static_cast<uint64_t>(std::max(key.x, 0) / block);
  const uint64_t by = static_cast<uint64_t>(std::max(key.y, 0) / block);
  return (plane << 40) | ((by & 0xFFFFFu) << 20) | (bx & 0xFFFFFu);
}

size_t VisitGrid::cell_offset(const VisitKey& key) const {
  const int32_t cx = std::max(key.x, 0) % block_;
  const int32_t cy = std::max(key.y, 0) % block_;
  return static_cast<size_t>(cy) * static_cast<size_t>(block_) +
         static_cast<size_t>(cx);
}

void VisitGrid::update(const std::vector<VisitKey>& keys) {
  // visits[idx] = visits[idx] + 1: the right-hand side is evaluated from the
  // pre-assignment values, so duplicate indices all write the same old + 1.
  // Increment each distinct cell exactly once.
  std::vector<std::pair<uint64_t, size_t>> cells;
  cells.reserve(keys.size());
  for (const VisitKey& key : keys) {
    cells.emplace_back(block_id(key, block_), cell_offset(key));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  const size_t cells_per_block =
      static_cast<size_t>(block_) * static_cast<size_t>(block_);
  for (const auto& c : cells) {
    Block& b = blocks_[c.first];
    if (b.empty()) b.assign(cells_per_block, 0.0f);
    b[c.second] = b[c.second] + 1.0f;  // float32 + 1
  }

  // visits = clip(visits - erase_coef, 0, 1000) over the whole grid, in
  // float32 (a numpy float32 array minus a python float stays float32).
  const float e = erase_coef_;
  for (auto it = blocks_.begin(); it != blocks_.end();) {
    bool any = false;
    for (float& v : it->second) {
      float nv = v - e;
      if (nv < 0.0f) nv = 0.0f;
      if (nv > clip_max_) nv = clip_max_;
      v = nv;
      any = any || (nv != 0.0f);
    }
    if (!any) {
      it = blocks_.erase(it);  // all-zero block == absent block
    } else {
      ++it;
    }
  }
}

void VisitGrid::block_sums(const std::vector<VisitKey>& keys,
                           std::vector<float>& out) const {
  out.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto it = blocks_.find(block_id(keys[i], block_));
    if (it == blocks_.end()) {
      out[i] = 0.0f;
      continue;
    }
    double sum = 0.0;
    for (const float v : it->second) sum += static_cast<double>(v);
    out[i] = static_cast<float>(sum);
  }
}

float VisitGrid::cell(const VisitKey& key) const {
  const auto it = blocks_.find(block_id(key, block_));
  if (it == blocks_.end()) return 0.0f;
  return it->second[cell_offset(key)];
}

size_t VisitGrid::nonzero_cells() const {
  size_t n = 0;
  for (const auto& kv : blocks_) {
    for (const float v : kv.second) n += (v != 0.0f) ? 1 : 0;
  }
  return n;
}

}  // namespace fg
