#include "fractal/visit_grid.hpp"

#include <algorithm>
#include <utility>

namespace fg {

VisitGrid::VisitGrid(int32_t block_size, float erase_coef, float clip_max)
    : block_(block_size < 1 ? 1 : block_size), erase_coef_(erase_coef), clip_max_(clip_max) {}

void VisitGrid::reset() { tiles_.clear(); }

uint64_t VisitGrid::pack_id(int32_t plane, int32_t bx, int32_t by) {
  const uint64_t p = static_cast<uint64_t>(static_cast<uint32_t>(plane));
  return (p << 40) | ((static_cast<uint64_t>(by) & 0xFFFFFu) << 20) |
         (static_cast<uint64_t>(bx) & 0xFFFFFu);
}

void VisitGrid::unpack_id(uint64_t id, int32_t& plane, int32_t& bx, int32_t& by) {
  plane = static_cast<int32_t>(static_cast<uint32_t>(id >> 40));
  by = static_cast<int32_t>((id >> 20) & 0xFFFFFu);
  bx = static_cast<int32_t>(id & 0xFFFFFu);
}

uint64_t VisitGrid::tile_id(int32_t plane, int32_t tx, int32_t ty) {
  return pack_id(plane, tx, ty);
}

const VisitGrid::Tile* VisitGrid::find_tile(int32_t plane, int32_t tx, int32_t ty) const {
  const auto it = tiles_.find(tile_id(plane, tx, ty));
  return it == tiles_.end() ? nullptr : &it->second;
}

void VisitGrid::update(const std::vector<VisitKey>& keys) {
  // visits[idx] = visits[idx] + 1: the right-hand side is evaluated from the
  // pre-assignment values, so duplicate indices all write the same old + 1.
  // Increment each distinct cell exactly once.
  std::vector<std::pair<uint64_t, size_t>> cells;
  cells.reserve(keys.size());
  for (const VisitKey& key : keys) {
    const int32_t x = std::max(key.x, 0), y = std::max(key.y, 0);
    const uint64_t id = tile_id(key.plane, x / kTile, y / kTile);
    const size_t off = static_cast<size_t>(y % kTile) * kTile + static_cast<size_t>(x % kTile);
    cells.emplace_back(id, off);
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  for (const auto& c : cells) {
    Tile& t = tiles_[c.first];
    if (t.empty()) t.assign(static_cast<size_t>(kTile) * kTile, 0.0f);
    t[c.second] = t[c.second] + 1.0f;  // float32 + 1
  }

  // visits = clip(visits - erase_coef, 0, 1000) over the whole grid, in
  // float32 (a numpy float32 array minus a python float stays float32).
  const float e = erase_coef_;
  for (auto it = tiles_.begin(); it != tiles_.end();) {
    bool any = false;
    for (float& v : it->second) {
      float nv = v - e;
      if (nv < 0.0f) nv = 0.0f;
      if (nv > clip_max_) nv = clip_max_;
      v = nv;
      any = any || (nv != 0.0f);
    }
    if (!any) {
      it = tiles_.erase(it);  // all-zero tile == absent tile
    } else {
      ++it;
    }
  }
}

double VisitGrid::block_sum(int32_t plane, int32_t bx, int32_t by) const {
  const int32_t x0 = bx * block_, y0 = by * block_;
  const int32_t x1 = x0 + block_, y1 = y0 + block_;  // exclusive
  double sum = 0.0;
  for (int32_t ty = y0 / kTile; ty <= (y1 - 1) / kTile; ++ty) {
    for (int32_t tx = x0 / kTile; tx <= (x1 - 1) / kTile; ++tx) {
      const Tile* t = find_tile(plane, tx, ty);
      if (t == nullptr) continue;
      const int32_t cx0 = std::max(x0, tx * kTile), cx1 = std::min(x1, (tx + 1) * kTile);
      const int32_t cy0 = std::max(y0, ty * kTile), cy1 = std::min(y1, (ty + 1) * kTile);
      for (int32_t y = cy0; y < cy1; ++y) {
        const float* row = t->data() + static_cast<size_t>(y - ty * kTile) * kTile;
        for (int32_t x = cx0; x < cx1; ++x) sum += static_cast<double>(row[x - tx * kTile]);
      }
    }
  }
  return sum;
}

void VisitGrid::block_sums(const std::vector<VisitKey>& keys, std::vector<float>& out) const {
  out.resize(keys.size());
  // Many walkers share a block: compute each distinct block once per call.
  std::unordered_map<uint64_t, float> cache;
  for (size_t i = 0; i < keys.size(); ++i) {
    const int32_t bx = std::max(keys[i].x, 0) / block_;
    const int32_t by = std::max(keys[i].y, 0) / block_;
    const uint64_t id = pack_id(keys[i].plane, bx, by);
    const auto it = cache.find(id);
    if (it != cache.end()) {
      out[i] = it->second;
      continue;
    }
    const float s = static_cast<float>(block_sum(keys[i].plane, bx, by));
    cache.emplace(id, s);
    out[i] = s;
  }
}

void VisitGrid::export_blocks(std::vector<int32_t>& keys, std::vector<float>& sums) const {
  keys.clear();
  sums.clear();
  // Bin every nonzero pixel into its BxB block.
  std::unordered_map<uint64_t, double> blocks;
  for (const auto& kv : tiles_) {
    int32_t plane = 0, tx = 0, ty = 0;
    unpack_id(kv.first, plane, tx, ty);
    const Tile& t = kv.second;
    for (int32_t cy = 0; cy < kTile; ++cy) {
      for (int32_t cx = 0; cx < kTile; ++cx) {
        const float v = t[static_cast<size_t>(cy) * kTile + static_cast<size_t>(cx)];
        if (v == 0.0f) continue;
        const int32_t bx = (tx * kTile + cx) / block_;
        const int32_t by = (ty * kTile + cy) / block_;
        blocks[pack_id(plane, bx, by)] += static_cast<double>(v);
      }
    }
  }
  keys.reserve(blocks.size() * 3);
  sums.reserve(blocks.size());
  for (const auto& kv : blocks) {
    const float s = static_cast<float>(kv.second);
    if (s == 0.0f) continue;
    int32_t plane = 0, bx = 0, by = 0;
    unpack_id(kv.first, plane, bx, by);
    keys.push_back(plane);
    keys.push_back(bx);
    keys.push_back(by);
    sums.push_back(s);
  }
}

size_t VisitGrid::n_blocks() const {
  std::vector<int32_t> keys;
  std::vector<float> sums;
  export_blocks(keys, sums);
  return sums.size();
}

float VisitGrid::cell(const VisitKey& key) const {
  const int32_t x = std::max(key.x, 0), y = std::max(key.y, 0);
  const Tile* t = find_tile(key.plane, x / kTile, y / kTile);
  if (t == nullptr) return 0.0f;
  return (*t)[static_cast<size_t>(y % kTile) * kTile + static_cast<size_t>(x % kTile)];
}

size_t VisitGrid::nonzero_cells() const {
  size_t n = 0;
  for (const auto& kv : tiles_) {
    for (const float v : kv.second) n += (v != 0.0f) ? 1 : 0;
  }
  return n;
}

}  // namespace fg
