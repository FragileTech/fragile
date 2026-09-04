// VisitGrid (src/visit_grid.hpp) against a dense float32 transcription of
// MontezumaTree.update_visits / aggregate_visits: the sparse grid must be
// bit-identical cell by cell, and block sums within 1 ulp.
#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"
#include "visit_grid.hpp"

using namespace fg;

namespace {

constexpr int kPlanes = 3, kH = 40, kW = 45;  // small grid, divisible by 5
constexpr int kBlock = 5;

struct DenseGrid {
  float erase;
  std::vector<float> v = std::vector<float>(kPlanes * kH * kW, 0.0f);
  float& at(int p, int y, int x) { return v[(p * kH + y) * kW + x]; }

  // update_visits: visits[idx] = visits[idx] + 1 (RHS from the old values,
  // duplicates write the same value), then clip(visits - e, 0, 1000).
  void update(const std::vector<VisitKey>& keys) {
    std::vector<float> old = v;
    for (const VisitKey& k : keys) at(k.plane, k.y, k.x) = old[(k.plane * kH + k.y) * kW + k.x] + 1.0f;
    for (float& c : v) {
      float nv = c - erase;
      if (nv < 0.0f) nv = 0.0f;
      if (nv > 1000.0f) nv = 1000.0f;
      c = nv;
    }
  }
  float block_sum(const VisitKey& k) {
    const int bx = k.x / kBlock, by = k.y / kBlock;
    float s = 0.0f;
    for (int dy = 0; dy < kBlock; ++dy)
      for (int dx = 0; dx < kBlock; ++dx) s += at(k.plane, by * kBlock + dy, bx * kBlock + dx);
    return s;
  }
};

}  // namespace

TEST_CASE(visit_grid_matches_dense_float32_reference) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> plane(0, kPlanes - 1);
  std::uniform_int_distribution<int> row(0, kH - 1);
  std::uniform_int_distribution<int> col(0, kW - 1);
  // Cluster hits so cells get hit repeatedly, duplicates occur inside a
  // batch, and some cells decay all the way to zero between hits.
  std::uniform_int_distribution<int> jitter(-3, 3);
  VisitGrid grid(kBlock, 0.05f);
  DenseGrid dense{0.05f};
  for (int step = 0; step < 400; ++step) {
    std::vector<VisitKey> keys;
    const int p = plane(gen);
    const int cy = row(gen), cx = col(gen);
    const int batch = 1 + (step % 7);
    for (int b = 0; b < batch; ++b) {
      int x = cx + jitter(gen), y = cy + jitter(gen);
      x = x < 0 ? 0 : (x >= kW ? kW - 1 : x);
      y = y < 0 ? 0 : (y >= kH ? kH - 1 : y);
      keys.push_back(VisitKey{p, x, y});
      if (b % 3 == 0) keys.push_back(VisitKey{p, x, y});  // duplicate in-batch
    }
    grid.update(keys);
    dense.update(keys);
    if (step % 25 == 0) {
      for (int pp = 0; pp < kPlanes; ++pp)
        for (int y = 0; y < kH; ++y)
          for (int x = 0; x < kW; ++x) {
            const float g = grid.cell(VisitKey{pp, x, y});
            const float d = dense.at(pp, y, x);
            CHECK(g == d);
          }
    }
    std::vector<float> sums;
    grid.block_sums(keys, sums);
    for (size_t i = 0; i < keys.size(); ++i) {
      const float expect = dense.block_sum(keys[i]);
      const float tol = std::fabs(expect) * 1.2e-7f;
      CHECK(std::fabs(sums[i] - expect) <= tol);
    }
  }
  size_t nonzero = 0;
  for (const float c : dense.v) nonzero += c != 0.0f ? 1 : 0;
  CHECK(grid.nonzero_cells() == nonzero);
  CHECK(nonzero > 0);
}

TEST_CASE(visit_grid_increments_once_per_distinct_cell) {
  VisitGrid grid(5, 0.0f);
  grid.update({VisitKey{0, 7, 3}, VisitKey{0, 7, 3}, VisitKey{0, 7, 3}, VisitKey{1, 7, 3}});
  CHECK(grid.cell(VisitKey{0, 7, 3}) == 1.0f);
  CHECK(grid.cell(VisitKey{1, 7, 3}) == 1.0f);
  CHECK(grid.cell(VisitKey{0, 8, 3}) == 0.0f);
  // Same 5x5 block (5..9, 0..4): the block sum counts both cells.
  grid.update({VisitKey{0, 9, 4}});
  std::vector<float> sums;
  grid.block_sums({VisitKey{0, 5, 0}, VisitKey{0, 10, 0}}, sums);
  CHECK(sums[0] == 2.0f);
  CHECK(sums[1] == 0.0f);
}

TEST_CASE(visit_grid_decay_erases_and_clips) {
  VisitGrid grid(5, 0.5f, 1000.0f);
  grid.update({VisitKey{2, 1, 1}});   // 1 - 0.5 = 0.5
  CHECK(grid.cell(VisitKey{2, 1, 1}) == 0.5f);
  grid.update({});                    // 0.5 - 0.5 = 0 -> erased
  CHECK(grid.cell(VisitKey{2, 1, 1}) == 0.0f);
  CHECK(grid.nonzero_cells() == 0);
  grid.update({VisitKey{2, 1, 1}});   // fresh again: 0 + 1 - 0.5
  CHECK(grid.cell(VisitKey{2, 1, 1}) == 0.5f);
  VisitGrid capped(5, 0.0f, 3.0f);
  for (int i = 0; i < 10; ++i) capped.update({VisitKey{0, 0, 0}});
  CHECK(capped.cell(VisitKey{0, 0, 0}) == 3.0f);
  capped.reset();
  CHECK(capped.cell(VisitKey{0, 0, 0}) == 0.0f);
}

TEST_CASE(visit_grid_export_blocks_matches_block_sums) {
  std::mt19937 gen(7);
  std::uniform_int_distribution<int> row(0, kH - 1);
  std::uniform_int_distribution<int> col(0, kW - 1);
  const int planes[3] = {0, 257, 8};  // incl. a Mario-style plane (1-1)
  VisitGrid grid(kBlock, 0.05f);
  DenseGrid dense{0.05f};
  // Use plane indices 0..2 in the dense reference, mapped to the real ids.
  for (int step = 0; step < 120; ++step) {
    std::vector<VisitKey> keys, dense_keys;
    for (int b = 0; b < 1 + (step % 5); ++b) {
      const int p = step % 3;
      const int x = col(gen), y = row(gen);
      keys.push_back(VisitKey{planes[p], x, y});
      dense_keys.push_back(VisitKey{p, x, y});
    }
    grid.update(keys);
    dense.update(dense_keys);
  }
  std::vector<int32_t> keys;
  std::vector<float> sums;
  grid.export_blocks(keys, sums);
  CHECK(keys.size() == 3 * sums.size());
  CHECK(sums.size() == grid.n_blocks());
  size_t dense_blocks = 0;
  for (int p = 0; p < 3; ++p)
    for (int by = 0; by < kH / kBlock; ++by)
      for (int bx = 0; bx < kW / kBlock; ++bx)
        if (dense.block_sum(VisitKey{p, bx * kBlock, by * kBlock}) != 0.0f) ++dense_blocks;
  CHECK(sums.size() == dense_blocks);
  for (size_t i = 0; i < sums.size(); ++i) {
    CHECK(sums[i] > 0.0f);
    const int32_t plane = keys[3 * i], bx = keys[3 * i + 1], by = keys[3 * i + 2];
    int p = -1;
    for (int k = 0; k < 3; ++k) if (planes[k] == plane) p = k;
    CHECK(p >= 0);
    std::vector<float> one;
    grid.block_sums({VisitKey{plane, bx * kBlock, by * kBlock}}, one);
    CHECK(one[0] == sums[i]);
    const float expect = dense.block_sum(VisitKey{p, bx * kBlock, by * kBlock});
    CHECK(std::fabs(sums[i] - expect) <= std::fabs(expect) * 1.2e-7f);
  }
}

TEST_CASE(visit_grid_pooling_size_is_live_and_storage_is_per_pixel) {
  VisitGrid grid(5, 0.0f);
  // Two hits in the first 5x5 block and one in the block to its right; a
  // hit far away on the same plane; one on another plane.
  grid.update({VisitKey{0, 1, 1}, VisitKey{0, 3, 4}, VisitKey{0, 7, 2},
               VisitKey{0, 40, 40}, VisitKey{3, 1, 1}});
  const size_t cells = grid.nonzero_cells();
  CHECK(cells == 5);
  std::vector<float> s;
  grid.block_sums({VisitKey{0, 0, 0}, VisitKey{0, 6, 0}}, s);
  CHECK(s[0] == 2.0f && s[1] == 1.0f);

  // B = 1: the block sum IS the pixel value.
  grid.set_block_size(1);
  grid.block_sums({VisitKey{0, 1, 1}, VisitKey{0, 2, 1}, VisitKey{0, 7, 2}}, s);
  CHECK(s[0] == 1.0f && s[1] == 0.0f && s[2] == 1.0f);
  // B = 10: the four 5-blocks merge; (40, 40) is block (4, 4).
  grid.set_block_size(10);
  grid.block_sums({VisitKey{0, 0, 0}, VisitKey{0, 9, 9}, VisitKey{0, 40, 40}, VisitKey{3, 0, 0}}, s);
  CHECK(s[0] == 3.0f && s[1] == 3.0f && s[2] == 1.0f && s[3] == 1.0f);
  // A window spanning several storage tiles (kTile = 32) sums across them.
  grid.set_block_size(64);
  grid.block_sums({VisitKey{0, 0, 0}}, s);
  CHECK(s[0] == 4.0f);
  // Changing the pooling never touched the stored counters.
  CHECK(grid.nonzero_cells() == cells);
  CHECK(grid.cell(VisitKey{0, 1, 1}) == 1.0f);

  // Export re-bins at the current size.
  std::vector<int32_t> keys;
  std::vector<float> sums;
  grid.set_block_size(10);
  grid.export_blocks(keys, sums);
  CHECK(sums.size() == 3);  // plane 0 blocks (0,0) and (4,4); plane 3 block (0,0)
  CHECK(grid.n_blocks() == 3);
  grid.set_block_size(1);
  grid.export_blocks(keys, sums);
  CHECK(sums.size() == 5);
  // Partial edge blocks: B = 7 on a 160-wide room puts x = 158 in block 22.
  grid.set_block_size(7);
  grid.update({VisitKey{1, 158, 3}});
  grid.export_blocks(keys, sums);
  bool found = false;
  for (size_t i = 0; i < sums.size(); ++i) {
    if (keys[3 * i] == 1) found = (keys[3 * i + 1] == 22 && keys[3 * i + 2] == 0 && sums[i] == 1.0f);
  }
  CHECK(found);
}
