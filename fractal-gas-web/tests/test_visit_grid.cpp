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
