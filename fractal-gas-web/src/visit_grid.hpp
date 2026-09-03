// Visit-count grid of the tree algorithm — translation of
// MontezumaTree.update_visits / calculate_other_reward + aggregate_visits
// (cb9f3296 src/fragile/videogames.py, src/fragile/core.py):
//
//   visits: float32 [planes, H, W], zero-initialized
//   update_visits(cells):
//     visits[cells] = visits[cells] + 1        # fancy-index assignment: a cell
//                                              # hit by several walkers in the
//                                              # same batch gets +1 ONCE
//     visits = clip(visits - erase_coef, 0, 1000)   # whole grid, float32
//   aggregate_visits(block_size=5, upsample=True)[cell] = sum of the 5x5
//     block containing the cell
//
// Storage is sparse (a hash map of 5x5 blocks) but the arithmetic is exactly
// the dense numpy program: cells are independent, an unvisited cell is
// identically 0 (clip(0 - e, 0, 1000) == 0), so absent blocks are the zero
// blocks, and the decay is applied EAGERLY to every stored cell in float32
// (a lazy "v - k*e" would not reproduce k successive float32 roundings of an
// erase_coef like 0.05 that is not representable). A block whose 25 cells
// all reach 0 is erased and re-created as zeros on the next hit, which is
// bit-identical to the dense version. Block sums accumulate the <= 25 cell
// values in double and round once (numpy's reduction order is unspecified,
// so bit-equality is only defined within 1 ulp anyway).
#ifndef FRACTAL_GAS_VISIT_GRID_HPP
#define FRACTAL_GAS_VISIT_GRID_HPP

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace fg {

struct VisitKey {
  int32_t plane = 0;  // room / level id
  int32_t x = 0;      // cell column (>= 0)
  int32_t y = 0;      // cell row (>= 0)
};

class VisitGrid {
 public:
  explicit VisitGrid(int32_t block_size = 5, float erase_coef = 0.05f,
                     float clip_max = 1000.0f);

  void reset();
  void set_erase_coef(float v) { erase_coef_ = v; }
  float erase_coef() const { return erase_coef_; }
  int32_t block_size() const { return block_; }

  /// update_visits(): +1 once per distinct cell in `keys`, then the global
  /// decay + clip over every stored cell.
  void update(const std::vector<VisitKey>& keys);

  /// aggregate_visits(block_size, upsample=True)[key] for each key: the sum
  /// of the block containing the cell (0 for untouched blocks).
  void block_sums(const std::vector<VisitKey>& keys,
                  std::vector<float>& out) const;

  /// Current value of one cell (0 when absent). For tests.
  float cell(const VisitKey& key) const;
  /// Number of stored cells with a non-zero value. For tests.
  size_t nonzero_cells() const;

 private:
  using Block = std::vector<float>;  // block_ * block_ cells, row-major (y, x)

  static uint64_t block_id(const VisitKey& key, int32_t block);
  size_t cell_offset(const VisitKey& key) const;

  int32_t block_;
  float erase_coef_;
  float clip_max_;
  std::unordered_map<uint64_t, Block> blocks_;
};

}  // namespace fg

#endif  // FRACTAL_GAS_VISIT_GRID_HPP
