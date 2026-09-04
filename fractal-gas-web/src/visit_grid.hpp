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
//   aggregate_visits(block_size=B, upsample=True)[cell] = sum of the BxB
//     block (aligned to multiples of B) containing the cell
//
// Storage is PER PIXEL and sparse: a hash map of fixed kTile x kTile tiles of
// float32 counters, so the arithmetic is exactly the dense numpy program:
// cells are independent, an unvisited cell is identically 0
// (clip(0 - e, 0, 1000) == 0), so absent tiles are the zero tiles, and the
// decay is applied EAGERLY to every stored cell in float32 (a lazy "v - k*e"
// would not reproduce k successive float32 roundings of an erase_coef like
// 0.05 that is not representable). A tile whose cells all reach 0 is erased
// and re-created as zeros on the next hit, which is bit-identical to the
// dense version.
//
// The pooling size B is NOT part of the storage: it is applied when the grid
// is read (block_sums for the reward, export_blocks for the heatmap) and can
// change live mid-run without touching the per-pixel history. The reference
// fixed B = 5 and required the grid dimensions to be divisible by it; here
// any B >= 1 works and edge blocks may be partial. Block sums accumulate the
// <= B*B cell values in double and round once (numpy's reduction order is
// unspecified, so bit-equality is only defined within 1 ulp anyway).
#ifndef FRACTAL_GAS_VISIT_GRID_HPP
#define FRACTAL_GAS_VISIT_GRID_HPP

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
  static constexpr int32_t kTile = 32;  // storage tile (pixels per side)

  explicit VisitGrid(int32_t block_size = 5, float erase_coef = 0.05f,
                     float clip_max = 1000.0f);

  void reset();
  void set_erase_coef(float v) { erase_coef_ = v; }
  float erase_coef() const { return erase_coef_; }
  /// Pooling window (aggregate_visits block_size). Live-tunable: only reads
  /// depend on it, the stored per-pixel counters never change.
  void set_block_size(int32_t b) { block_ = b < 1 ? 1 : b; }
  int32_t block_size() const { return block_; }

  /// update_visits(): +1 once per distinct cell in `keys`, then the global
  /// decay + clip over every stored cell.
  void update(const std::vector<VisitKey>& keys);

  /// aggregate_visits(block_size, upsample=True)[key] for each key: the sum
  /// of the BxB block containing the cell (0 for untouched blocks).
  void block_sums(const std::vector<VisitKey>& keys,
                  std::vector<float>& out) const;

  /// Every nonzero BxB block for display: keys = [plane, bx, by] triples
  /// (block coordinates: cell / B), sums = the block sum (the
  /// aggregate_visits value shown by the old demo's heatmap). Order
  /// unspecified.
  void export_blocks(std::vector<int32_t>& keys, std::vector<float>& sums) const;
  /// Number of nonzero BxB blocks at the current pooling size.
  size_t n_blocks() const;

  /// Current value of one cell (0 when absent). For tests.
  float cell(const VisitKey& key) const;
  /// Number of stored cells with a non-zero value. For tests.
  size_t nonzero_cells() const;
  /// Number of stored tiles. For tests.
  size_t n_tiles() const { return tiles_.size(); }

 private:
  using Tile = std::vector<float>;  // kTile * kTile cells, row-major (y, x)

  static uint64_t tile_id(int32_t plane, int32_t tx, int32_t ty);
  static uint64_t pack_id(int32_t plane, int32_t bx, int32_t by);
  static void unpack_id(uint64_t id, int32_t& plane, int32_t& bx, int32_t& by);
  const Tile* find_tile(int32_t plane, int32_t tx, int32_t ty) const;
  /// Sum of the block with block coordinates (bx, by) on `plane`.
  double block_sum(int32_t plane, int32_t bx, int32_t by) const;

  int32_t block_;
  float erase_coef_;
  float clip_max_;
  std::unordered_map<uint64_t, Tile> tiles_;
};

}  // namespace fg

#endif  // FRACTAL_GAS_VISIT_GRID_HPP
