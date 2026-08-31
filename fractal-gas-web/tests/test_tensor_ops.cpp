#include <algorithm>

#include "rng.hpp"
#include "tensor_ops.hpp"
#include "test_framework.hpp"
#include "fixtures/fixtures_generated.hpp"

using namespace fg;

TEST_CASE(asymmetric_rescale_matches_python) {
  for (const auto& c : fixtures::kAsymCases) {
    const std::vector<float> out = asymmetric_rescale(*c.in);
    CHECK(out.size() == c.out->size());
    for (size_t i = 0; i < out.size(); ++i) {
      CHECK_CLOSE(out[i], (*c.out)[i], 1e-5);
    }
  }
}

TEST_CASE(l2_norm_matches_python) {
  const std::vector<float> dist = l2_norm_companions(
      fixtures::kL2Obs, fixtures::kL2Companions, fixtures::kL2N, fixtures::kL2Dim);
  CHECK(dist.size() == fixtures::kL2Expected.size());
  for (size_t i = 0; i < dist.size(); ++i) {
    CHECK_CLOSE(dist[i], fixtures::kL2Expected[i], 1e-5);
  }
}

TEST_CASE(random_alive_compas_all_alive_is_permutation) {
  Mt19937Rng rng(42);
  const std::vector<uint8_t> alive(16, 1);
  const std::vector<int32_t> compas = random_alive_compas(alive, rng);
  CHECK(compas.size() == 16);
  std::vector<int32_t> sorted = compas;
  std::sort(sorted.begin(), sorted.end());
  for (int32_t i = 0; i < 16; ++i) CHECK(sorted[static_cast<size_t>(i)] == i);
}

TEST_CASE(random_alive_compas_some_dead_samples_alive_only) {
  Mt19937Rng rng(7);
  std::vector<uint8_t> alive(16, 1);
  alive[3] = alive[9] = alive[15] = 0;
  for (int trial = 0; trial < 20; ++trial) {
    const std::vector<int32_t> compas = random_alive_compas(alive, rng);
    CHECK(compas.size() == 16);
    for (const int32_t c : compas) CHECK(alive[static_cast<size_t>(c)] == 1);
  }
}

TEST_CASE(random_alive_compas_all_dead_is_permutation) {
  Mt19937Rng rng(3);
  const std::vector<uint8_t> alive(8, 0);
  const std::vector<int32_t> compas = random_alive_compas(alive, rng);
  std::vector<int32_t> sorted = compas;
  std::sort(sorted.begin(), sorted.end());
  for (int32_t i = 0; i < 8; ++i) CHECK(sorted[static_cast<size_t>(i)] == i);
}

TEST_CASE(gather_clone_reads_original_values) {
  const std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f};
  // Chain 0<-1<-2: companion values must come from the ORIGINAL array, so
  // walker 2 gets the original 20, not walker 1's cloned value.
  const std::vector<int32_t> companions = {0, 0, 1, 3};
  const std::vector<uint8_t> will_clone = {0, 1, 1, 0};
  const std::vector<float> out = gather_clone(data, companions, will_clone);
  CHECK(out[0] == 10.0f);
  CHECK(out[1] == 10.0f);
  CHECK(out[2] == 20.0f);
  CHECK(out[3] == 40.0f);
}
