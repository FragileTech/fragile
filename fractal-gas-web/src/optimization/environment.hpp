#pragma once
#include "optimization/engine.hpp"

namespace fg::optimization {
// Action IDs are seeds, exactly representable in the existing float action
// recorder. A recorded (state, action, dt) reproduces proposals and noisy
// rewards. Adaptive proposals additionally require the same frozen model;
// planner search and execution share that model until the next cycle.
class BenchmarkEnvironment final : public BatchEnv {
 public:
  void collect_perturbations(bool enabled) { collecting = enabled; }
  void update_perturbation() { perturbation->update(); }
  Benchmark& b;
  Settings s;
  BenchmarkEnvironment(Benchmark&, const Settings&);
  int32_t n_actions() const override { return 1 << 24; }
  int32_t obs_dim() const override { return b.d; }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }
  void render_frame(const std::vector<char>&,
                    std::vector<uint8_t>& out) override {
    out.clear();
  }
  void reset(std::vector<char>&, std::vector<float>&) override;
  double decode(const std::vector<char>&, float*) const;
  void step_batch(const std::vector<std::vector<char>>&,
                  const std::vector<int32_t>&, const std::vector<int32_t>&,
                  std::vector<std::vector<char>>&, std::vector<float>&,
                  std::vector<float>&, std::vector<uint8_t>&,
                  std::vector<uint8_t>&) override;

 private:
  bool collecting = true;
  std::unique_ptr<Perturbation> perturbation;
  size_t bytes() const { return 1 + sizeof(double) + sizeof(float) * b.d; }
  void encode(std::vector<char>&, const float*, double) const;
};
}  // namespace fg::optimization
