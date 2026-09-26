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
  void update_perturbation() { perturbation->observed_update(); }
  void begin_geometry_step() { begin_geometry(*perturbation); }
  void set_geometry(bool enabled) { enable_geometry(*perturbation,b,s.json,enabled); }
  Json visual_geometry() const { return geometry_diagnostics(*perturbation); }
  bool uses_cloning_evidence() const override { return perturbation->uses_cloning_evidence() || bool(perturbation->observer); }
  void observe_cloning(const std::vector<std::vector<char>>&, const fractal::SelectionEvidence&) override;
  std::unique_ptr<Perturbation> prepare_settings(const Settings& next) const {
    return retune_perturbation(*perturbation, b, s.json, next.json);
  }
  void configure(Settings next, std::unique_ptr<Perturbation> proposal) {
    s = std::move(next);
    perturbation = std::move(proposal);
  }
  Json geometry() const { return perturbation_geometry(*perturbation); }
  void restore_geometry(const Json& geometry) {restore_perturbation_geometry(*perturbation,geometry);}
  Json diagnostics() const { return perturbation_diagnostics(*perturbation); }
  uint64_t lineage(const std::vector<char>&) const;
  void freeze(const std::vector<std::vector<char>>&);
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
  size_t bytes() const { return 1 + sizeof(double) + sizeof(float) * b.d + sizeof(uint64_t); }
  void encode(std::vector<char>&, const float*, double, uint64_t family = 1) const;
};
}  // namespace fg::optimization
