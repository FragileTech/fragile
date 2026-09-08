#include "fractal_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensor_ops.hpp"
#include "thread_pool.hpp"

namespace fg {

// ---- sampler ---------------------------------------------------------------

void FractalTreeSampler::sample_companions_into(const std::vector<uint8_t>& alive, Rng& rng,
                                                std::vector<int32_t>& out) const {
  random_alive_compas_into(alive, rng, out, companion_scratch_);
}
void FractalTreeSampler::sample_uniforms_into(int32_t n, Rng& rng, std::vector<float>& out) const {
  out.resize(n);
  for (auto& v : out) v = rng.uniform01();
}
void FractalTreeSampler::sample_actions_into(int32_t n, int32_t count, Rng& rng,
                                             std::vector<int32_t>& out) const {
  out.resize(n);
  for (auto& a : out) a = int32_t(rng.randint(0, count));
}
void FractalTreeSampler::sample_dt_into(int32_t n, int32_t low, int32_t high, Rng& rng,
                                        std::vector<int32_t>& out) const {
  out.resize(n);
  for (auto& d : out) d = int32_t(rng.randint(low, high + 1));
}
std::vector<int32_t> FractalTreeSampler::sample_companions(const std::vector<uint8_t>& alive,
                                                           Rng& rng) const {
  std::vector<int32_t> out;
  sample_companions_into(alive, rng, out);
  return out;
}
std::vector<float> FractalTreeSampler::sample_uniforms(int32_t n, Rng& rng) const {
  std::vector<float> out;
  sample_uniforms_into(n, rng, out);
  return out;
}
std::vector<int32_t> FractalTreeSampler::sample_actions(int32_t n, int32_t count, Rng& rng) const {
  std::vector<int32_t> out;
  sample_actions_into(n, count, rng, out);
  return out;
}
std::vector<int32_t> FractalTreeSampler::sample_dt(int32_t n, int32_t low, int32_t high,
                                                   Rng& rng) const {
  std::vector<int32_t> out;
  sample_dt_into(n, low, high, rng, out);
  return out;
}

FractalTree::FractalTree(BatchEnv& env, FractalTreeParams params, std::unique_ptr<Rng> rng,
                         std::unique_ptr<FractalTreeSampler> sampler)
    : env_(env),
      params_(params),
      rng_(rng ? std::move(rng) : std::make_unique<Mt19937Rng>(params.seed)),
      sampler_(sampler ? std::move(sampler) : std::make_unique<FractalTreeSampler>()),
      backend_(env, adapter_visits_),
      core_(backend_, *sampler_, *rng_, params_),
      count_visits_(core_.count_visits_),
      state_(core_.state_),
      visits_(core_.visits_),
      total_steps_(core_.total_steps_),
      total_clones_(core_.total_clones_),
      total_frames_(core_.total_frames_),
      iteration_(core_.iteration_) {
  params_.start_walkers = std::max(1, params_.start_walkers);
  params_.min_leafs = std::max(1, params_.min_leafs);
  params_.max_walkers = std::max(params_.start_walkers, params_.max_walkers);
}
void FractalTree::reset() {
  core_.reset();
  best_frame_.clear();
}
int32_t FractalTree::best_index() const { return core_.best_index(); }
std::pair<int32_t, float> FractalTree::get_best_walker() const { return core_.get_best_walker(); }
StepInfo FractalTree::step() {
  auto info = core_.step();
  const int n = state_.n;
  // Showcase: the env's display score when it has one (read from the stored
  // per-walker info, never from the env's batch cache), else the best
  // cumulative reward. Walkers without a state cannot be rendered.
  int32_t best = best_index();
  if (env_.has_display_score() && env_.has_walker_info()) {
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_scored = -1;
    for (int32_t i = 0; i < n; ++i) {
      const auto ui = static_cast<size_t>(i);
      if (state_.states[ui].empty()) continue;
      if (state_.info[ui].score > best_score) {
        best_score = state_.info[ui].score;
        best_scored = i;
      }
    }
    if (best_scored >= 0) best = best_scored;
  }
  info.best_walker_idx = best;
  if (params_.record_frames && !state_.states[static_cast<size_t>(best)].empty()) {
    env_.render_frame(state_.states[static_cast<size_t>(best)], best_frame_);
  }
  return info;
}

}  // namespace fg
