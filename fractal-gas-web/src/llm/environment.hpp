#pragma once
#include "env.hpp"
#include <cstring>
#include <functional>
#include <stdexcept>
#include <cmath>

namespace fg::llm {
struct Snapshot {
  uint32_t id = 0, tokens = 0;
  double logp = 0;
  double utility = 0;
  uint32_t status = 0; // 0 active, 1 model stop, 2 sequence cap
};
struct Request { Snapshot source; int32_t action, duration; };
struct Result { Snapshot state; std::vector<float> embedding; bool skipped = false; };
using Transport = std::function<std::vector<Result>(const std::vector<Request>&)>;
class LlmEnvironment final : public BatchEnv {
 public:
  LlmEnvironment(int dimensions, Transport transport)
      : dimensions_(dimensions), transport_(std::move(transport)) {
    if (dimensions < 1 || dimensions > 65536) throw std::invalid_argument("Invalid embedding dimensions");
  }
  static Snapshot decode(const std::vector<char>& data) {
    if (data.size() != sizeof(Snapshot)) throw std::invalid_argument("Invalid LLM snapshot");
    Snapshot s; std::memcpy(&s, data.data(), sizeof(s)); return s;
  }
  static std::vector<char> encode(const Snapshot& state) {
    std::vector<char> out(sizeof(state)); std::memcpy(out.data(), &state, sizeof(state)); return out;
  }
  bool best_candidate(const std::vector<char>& data) const override {
    // Completed answers live in the recording archive, not in protected slots.
    return data.size() == sizeof(Snapshot) && decode(data).tokens > 0 && decode(data).status == 0;
  }
  int32_t n_actions() const override { return 1 << 24; }
  int32_t obs_dim() const override { return dimensions_; }
  void reset(std::vector<char>& state, std::vector<float>& obs) override {
    state = encode({}); obs.assign(dimensions_, 0);
  }
  double score(const Snapshot& s) const { return s.utility; }
  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions, const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& next, std::vector<float>& obs,
                  std::vector<float>& rewards, std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override {
    if (states.size() != actions.size() || dt.size() != states.size())
      throw std::invalid_argument("Invalid token transition shape");
    std::vector<Request> requests;
    for (size_t i = 0; i < states.size(); ++i) {
      if (actions[i] < 0 || actions[i] >= n_actions() || dt[i] < 1)
        throw std::invalid_argument("Invalid token action or duration");
      requests.push_back({decode(states[i]), actions[i], dt[i]});
    }
    const auto results = transport_(requests);
    if (results.size() != requests.size()) throw std::invalid_argument("Incomplete token batch");
    // Validate the whole batch before writing any state rows.
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i]; const auto& before = requests[i].source;
      const bool unchanged = r.state.id == before.id && r.state.tokens == before.tokens &&
                             r.state.logp == before.logp && r.state.status == before.status && r.state.utility == before.utility;
      if (r.embedding.size() != size_t(dimensions_) || !std::isfinite(r.state.logp) || !std::isfinite(r.state.utility) ||
          r.state.logp > before.logp + 1e-8 || r.state.tokens < before.tokens ||
          r.state.tokens - before.tokens > uint32_t(dt[i]) || r.state.status > 2 ||
          ((r.skipped || before.status != 0) && !unchanged) ||
          (r.state.tokens == before.tokens && r.state.status == 0 && !r.skipped) ||
          !std::isfinite(float(score(r.state) - score(before))))
        throw std::invalid_argument("Invalid token transition result");
      for (float v : r.embedding) if (!std::isfinite(v))
        throw std::invalid_argument("Nonfinite LLM observation");
    }
    next.resize(states.size()); obs.resize(states.size() * dimensions_);
    rewards.resize(states.size()); dones.resize(states.size()); truncated.resize(states.size());
    actual_.resize(states.size());
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i];
      next[i] = encode(r.state);
      std::copy(r.embedding.begin(), r.embedding.end(), obs.begin() + i * dimensions_);
      rewards[i] = float(score(r.state) - score(requests[i].source));
      dones[i] = r.state.status == 1; truncated[i] = r.state.status == 2;
      actual_[i] = r.state.tokens - requests[i].source.tokens;
    }
  }
  int32_t frames_stepped(int i) const override { return actual_.at(i); }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }
  void render_frame(const std::vector<char>&, std::vector<uint8_t>& out) override { out.clear(); }
 private:
  int dimensions_; Transport transport_; std::vector<int32_t> actual_;
};
} // namespace fg::llm
