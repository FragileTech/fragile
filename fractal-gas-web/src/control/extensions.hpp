#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "control/json.hpp"

namespace fg::control {
struct Scene;
struct StepResult;
struct WorldExtension;
using ExtensionStep = void (*)(const Scene&, const WorldExtension&, float*,
                               const float*, StepResult&);
using ExtensionObserve = void (*)(const Scene&, const WorldExtension&,
                                  const float*, float*);
struct WorldExtension {
  uint32_t state_offset = 0, observation_size = 0;
  std::vector<float> initial_state;
  std::shared_ptr<const void> parameters;
  ExtensionStep step = nullptr;
  ExtensionObserve observe = nullptr;
};
// Registered hooks run in scene order after each physical frame. All mutable
// data belongs in the packed row; callbacks must be thread-safe and
// deterministic.
using ExtensionCompiler = std::function<WorldExtension(const Json&)>;
void register_world_extension(const std::string& name,
                              ExtensionCompiler compiler);
WorldExtension compile_world_extension(const Json& definition);
}  // namespace fg::control
