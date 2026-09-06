#include "control/extensions.hpp"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
namespace fg::control {
namespace {
std::map<std::string, ExtensionCompiler>& registry() {
  static std::map<std::string, ExtensionCompiler> value;
  return value;
}
std::mutex& registry_mutex() {
  static std::mutex value;
  return value;
}
}  // namespace
void register_world_extension(const std::string& name,
                              ExtensionCompiler compiler) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  if (name.empty() || !compiler || !registry().emplace(name, compiler).second)
    throw std::invalid_argument("Invalid or duplicate world extension: " +
                                name);
}
WorldExtension compile_world_extension(const Json& j) {
  if (j.kind != Json::Object)
    throw std::invalid_argument("World extension must be an object");
  ExtensionCompiler compiler;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = registry().find(j["kind"].str());
    if (it == registry().end())
      throw std::invalid_argument("Unknown world extension: " +
                                  j["kind"].str());
    compiler = it->second;
  }
  auto extension = compiler(j);
  if (extension.initial_state.size() > 4096 ||
      extension.observation_size > 4096 ||
      (!extension.step && !extension.observe) ||
      (extension.observation_size && !extension.observe))
    throw std::invalid_argument("Invalid world extension layout");
  for (float value : extension.initial_state)
    if (!std::isfinite(value))
      throw std::invalid_argument("Invalid world extension initial state");
  return extension;
}
}  // namespace fg::control
