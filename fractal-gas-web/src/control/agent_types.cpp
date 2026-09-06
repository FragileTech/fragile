#include "control/agent_types.hpp"

#include <algorithm>

namespace fg::control {
namespace {
void overlay(Json& target, const Json& source) {
  if (source.kind != Json::Null && source.kind != Json::Object)
    throw std::invalid_argument("Agent physics defaults must be an object");
  target.kind = Json::Object;
  for (const auto& [key, value] : source.object) {
    if (key == "agent_type")
      throw std::invalid_argument("Use extends to inherit agent types");
    target.object[key] = value;
  }
}
}  // namespace
AgentTypes::AgentTypes(const Json& definitions) {
  if (definitions.kind != Json::Null && definitions.kind != Json::Object)
    throw std::invalid_argument("agent_types must be an object");
  if (definitions.object.size() > 256)
    throw std::invalid_argument("At most 256 agent types are supported");
  definitions_ = definitions.object;
  std::vector<std::string> visiting;
  for (const auto& [name, definition] : definitions_) resolve(name, visiting);
}
Json AgentTypes::resolve(const std::string& name,
                         std::vector<std::string>& visiting) {
  if (resolved_.count(name)) return resolved_.at(name);
  auto found = definitions_.find(name);
  if (found == definitions_.end())
    throw std::invalid_argument("Unknown agent type: " + name);
  if (std::find(visiting.begin(), visiting.end(), name) != visiting.end())
    throw std::invalid_argument("Cyclic agent type inheritance: " + name);
  const Json& definition = found->second;
  if (name.empty() || definition.kind != Json::Object)
    throw std::invalid_argument(
        "Agent types need a name and an object definition");
  visiting.push_back(name);
  Json defaults;
  if (definition["extends"].kind != Json::Null)
    defaults = resolve(definition["extends"].str(), visiting);
  overlay(defaults, definition["physics"]);
  visiting.pop_back();
  return resolved_[name] = std::move(defaults);
}
Json AgentTypes::body(const Json& instance) const {
  if (instance.kind != Json::Object)
    throw std::invalid_argument("Body must be an object");
  if (instance["agent_type"].kind == Json::Null) return instance;
  const std::string name = instance["agent_type"].str();
  auto found = resolved_.find(name);
  if (found == resolved_.end())
    throw std::invalid_argument("Unknown agent type: " + name);
  Json result = found->second;
  for (const auto& [key, value] : instance.object) result.object[key] = value;
  return result;
}
}  // namespace fg::control
