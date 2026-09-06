// Compile-time archetypes. No registry lookups or type data in mutable states.
#pragma once
#include "control/json.hpp"

namespace fg::control {
class AgentTypes {
 public:
  explicit AgentTypes(const Json& definitions);
  Json body(const Json& instance) const;

 private:
  std::map<std::string, Json> definitions_, resolved_;
  Json resolve(const std::string& name, std::vector<std::string>& visiting);
};
}  // namespace fg::control
