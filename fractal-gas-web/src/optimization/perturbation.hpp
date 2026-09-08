#pragma once
#include <functional>
#include <memory>

#include "fractal/proposal.hpp"
#include "optimization/benchmark.hpp"

namespace fg::optimization {
using fractal::Perturbation;
using fractal::PerturbationContext;
using fractal::PerturbationTransition;
using PerturbationFactory =
    std::function<std::unique_ptr<Perturbation>(const Benchmark&, const Json&)>;
void register_perturbation(const std::string& id, const std::string& name,
                           PerturbationFactory factory, const Json& parameters,
                           const Json& algorithms = Json{});
std::unique_ptr<Perturbation> make_perturbation(const Benchmark&, const Json&);
Json perturbation_catalog();
}  // namespace fg::optimization
