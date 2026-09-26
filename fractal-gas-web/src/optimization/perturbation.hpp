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
inline bool evaluated_perturbation(const std::string& id) {return id=="adaptive_fractal" || id=="cloning_guided";}
inline bool bounded_scale_perturbation(const std::string& id) {return evaluated_perturbation(id);}
Json perturbation_diagnostics(const Perturbation&);
Json proposal_visual_geometry(const Perturbation&);
void enable_geometry(Perturbation&, const Benchmark&, const Json&, bool);
void inherit_geometry(const Perturbation&, Perturbation&, const Benchmark&, const Json&);
Json geometry_diagnostics(const Perturbation&);
void begin_geometry(Perturbation&);

Json perturbation_geometry(const Perturbation&);
void restore_perturbation_geometry(Perturbation&,const Json&);
std::unique_ptr<Perturbation> retune_perturbation(const Perturbation&, const Benchmark&,
                                               const Json& previous, const Json& next);
}  // namespace fg::optimization
