#include "optimization/environment.hpp"
#include "optimization/adaptive.hpp"

#include <cstring>

namespace fg::optimization {
void BenchmarkEnvironment::observe_cloning(const std::vector<std::vector<char>>& states,
                                         const fractal::SelectionEvidence& evidence) {
  if(!collecting || (!perturbation->uses_cloning_evidence() && !perturbation->observer)) return;
  fractal::FrozenPopulation pop;pop.dimensions=b.d;
  pop.positions.resize(states.size()*b.d);
  for(size_t i=0;i<states.size();++i) {
    float* x=pop.positions.data()+i*b.d;
    double value=decode(states[i],x);
    pop.objectives.push_back(value);pop.valid.push_back(b.valid(x)&&std::isfinite(value));
    pop.families.push_back(lineage(states[i]));
  }
  if(perturbation->uses_cloning_evidence()) {
    perturbation->observe_cloning(pop,evidence);
    if(!s.planning()) perturbation->update();
  }
  if(perturbation->observer) {
    perturbation->observer->cloning(pop,evidence);
    for(size_t k=0;k<evidence.sources.size();++k) {
      const size_t i=evidence.destinations.empty()?k:size_t(evidence.destinations[k]);
      const size_t source=size_t(evidence.sources[k]);
      if(i==source || i>=states.size() || source>=states.size()) continue;
      if(s.algorithm=="wave" && i<size_t(s.elites)) continue;
      perturbation->observer->movement(pop.positions.data()+i*b.d,pop.positions.data()+source*b.d,b.d,pop.families[i],"cloning");
    }
  }
}

BenchmarkEnvironment::BenchmarkEnvironment(Benchmark& bench,
                                           const Settings& settings)
    : b(bench), s(settings), perturbation(make_perturbation(b, s.json)) {}
void BenchmarkEnvironment::encode(std::vector<char>& state, const float* x,
                                  double u,uint64_t family) const {
  state.assign(bytes(), 0);
  state[0] = 1;
  std::memcpy(state.data()+bytes()-sizeof(family),&family,sizeof(family));
  std::memcpy(state.data() + 1, &u, sizeof(u));
  std::memcpy(state.data() + 1 + sizeof(u), x, sizeof(float) * b.d);
}
void BenchmarkEnvironment::reset(std::vector<char>& state,
                                 std::vector<float>& obs) {
  perturbation->reset();
  collecting = true;
  state.assign(bytes(), 0);
  obs.assign(b.d, 0);
  if (s.planning()) {
    OptimizationRng rng(s.seed);
    b.initial(obs.data(), rng);
    const double u = b.evaluate_optimization(obs.data(), &rng);
    encode(state, obs.data(), u);
  }
}
double BenchmarkEnvironment::decode(const std::vector<char>& state,
                                    float* x) const {
  if (state.size() != bytes()) return INFINITY;
  double u;
  std::memcpy(&u, state.data() + 1, sizeof(u));
  std::memcpy(x, state.data() + 1 + sizeof(u), sizeof(float) * b.d);
  return state[0] ? u : INFINITY;
}
uint64_t BenchmarkEnvironment::lineage(const std::vector<char>& state) const {
  uint64_t result=0;
  if(state.size()==bytes()) std::memcpy(&result,state.data()+bytes()-sizeof(result),sizeof(result));
  return result;
}
void BenchmarkEnvironment::freeze(const std::vector<std::vector<char>>& states) {
  if(!perturbation->tracks_trials() && !perturbation->observer) return;
  fractal::FrozenPopulation population;population.dimensions=b.d;
  population.positions.resize(states.size()*b.d);
  for(size_t i=0;i<states.size();++i) {
    float* x=population.positions.data()+i*b.d;
    double value=decode(states[i],x);
    population.valid.push_back(b.valid(x)&&std::isfinite(value));
    population.families.push_back(lineage(states[i]));
    population.objectives.push_back(value);
  }
  perturbation->observed_freeze(population);
}
void BenchmarkEnvironment::step_batch(
    const std::vector<std::vector<char>>& states,
    const std::vector<int32_t>& actions, const std::vector<int32_t>& dt,
    std::vector<std::vector<char>>& next, std::vector<float>& observations,
    std::vector<float>& rewards, std::vector<uint8_t>& dones,
    std::vector<uint8_t>& truncated) {
  std::vector<float> delta(b.d);
  for (size_t i = 0; i < states.size(); ++i) {
    if (actions[i] < 0 || actions[i] >= n_actions())
      throw std::invalid_argument("Invalid perturbation action seed");
    OptimizationRng rng((uint64_t(s.seed) << 24) | uint32_t(actions[i]));
    float* x = observations.data() + i * b.d;
    const bool initialized = states[i].size() == bytes() && states[i][0];
    const double old = initialized ? decode(states[i], x) : 0;
    if (!initialized) b.initial(x, rng);
    const uint64_t parent=lineage(states[i]);
    if(perturbation->tracks_trials()) {
      double value=old;bool valid=true;
      if(!initialized) { value=b.evaluate_optimization(x,&rng);valid=b.valid(x)&&std::isfinite(value); }
      const int draws=dt[i]-(initialized?0:1);
      if(draws>0 && valid) {
        auto trial=evaluate_adaptive_position(*perturbation,b,s.json,x,value,rng,
          {uint64_t(s.json["round_id"].num()),0,parent,uint64_t(actions[i]),0},!collecting,draws);
        std::copy(trial.position.begin(),trial.position.end(),x);
        value=trial.objective;valid=trial.valid;
      }
      valid &= std::isfinite(float(value))&&std::isfinite(float(s.score(value)-s.score(old)));
      encode(next[i],x,value,fractal::descendant_lineage(parent,actions[i]));
      rewards[i]=valid?float(s.score(value)-s.score(old)):0;
      dones[i]=!valid;truncated[i]=0;
      continue;
    }
    PerturbationTransition transition;
    transition.origin.assign(x, x + b.d);
    transition.parent=parent;transition.action=uint64_t(actions[i]);
    transition.source_scale=s.json["perturbation_std"].num(1);
    transition.displacement.assign(b.d, 0);
    for (int t = 0; t < dt[i]; ++t) {
      if (initialized || t > 0) {
        perturbation->sample_action(transition.origin.data(), x, delta.data(), b.d, rng);
        ++transition.draws;
        for (int k = 0; k < b.d; ++k) {
          x[k] += delta[k];
          transition.displacement[k] += delta[k];
        }
      }
      b.boundary(x, s.periodic ? "periodic" : s.json["boundary"].str()=="cma" ? "cma" : "none");
      if (!b.valid(x)) break;
    }
    transition.direction=transition.displacement;
    if (s.json["boundary"].str()=="cma")
      for (int k=0;k<b.d;++k) transition.displacement[k]=x[k]-transition.origin[k];
    const double u = b.evaluate_optimization(x, &rng);
    const bool valid = b.valid(x) && std::isfinite(u) &&
                       std::isfinite(float(u)) &&
                       std::isfinite(float(s.score(u) - s.score(old)));
    if (collecting && initialized && valid && transition.draws > 0) {
      transition.improvement = s.score(u) - s.score(old);
      perturbation->observed_transition(transition);
    }
    if(perturbation->observer && initialized && valid)
      perturbation->observer->movement(transition.origin.data(),x,b.d,parent,collecting?"proposal":"execution");
    encode(next[i], x, u,fractal::descendant_lineage(parent,actions[i]));
    rewards[i] = valid ? float(s.score(u) - s.score(old)) : 0;
    dones[i] = !valid;
    truncated[i] = 0;
  }
}
}  // namespace fg::optimization
