#include "optimization/controller.hpp"
#include <algorithm>
namespace fg::optimization {
RunController::RunController(Benchmark& b,const Settings& s)
 :archive(b,s.periodic),placement(fractal::descendant_lineage(s.seed,0x504c4143)),benchmark(b),
  scheduling(fractal::descendant_lineage(s.seed,0x53434845)),enabled(s.json["controller_enabled"].flag(false)),
  baseline_population(s.walkers),exploration_population(s.walkers),baseline_scale(s.json["perturbation_std"].num()),round_best(s.worst()),experiment_seed(uint64_t(s.json["seed"].num())) {}
void RunController::configure(const Settings& next,const Settings& previous) {
  const bool on=next.json["controller_enabled"].flag(false);
  if(on&&!enabled) {
    baseline_population=exploration_population=next.walkers;baseline_scale=next.json["perturbation_std"].num();
    start_evaluations=last_improvement=benchmark.evaluations;round_best=benchmark.best_observed;candidates.clear();
    exploration_evaluations=focused_evaluations=0;
  }
  enabled=on;
  if(next.json["restart_token"].num()!=previous.json["restart_token"].num()) requested=true;
  archive.set_periodic(next.periodic);
}
void RunController::observe(const double* x,double value) {
  if(!enabled&&!requested) return;
  const bool minimize=benchmark.config["objective"].str("minimize")=="minimize";
  if(!std::isfinite(round_best) || (minimize?value<round_best-1e-10*std::max(1.,std::abs(round_best)):value>round_best+1e-10*std::max(1.,std::abs(round_best)))) {
    round_best=value;last_improvement=benchmark.evaluations;
  }
  BasinEntry entry;entry.position.assign(x,x+benchmark.d);entry.objective=value;
  if(std::any_of(candidates.begin(),candidates.end(),[&](const auto& old){return archive.distance(old.position,entry.position)<1e-4;})) return;
  candidates.push_back(std::move(entry));
  std::stable_sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b){return minimize?a.objective<b.objective:a.objective>b.objective;});
  if(candidates.size()>32) candidates.resize(32);
}
bool RunController::wants_restart(const Settings& s,bool finished,bool alive) const {
  if(requested) {reason="manual";return true;}
  if(!enabled) return false;
  const uint64_t allowance=50*uint64_t(benchmark.d)*s.walkers,interval=10*uint64_t(benchmark.d)*s.walkers;
  const uint64_t spent=benchmark.evaluations-start_evaluations;
  if(finished) reason="algorithm finished";
  else if(!alive) reason="no valid walkers";
  else if(spent>=allowance) reason="round allowance";
  else if(!benchmark.stochastic && spent>=interval && benchmark.evaluations-last_improvement>=interval) reason="stalled";
  else return false;
  return true;
}
RoundChoice RunController::next(const Settings& s) const {
  RoundChoice choice(s.json,scheduling);
  const uint64_t spent=benchmark.evaluations-start_evaluations;
  const uint64_t explore=exploration_evaluations+(regime!="focused"?spent:0),focused=focused_evaluations+(regime=="focused"?spent:0);
  choice.regime=explore<=focused?"exploration":"focused";
  int population=s.walkers;double scale=s.json["perturbation_std"].num();
  if(enabled) {
    if(choice.regime=="exploration") {
      if(s.json["population_auto"].flag(true)) population=std::min(s.max_walkers,std::max(baseline_population,exploration_population*2));
      if(s.json["scale_auto"].flag(true)) scale=baseline_scale;
    } else {
      const int lower=std::max({2,std::max(s.elites,int(s.json["population_imports"].num()))+int(s.json["population_imports"].num()),(baseline_population+3)/4});
      const int upper=std::max(lower,exploration_population/2);
      if(s.json["population_auto"].flag(true)) population=std::clamp(int(std::exp(std::log(double(lower))+choice.random.uniform01()*std::log(double(upper)/lower))),lower,upper);
      if(s.json["scale_auto"].flag(true)) scale=baseline_scale*std::exp(std::log(.01)*(1-choice.random.uniform01()));
    }
  } else choice.regime="manual";
  if(enabled && s.json["population_auto"].flag(true)) population=std::min(population,population_capacity(benchmark,s));
  if(evaluated_perturbation(s.perturbation)) {
    // The user bounds remain fixed. Focused rounds lower only the effective
    // upper endpoint, interpolated logarithmically inside those bounds.
    const double fraction=enabled && s.json["scale_auto"].flag(true) && choice.regime=="focused"
                              ? choice.random.uniform01() : 1;
    choice.config.object["adaptive_round_fraction"]=number(fraction);
    scale=s.json["perturbation_std"].num();
  }
  choice.config.object["round_id"]=number(round+1);
  choice.config.object["walkers"]=number(population);choice.config.object["perturbation_std"]=number(scale);
  choice.config.object["round_seed"]=number(fractal::descendant_lineage(experiment_seed,round+1)&0x7fffffffULL);
  return choice;
}
void RunController::finish(const Settings& s,const Json& geometry,const Json& refinements) {
  for(const auto& result:refinements.array) {
    BasinEntry entry;for(const auto& value:result["position"].array) entry.position.push_back(value.num());
    if(entry.position.size()!=size_t(benchmark.d) || !benchmark.valid(entry.position.data())) continue;
    entry.objective=result["objective"].num();entry.refined=true;entry.refinement_cost=uint64_t(result["cost"].num());
    auto found=std::find_if(candidates.begin(),candidates.end(),[&](const auto& old){return archive.distance(old.position,entry.position)<1e-8;});
    if(found!=candidates.end()) {found->refined=true;found->refinement_cost+=entry.refinement_cost;}
    else candidates.push_back(std::move(entry));
  }
  if(candidates.empty() && !benchmark.best_position.empty()) {
    BasinEntry e;e.position=benchmark.best_position;e.objective=benchmark.best_observed;candidates.push_back(e);
  }
  if(benchmark.stochastic) {
    if(candidates.size()>4) candidates.resize(4);
    for(auto& e:candidates) {
      double mean=0,moment=0;
      for(int k=0;k<3;++k) {const double value=benchmark.evaluate_optimization(e.position.data(),&placement);const double delta=value-mean;mean+=delta/(k+1);moment+=delta*(value-mean);}
      e.objective=mean;e.uncertainty=std::sqrt(moment/6);
    }
  }
  const uint64_t cost=benchmark.evaluations-start_evaluations;
  if(regime=="focused") focused_evaluations+=cost;else exploration_evaluations+=cost;
  for(auto& e:candidates) {
    e.cost=cost;e.settings.kind=Json::Object;
    static const auto known=Settings(JsonReader(std::string("{}")).read()).json.object;
    for(const auto& field:known) e.settings.object[field.first]=s.json[field.first];
    double nearest=.1;
    for(const auto& model:geometry.array) {
      std::vector<double> anchor;for(const auto& v:model["anchor"].array) anchor.push_back(v.num());
      const double distance=archive.distance(e.position,anchor);
      if(distance<nearest) {nearest=distance;e.geometry=model;}
    }
    std::vector<double> distances;
    for(const auto& other:candidates) {
      const double distance=archive.distance(e.position,other.position);
      if(distance>0 && distance<=.1) distances.push_back(distance);
    }
    if(!distances.empty()) {std::sort(distances.begin(),distances.end());e.radius=std::clamp(distances[distances.size()/2],1e-4,.1);}
  }
  uint64_t visits=0;for(const auto& e:archive.entries()) visits+=e.visits;
  if(collect_basin_events) {
    BasinArchive evidence(benchmark,s.periodic);
    evidence.complete_round(candidates,round,reason=="stalled");
    auto event=JsonReader(evidence.export_json()).read();
    event.object["member_round"]=number(round);
    event.object["stalled"].kind=Json::Boolean;event.object["stalled"].number=reason=="stalled";
    basin_events.push_back(std::move(event));
  }
  archive.complete_round(candidates,round,reason=="stalled");
  uint64_t after=0;for(const auto& e:archive.entries()) after+=e.visits;
  if(after>visits && std::any_of(archive.entries().begin(),archive.entries().end(),[&](const auto& e){return e.last_round==round && e.visits>1;})) repeated_evaluations+=cost;
}
void RunController::begin(const Settings& s,const RoundChoice& choice,uint64_t start) {
  scheduling=choice.random;++round;round_steps=0;regime=choice.regime;requested=false;
  if(regime=="exploration") exploration_population=s.walkers;
  start_evaluations=last_improvement=start;round_best=s.worst();candidates.clear();
}
Json RunController::status(const Settings& s) const {
  Json result;result.kind=Json::Object;
  result.object["round"]=number(round);result.object["round_iteration"]=number(round_steps);
  result.object["round_evaluations"]=number(benchmark.evaluations-start_evaluations);
  result.object["global_evaluations"]=number(benchmark.evaluations);
  result.object["remaining_evaluations"]=number(s.max_evaluations>benchmark.evaluations?s.max_evaluations-benchmark.evaluations:0);
  result.object["regime"].kind=Json::String;result.object["regime"].string=regime;
  result.object["restart_reason"].kind=Json::String;result.object["restart_reason"].string=reason;
  result.object["restart_pending"].kind=Json::Boolean;result.object["restart_pending"].number=requested;
  result.object["repeated_basin_evaluations"]=number(repeated_evaluations);
  result.object["basins"]=archive.summary();return result;
}
}  // namespace fg::optimization
