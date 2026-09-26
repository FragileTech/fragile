#include "optimization/perturbation.hpp"
#include "optimization/adaptive.hpp"
#include "optimization/cloning_guided.hpp"
#include <algorithm>
#include <cmath>

namespace fg::optimization {
namespace {
void text(Json& j,const char* key,const std::string& value) {
  j.object[key].kind=Json::String;j.object[key].string=value;
}
void flag(Json& j,const char* key,bool value) {
  j.object[key].kind=Json::Boolean;j.object[key].number=value;
}
// Contains no sampler and never calls a benchmark evaluation. These estimates
// describe evidence from this run, not counterfactual optimizer trajectories.
class GeometryObserver final : public fractal::ProposalObserver {
 public:
  const Perturbation& active;
  Json config;
  int d;
  double width;
  std::string strategy;
  bool velocity;
  uint64_t frame=0,sequence=0;
  size_t event_count=0;
  fractal::FrozenPopulation frozen;
  Json events, references;
  std::vector<std::pair<std::string,std::unique_ptr<Perturbation>>> estimates;
  GeometryObserver(const Perturbation& p,const Benchmark& b,const Json& c)
      :active(p),config(c),d(b.d),width(b.high-b.low),strategy(c["perturbation"].str("gaussian")),
       velocity(c["algorithm"].str()=="euclidean" &&
         !(evaluated_perturbation(strategy)&&c["adaptive_euclidean_mode"].str("velocity")=="position")) {
    events.kind=Json::Array;references.kind=Json::Array;
    for(const std::string id:{"local_covariance","adaptive_fractal","cloning_guided"}) {
      if(id==strategy || (id=="local_covariance" && d>256)) continue;
      Json options;options.kind=Json::Object;
      text(options,"algorithm","wave");text(options,"perturbation",id);
      for(const char* key:{"periodic","objective","boundary"}) if(c[key].kind!=Json::Null) options.object[key]=c[key];
      text(options,"adaptive_euclidean_mode",velocity?"velocity":"position");
      estimates.emplace_back(id,make_perturbation(b,options));
    }
  }
  GeometryObserver(const Perturbation& p,const Benchmark& b,const Json& c,const GeometryObserver& old)
      :GeometryObserver(p,b,c) {
    if(velocity!=old.velocity || c["boundary"].str()!=old.config["boundary"].str() || c["periodic"].flag()!=old.config["periodic"].flag()) return;
    frame=old.frame;sequence=old.sequence;frozen=old.frozen;events=old.events;
    references=old.references;event_count=old.event_count;
    for(auto& entry:estimates) for(const auto& source:old.estimates) if(entry.first==source.first) {
      Json options;options.kind=Json::Object;text(options,"algorithm","wave");text(options,"perturbation",entry.first);
      for(const char* key:{"periodic","objective","boundary"}) if(c[key].kind!=Json::Null) options.object[key]=c[key];
      text(options,"adaptive_euclidean_mode",velocity?"velocity":"position");
      entry.second=retune_perturbation(*source.second,b,options,options);
    }
  }
  void begin() override {++frame;events.array.clear();references.array.clear();event_count=0;}
  void freeze(const fractal::FrozenPopulation& p) override {
    frozen=p;
    for(auto& entry:estimates) entry.second->freeze_population(p);
  }
  void transition(const PerturbationTransition& input) override {
    auto t=input;
    if(!std::isfinite(t.improvement) || t.draws<=0) return;
    if(velocity) t.source_scale=config["perturbation_std"].num(1);
    for(auto& entry:estimates) {
      if(entry.first=="local_covariance" && t.source_scale>0) {
        auto measured=t;measured.scale*=t.source_scale;entry.second->observe(measured);
      }
    }
    if(t.direction.size()==size_t(d) && t.draws>0) {
      fractal::TrialOutcome o;o.trial.origin=t.origin;o.displacement=t.displacement;
      o.trial.identity.parent=t.parent;o.trial.identity.action=t.action;
      o.trial.direction=t.direction;o.improvement=t.improvement;
      const double scale=t.source_scale*t.scale*std::sqrt(double(t.draws));
      if(scale>0) {
        for(auto& v:o.trial.direction) v/=scale;
        o.trial.scale=scale;
        for(auto& entry:estimates) if(auto* p=dynamic_cast<AdaptiveExploration*>(entry.second.get())) p->observe_external(o);
      }
    }
  }
  void outcome(const fractal::TrialOutcome& o) override {
    for(auto& entry:estimates) {
      if(auto* adaptive=dynamic_cast<AdaptiveExploration*>(entry.second.get())) adaptive->observe_external(o);
      if(entry.first=="local_covariance" && o.valid) {
        PerturbationTransition t;t.origin=o.trial.origin;t.displacement=o.displacement;
        if(velocity) {t.displacement=o.trial.direction;for(auto& v:t.displacement) v*=o.trial.scale;}
        t.draws=velocity?1:o.draws;t.scale=o.trial.scale;t.improvement=o.improvement;
        // Local Gaussian fits measured displacements; source scale removes
        // proposal magnitude without claiming they were draws from this model.
        entry.second->observe(t);
      }
    }
  }
  void cloning(const fractal::FrozenPopulation& p,const fractal::SelectionEvidence& e) override {
    freeze(p);
    for(auto& entry:estimates) if(entry.second->uses_cloning_evidence()) {
      entry.second->observe_cloning(p,e);entry.second->update();
    }
  }
  void movement(const float* from,const float* to,int dimensions,uint64_t parent,const std::string& kind) override {
    if(dimensions!=d) return;
    ++event_count;
    // Bound high-dimensional recordings as well as ordinary 2D/3D scenes.
    const size_t limit=std::min<size_t>(512,262144/std::max(1,d));
    const auto count=std::count_if(events.array.begin(),events.array.end(),[&](const Json& e){return e["kind"].str()==kind;});
    if(size_t(count)>=std::max<size_t>(1,limit/3) || events.array.size()>=limit) return;
    for(int k=0;k<d;++k) if(!std::isfinite(from[k])||!std::isfinite(to[k])) return;
    Json j;j.kind=Json::Object;
    j.object["origin"]=array(std::vector<double>(from,from+d));
    j.object["destination"]=array(std::vector<double>(to,to+d));
    text(j,"kind",kind);text(j,"parent",std::to_string(parent));
    j.object["event"]=number(++sequence);events.array.push_back(std::move(j));
  }
  void reference(const float* origin,int dimensions,double scale) override {
    if(dimensions!=d || references.array.size()>=16) return;
    Json model;model.kind=Json::Object;
    model.object["anchor"]=array(std::vector<double>(origin,origin+d));
    model.object["shape"]=array(std::vector<double>(d,1));model.object["columns"]=number(1);
    model.object["scale"]=number(scale);text(model,"representation","diagonal");references.array.push_back(std::move(model));
  }
  void update() override {for(auto& entry:estimates) entry.second->update();}
  Json snapshot() const {
    Json result;result.kind=Json::Object;result.object["version"]=number(1);
    result.object["dimensions"]=number(d);result.object["step"]=number(frame);
    result.object["events"]=events;result.object["event_count"]=number(event_count);
    text(result,"units",velocity?"velocity kick":"position proposal");
    flag(result,"periodic",config["periodic"].flag(false));
    Json methods;methods.kind=Json::Array;
    auto append=[&](const std::string& id,const Perturbation* p,bool is_active) {
      Json method;method.kind=Json::Object;text(method,"id",id);flag(method,"active",is_active);
      Json models=p?proposal_visual_geometry(*p):Json{};
      if(models.kind!=Json::Array) {models.kind=Json::Array;models.array.clear();}
      if(is_active && (id=="gaussian"||id=="uniform"||id=="gas_adaptive")) {
        const size_t count=std::min<size_t>(16,frozen.valid.size());
        for(size_t i=0;i<count;++i) if(frozen.valid[i]) {
          Json model;model.kind=Json::Object;
          model.object["anchor"]=array(std::vector<double>(frozen.positions.begin()+i*d,frozen.positions.begin()+(i+1)*d));
          model.object["shape"]=array(std::vector<double>(d,1));model.object["columns"]=number(1);
          text(model,"representation","diagonal");
          // GAS scale depends on post-cloning normalized flow context; it is
          // supplied by its adapter rather than guessed from objective ranks.
          double scale=config["perturbation_std"].num(1);
          model.object["scale"]=number(scale);models.array.push_back(std::move(model));
        }
        if(id=="gas_adaptive") models=references;
      }
      text(method,"status",!p?"Unavailable above 256 dimensions":models.array.empty()?"Waiting for compatible evidence":"ready");
      if(id=="adaptive_fractal") text(method,"component","Local Gaussian component; excludes broad/difference mixture");

      text(method,"scale_reference","Objective of nearest frozen walker at each anchor");
      for(auto& m:models.array) {
        size_t nearest=frozen.valid.size();double distance=INFINITY;
        for(size_t i=0;i<frozen.valid.size();++i) if(frozen.valid[i]) {
          double sum=0;
          for(int k=0;k<d;++k) {
            double delta=m["anchor"].items()[k].num()-frozen.positions[i*d+k];
            if(config["periodic"].flag(false)) delta=std::remainder(delta,width);
            sum+=delta*delta;
          }
          if(sum<distance) {distance=sum;nearest=i;}
        }
        const double objective=nearest<frozen.objectives.size()?frozen.objectives[nearest]:NAN;
        if(auto* adaptive=dynamic_cast<const AdaptiveExploration*>(p)) m.object["scale"]=number(adaptive->visual_scale(objective));
        if(auto* guided=dynamic_cast<const CloningGuided*>(p)) m.object["scale"]=number(guided->visual_scale(objective));
      }
      const size_t total=models.array.empty()?0:size_t(models.array.front()["available_models"].num(models.array.size()));
      // Keep JSON and recording admission bounded even for dense CMA-sized models.
      size_t coefficients=0,keep=0;
      for(const auto& m:models.array) {
        coefficients+=m["shape"].items().size()+5*size_t(d);
        if(coefficients>std::max<size_t>(262144,size_t(d)*d+5*size_t(d))) break;
        ++keep;
      }
      models.array.resize(keep);method.object["model_count"]=number(total);
      if(total && !keep) text(method,"status","Model exceeds diagnostic payload limit");
      if(velocity) {
        const double multiplier=std::sqrt(-std::expm1(-2*config["gamma"].num(1)*config["delta_t"].num(.01))/config["beta"].num(1));
        for(auto& m:models.array) m.object["scale"]=number(m["scale"].num(1)*multiplier);
      }
      method.object["models"]=std::move(models);methods.array.push_back(std::move(method));
    };
    append(strategy,&active,true);
    for(const auto& entry:estimates) append(entry.first,entry.second.get(),false);
    if(d>256&&strategy!="local_covariance") append("local_covariance",nullptr,false);
    result.object["methods"]=std::move(methods);return result;
  }
};
}
void enable_geometry(Perturbation& p,const Benchmark& b,const Json& c,bool enabled) {
  if(!enabled) {p.observer.reset();return;}
  p.observer=std::make_shared<GeometryObserver>(p,b,c);
}
void inherit_geometry(const Perturbation& old,Perturbation& p,const Benchmark& b,const Json& c) {
  if(const auto* observer=dynamic_cast<const GeometryObserver*>(old.observer.get()))
    p.observer=std::make_shared<GeometryObserver>(p,b,c,*observer);
  else p.observer.reset();
}
Json geometry_diagnostics(const Perturbation& p) {
  if(const auto* observer=dynamic_cast<const GeometryObserver*>(p.observer.get())) return observer->snapshot();
  return Json{};
}
void begin_geometry(Perturbation& p) {if(p.observer) p.observer->begin();}
}
