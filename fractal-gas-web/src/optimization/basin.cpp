#include "optimization/basin.hpp"
#include "optimization/adaptive.hpp"
#include <algorithm>
#include <set>

namespace fg::optimization {
namespace {
Json boolean(bool value) { Json j;j.kind=Json::Boolean;j.number=value;return j; }
Json text_value(const std::string& value) { Json j;j.kind=Json::String;j.string=value;return j; }
Json entry_json(const BasinEntry& e,bool geometry) {
  Json j;j.kind=Json::Object;
  j.object["id"]=number(e.id);j.object["position"]=array(e.position);
  j.object["objective"]=number(e.objective);j.object["uncertainty"]=number(e.uncertainty);
  j.object["radius"]=number(e.radius);j.object["confidence"]=number(e.confidence);
  j.object["discovery_round"]=number(e.discovery_round);j.object["last_round"]=number(e.last_round);
  j.object["visits"]=number(e.visits);j.object["cost"]=number(e.cost);
  j.object["refinement_cost"]=number(e.refinement_cost);
  j.object["validated"]=boolean(e.validated);j.object["refined"]=boolean(e.refined);
  j.object["settings"]=e.settings;
  if(geometry) j.object["geometry"]=e.geometry;
  return j;
}
}
BasinArchive::BasinArchive(const Benchmark& b,bool wrapping,size_t maximum)
 :dimensions(b.d),low(b.low),high(b.high),periodic(wrapping),
  minimize(b.config["objective"].str("minimize")=="minimize"),capacity(maximum) {
  if(capacity<1 || capacity>64) throw std::invalid_argument("Basin capacity must be between 1 and 64");
  compatibility.kind=Json::Object;
  for(auto key:{"benchmark","dimensions","low","high","coco_version","coco_problem_id","centers","stds","weights"})
    if(b.config[key].kind!=Json::Null) compatibility.object[key]=b.config[key];
  auto catalog=JsonReader(catalog_json()).read();
  for(const auto& benchmark:catalog["benchmarks"].array) if(benchmark["id"].str()==b.id)
    for(const auto& parameter:benchmark["parameters"].array) {
      const auto key=parameter["id"].str();
      if(key=="benchmark_seed" && b.config["centers"].kind!=Json::Null) continue;
      compatibility.object[key]=b.config[key].kind!=Json::Null?b.config[key]:parameter["default"];
    }
  for(const auto& benchmark:catalog["benchmarks"].array) if(benchmark["id"].str()==b.id)
    for(const auto& field:benchmark["parameters"].object) {
      if(field.first=="benchmark_seed" && b.config["centers"].kind!=Json::Null) continue;
      compatibility.object[field.first]=b.config[field.first].kind!=Json::Null?b.config[field.first]:field.second;
    }
  compatibility.object["objective"]=text_value(minimize?"minimize":"maximize");
  compatibility.object["periodic"]=boolean(periodic);
  compatibility.object["stochastic"]=boolean(b.stochastic);
}
double BasinArchive::distance(const std::vector<double>& a,const std::vector<double>& b) const {
  double sum=0;
  for(int j=0;j<dimensions;++j) {
    double delta=a[j]-b[j];if(periodic) delta=std::remainder(delta,high-low);
    sum+=std::pow(delta/(high-low),2);
  }
  return std::sqrt(sum/dimensions);
}
void BasinArchive::complete_round(std::vector<BasinEntry> candidates,uint64_t round,bool stalled) {
  for(auto& e:records) e.confidence*=.9;
  std::stable_sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b){return better(a.objective,b.objective);});
  std::vector<std::vector<double>> chosen;
  for(auto candidate:candidates) {
    if(chosen.size()==4) break;
    if(candidate.position.size()!=size_t(dimensions) || !std::isfinite(candidate.objective)) continue;
    if(std::any_of(candidate.position.begin(),candidate.position.end(),[&](double v){return !std::isfinite(v)||v<low||v>high;})) continue;
    candidate.radius=std::clamp(candidate.radius,1e-4,.1);
    if(std::any_of(chosen.begin(),chosen.end(),[&](const auto& x){return distance(x,candidate.position)<candidate.radius;})) continue;
    chosen.push_back(candidate.position);
    auto match=std::find_if(records.begin(),records.end(),[&](const auto& e) {
      double tolerance=1e-6*std::max({1.,std::abs(e.objective),std::abs(candidate.objective)});
      return e.validated && distance(e.position,candidate.position)<=std::min(e.radius,candidate.radius) &&
        std::abs(e.objective-candidate.objective)<=std::max(tolerance,2*(e.uncertainty+candidate.uncertainty));
    });
    if(match!=records.end()) {
      if(match->last_round!=round) { ++match->visits;match->last_round=round;if(stalled) match->confidence=std::min(1.,match->confidence+.2); }
      if(better(candidate.objective,match->objective)) {
        const bool meaningful=std::abs(candidate.objective-match->objective)>2*(candidate.uncertainty+match->uncertainty);
        match->position=candidate.position;match->objective=candidate.objective;match->uncertainty=candidate.uncertainty;
        if(meaningful) match->confidence*=.5;
      }
      match->cost+=candidate.cost;match->refinement_cost+=candidate.refinement_cost;match->refined|=candidate.refined;
      if(candidate.geometry.kind!=Json::Null) match->geometry=std::move(candidate.geometry);
      continue;
    }
    candidate.id=next_id++;candidate.discovery_round=candidate.last_round=round;candidate.confidence=0;
    if(records.size()==capacity) {
      auto best=std::min_element(records.begin(),records.end(),[&](const auto& a,const auto& b){return better(a.objective,b.objective);});
      size_t victim=records.size();
      for(size_t i=0;i<records.size();++i) if(&records[i]!=&*best)
        if(victim==records.size() || records[i].confidence<records[victim].confidence ||
           (records[i].confidence==records[victim].confidence && records[i].last_round<records[victim].last_round)) victim=i;
      if(victim==records.size()) { if(!better(candidate.objective,best->objective)) continue;victim=0; }
      records.erase(records.begin()+victim);
    }
    records.push_back(std::move(candidate));
  }
}
double BasinArchive::penalty(const std::vector<double>& x,int ignored) const {
  double result=0;
  for(size_t i=0;i<records.size();++i) if(int(i)!=ignored && records[i].validated) {
    const auto& e=records[i];
    result=std::max(result,e.confidence*std::exp(-.5*std::pow(distance(x,e.position)/e.radius,2)));
  }
  return result;
}
void BasinArchive::place(float* output,Rng& rng,int refinement,bool avoidance) const {
  std::vector<double> best(dimensions),candidate(dimensions);double least=INFINITY;
  for(int attempt=0;attempt<64;++attempt) {
    for(int j=0;j<dimensions;++j) {
      candidate[j]=refinement>=0 && refinement<int(records.size()) ?
        records[refinement].position[j]+(high-low)*records[refinement].radius*normal(rng) : low+(high-low)*rng.uniform01();
      if(periodic) candidate[j]=low+std::fmod(std::fmod(candidate[j]-low,high-low)+high-low,high-low);
    }
    if(std::any_of(candidate.begin(),candidate.end(),[&](double x){return x<low||x>high||!std::isfinite(x);})) continue;
    const double value=avoidance?penalty(candidate,refinement):0;
    if(value<least) {least=value;best=candidate;}
    if(rng.uniform01()<.05+.95*(1-value)) {best=candidate;break;}
  }
  if(!std::isfinite(least)) for(int j=0;j<dimensions;++j) best[j]=low+(high-low)*rng.uniform01();
  for(int j=0;j<dimensions;++j) output[j]=float(best[j]);
}
void BasinArchive::set_periodic(bool value) {
  if(value==periodic) return;
  periodic=value;compatibility.object["periodic"]=boolean(value);
  for(auto& e:records) {e.confidence=0;e.geometry=Json{};}
}
Json BasinArchive::summary() const {
  Json list;list.kind=Json::Array;
  for(const auto& e:records) list.array.push_back(entry_json(e,false));
  return list;
}
std::string BasinArchive::export_json() const {
  Json root;root.kind=Json::Object;root.object["format"]=text_value("fractal-basin-archive");root.object["version"]=number(1);
  root.object["compatibility"]=compatibility;root.object["entries"].kind=Json::Array;
  for(const auto& e:records) root.object["entries"].array.push_back(entry_json(e,true));
  return stringify(root);
}
void BasinArchive::import_json(const std::string& text) {
  if(text.size()>8*1024*1024) throw std::invalid_argument("Basin archive exceeds 8 MiB");
  const size_t nodes=std::count(text.begin(),text.end(),',')+std::count(text.begin(),text.end(),'{')+std::count(text.begin(),text.end(),'[');
  if(nodes>64*(size_t(dimensions)*(dimensions<=64?dimensions:9)+dimensions+512))
    throw std::invalid_argument("Basin archive has too many data elements");
  auto root=JsonReader(text).read();
  if(root["format"].str()!="fractal-basin-archive" || root["version"].num()!=1 ||
     stringify(root["compatibility"])!=stringify(compatibility)) throw std::invalid_argument("Incompatible basin archive");
  const auto& input=root["entries"].items();
  if(input.size()>capacity) throw std::invalid_argument("Too many archived basins");
  std::vector<BasinEntry> imported;
  for(const auto& j:input) {
    BasinEntry e;e.id=imported.size()+1;
    for(const auto& value:j["position"].items()) e.position.push_back(bounded(value,0,low,high,"basin coordinate"));
    if(e.position.size()!=size_t(dimensions)) throw std::invalid_argument("Wrong basin dimensionality");
    if(j["objective"].kind!=Json::Number) throw std::invalid_argument("Missing basin objective");
    e.discovery_round=integer(j["discovery_round"],0,0,1000000000,"discovery round");
    e.last_round=UINT64_MAX;
    e.cost=uint64_t(bounded(j["cost"],0,0,1e12,"basin discovery cost"));
    e.refinement_cost=uint64_t(bounded(j["refinement_cost"],0,0,1e12,"refinement cost"));
    e.objective=bounded(j["objective"],0,-1e300,1e300,"basin objective");
    e.uncertainty=bounded(j["uncertainty"],0,0,1e300,"basin uncertainty");
    e.radius=bounded(j["radius"],1e-4,1e-4,.1,"basin radius");
    e.confidence=bounded(j["confidence"],0,0,1,"basin confidence");
    e.visits=integer(j["visits"],1,1,1000000000,"basin visits");
    e.validated=false;e.refined=j["refined"].flag(false);e.settings=j["settings"];
    if(j["geometry"].kind!=Json::Null) {
      validate_adaptive_geometry(j["geometry"],dimensions);
      for(const auto& v:j["geometry"]["anchor"].array) bounded(v,0,low,high,"geometry anchor");
      e.geometry=j["geometry"];
    }
    imported.push_back(std::move(e));
  }
  records=std::move(imported);next_id=records.size()+1;
}
void BasinArchive::synchronize_json(const std::string& text) {
  auto prepared=*this;prepared.import_json(text);
  const auto root=JsonReader(text).read();
  uint64_t maximum=0;
  for(size_t i=0;i<prepared.records.size();++i) {
    const auto& j=root["entries"].array[i];auto& e=prepared.records[i];
    e.id=uint64_t(bounded(j["id"],0,1,1e12,"basin id"));
    e.last_round=uint64_t(bounded(j["last_round"],0,0,1e12,"basin round"));
    e.validated=j["validated"].flag(false);maximum=std::max(maximum,e.id);
  }
  prepared.next_id=maximum+1;*this=std::move(prepared);
}
void BasinArchive::merge_event(const Json& event,uint64_t round) {
  auto evidence=*this;evidence.synchronize_json(stringify(event));
  complete_round(evidence.records,round,event["stalled"].flag(false));
}
uint64_t BasinArchive::validation_cost(bool stochastic) const {
  return std::count_if(records.begin(),records.end(),[](const auto& e){return !e.validated;})*(stochastic?3:1);
}
void BasinArchive::validate_imports(Benchmark& b,Rng& rng,uint64_t budget) {
  const auto required=validation_cost(b.stochastic);
  if(budget && (b.evaluations>budget || required>budget-b.evaluations)) throw std::runtime_error("Budget cannot validate all imported basins");
  for(auto& e:records) if(!e.validated) {
    const int count=b.stochastic?3:1;double mean=0,moment=0;
    for(int k=0;k<count;++k) {double value=b.evaluate_optimization(e.position.data(),&rng);double delta=value-mean;mean+=delta/(k+1);moment+=delta*(value-mean);}
    const double uncertainty=count>1?std::sqrt(moment/(count*(count-1))):0;
    if(std::abs(mean-e.objective)>2*(uncertainty+e.uncertainty)+1e-6*std::max(1.,std::abs(e.objective))) e.confidence=0;
    e.objective=mean;e.uncertainty=uncertainty;
    e.validated=std::isfinite(mean);e.cost+=count;
    if(!e.validated) e.confidence=0;
  }
}
}  // namespace fg::optimization
