#include "optimization/c_api.h"

#include <map>

#include "optimization/engine.hpp"
#include "optimization/populations.hpp"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif
using namespace fg::optimization;
namespace {
std::map<uint32_t, std::unique_ptr<Session>> sessions;
uint32_t next_handle = 1;
struct ExchangeBinding {
  std::string id;
  int count=-1;
  std::unique_ptr<fg::fractal::PopulationMember> pending;
};
std::map<uint32_t,ExchangeBinding> exchanges;
std::map<uint32_t,std::unique_ptr<PopulationExperiment>> populations;

thread_local std::string error;
Session& get(uint32_t h) {
  auto it = sessions.find(h);
  if (it == sessions.end())
    throw std::invalid_argument("Invalid optimization session");
  return *it->second;
}
template <class T, class F>
T guard(T fail, F fn) {
  try {
    error.clear();
    return fn();
  } catch (const std::exception& e) {
    error = e.what();
    return fail;
  } catch (...) {
    error = "Unknown optimization engine error";
    return fail;
  }
}
}  // namespace
extern "C" {
EXPORT uint32_t fgp_create(const char* config,int remote) {
  return guard<uint32_t>(0,[&]{
    if(!config)throw std::invalid_argument("Missing population configuration");
    auto population=std::make_unique<PopulationExperiment>(JsonReader(std::string(config)).read(),remote!=0);
    auto h=next_handle++;populations.emplace(h,std::move(population));return h;
  });
}
EXPORT int fgp_destroy(uint32_t h) {
  return guard<int>(-1,[&]{if(!populations.erase(h))throw std::invalid_argument("Invalid population handle");return 0;});
}
EXPORT const char* fgp_request(uint32_t h,const char* request) {
  static std::string output;
  return guard<const char*>(nullptr,[&]{
    auto it=populations.find(h);if(it==populations.end() || !request)throw std::invalid_argument("Invalid population request");
    output=stringify(it->second->request(JsonReader(std::string(request)).read()));return output.c_str();
  });
}
EXPORT const char* fgo_exchange(uint32_t h,const char* request) {
  static std::string output;
  return guard<const char*>(nullptr,[&]{
    if(!request)throw std::invalid_argument("Missing exchange request");
    auto& session=get(h);auto req=JsonReader(std::string(request)).read();auto op=req["op"].str();
    Json result;result.kind=Json::Object;
    if(op=="configure") {
      auto id=req["id"].str();int count=integer(req["count"],session.settings.elites,0,100000,"exchange count");
      auto member=session.exchange_member(id,count);
      fg::fractal::PopulationController c;c.capture(*member);
      exchanges[h]={id,count,nullptr};
    } else {
      auto found=exchanges.find(h);if(found==exchanges.end())throw std::invalid_argument("Configure exchange first");
      auto& binding=found->second;
      if(op=="capture") {
        auto member=session.exchange_member(binding.id,binding.count);fg::fractal::PopulationController c;
        result.object["frame"]=exchange_frame_json(c.capture(*member));
        result.object["id"].kind=Json::String;result.object["id"].string=binding.id;
        result.object["evaluations"]=number(session.benchmark.evaluations);
        result.object["next_evaluations"]=number(session.next_evaluations());result.object["settings"]=session.settings.json;
        result.object["events"]=session.take_basin_events();
      } else if(op=="stage") {
        auto member=session.exchange_member(binding.id,binding.count);
        member->stage(exchange_imports_read(req["imports"]));binding.pending=std::move(member);
      } else if(op=="commit") {
        if(!binding.pending)throw std::logic_error("No staged walker imports");
        binding.pending->commit();binding.pending.reset();session.refresh_exchange();
      } else if(op=="discard") binding.pending.reset();
      else if(op=="sync") session.synchronize_basins(req["archive"].str());
      else throw std::invalid_argument("Unknown exchange operation");
    }
    output=stringify(result);return output.c_str();
  });
}
EXPORT const char* fgo_precision() {
  static const std::string precision =
      "{\"swarm_coordinates_bits\":" +
      std::to_string(8 * sizeof(decltype(Population::x)::value_type)) +
      ",\"swarm_fitness_bits\":" +
      std::to_string(8 * sizeof(decltype(Population::fitness)::value_type)) +
      ",\"objective_bits\":" +
      std::to_string(8 * sizeof(decltype(Population::objective)::value_type)) +
      ",\"cma_coordinates_bits\":" + std::to_string(8 * sizeof(double)) + "}";
  return precision.c_str();
}
EXPORT const char* fgo_catalog() {
  static std::string catalog;
  catalog = discovery_json();
  return catalog.c_str();
}
EXPORT const char* fgo_error() { return error.c_str(); }
EXPORT uint32_t fgo_create(const char* config) {
  return guard<uint32_t>(0, [&] {
    if (!config) throw std::invalid_argument("Missing configuration");
    if (sessions.size() >= 16)
      throw std::invalid_argument("Too many optimization sessions");
    auto session =
        std::make_unique<Session>(JsonReader(std::string(config)).read());
    uint32_t h = next_handle++;
    sessions.emplace(h, std::move(session));
    return h;
  });
}
EXPORT int fgo_destroy(uint32_t h) {
  return guard<int>(-1, [&] {
    get(h);
    exchanges.erase(h);
    sessions.erase(h);
    return 0;
  });
}
EXPORT const char* fgo_config(uint32_t h) {
  return guard<const char*>(nullptr,
                            [&] { return get(h).config_json.c_str(); });
}
EXPORT const char* fgo_status(uint32_t h) {
  static std::string status;
  return guard<const char*>(nullptr, [&] { status = get(h).status_json(); return status.c_str(); });
}
EXPORT int fgo_geometry_diagnostics(uint32_t h, int enabled) {
  return guard<int>(-1,[&] {get(h).set_geometry_diagnostics(enabled!=0);return 0;});
}
EXPORT int fgo_update_settings(uint32_t h, const char* patch) {
  return guard<int>(-1, [&] {
    if (!patch) throw std::invalid_argument("Missing settings patch");
    if(exchanges.count(h) && exchanges.at(h).pending)throw std::logic_error("Commit or discard imports before changing settings");
    get(h).update_settings(JsonReader(std::string(patch)).read());
    return 0;
  });
}
EXPORT const char* fgo_preview_settings(uint32_t h, const char* patch) {
  static std::string result;
  return guard<const char*>(nullptr, [&] {
    if (!patch) throw std::invalid_argument("Missing settings patch");
    result = stringify(get(h).preview_settings(JsonReader(std::string(patch)).read()));
    return result.c_str();
  });
}
EXPORT int fgo_set_population(uint32_t h, int count, const char* policy) {
  return guard<int>(-1, [&] {
    if (!policy) throw std::invalid_argument("Missing removal policy");
    if(exchanges.count(h) && exchanges.at(h).pending)throw std::logic_error("Commit or discard imports before resizing");
    get(h).set_population(count, policy);
    return 0;
  });
}
EXPORT const char* fgo_export_basins(uint32_t h) {
  static std::string result;
  return guard<const char*>(nullptr,[&]{result=get(h).export_basins();return result.c_str();});
}
EXPORT int fgo_import_basins(uint32_t h,const char* data) {
  return guard<int>(-1,[&]{if(!data) throw std::invalid_argument("Missing basin archive");get(h).import_basins(data);return 0;});
}
EXPORT int fgo_step(uint32_t h) {
  return guard<int>(-1, [&] {
    if(exchanges.count(h) && exchanges.at(h).pending)throw std::logic_error("Commit or discard imports before stepping");
    get(h).step();
    return 0;
  });
}
EXPORT const double* fgo_snapshot(uint32_t h) {
  return guard<const double*>(nullptr, [&] { return get(h).snapshot.data(); });
}
EXPORT int fgo_snapshot_size(uint32_t h) {
  return guard<int>(-1, [&] { return int(get(h).snapshot.size()); });
}
EXPORT int fgo_sample(uint32_t h, const float* x, int n, double* out) {
  return guard<int>(-1, [&] {
    auto& b = get(h).benchmark;
    if (n < 0 || n > 1000000 || !x || !out)
      throw std::invalid_argument("Invalid sampling buffers");
    for (int i = 0; i < n; ++i) out[i] = b.evaluate(x + size_t(i) * b.d);
    return 0;
  });
}
EXPORT int fgo_sample64(uint32_t h, const double* x, int n, double* out) {
  return guard<int>(-1, [&] {
    auto& b = get(h).benchmark;
    if (n < 0 || n > 1000000 || !x || !out)
      throw std::invalid_argument("Invalid sampling buffers");
    for (int i = 0; i < n; ++i) out[i] = b.evaluate(x + size_t(i) * b.d);
    return 0;
  });
}
}
