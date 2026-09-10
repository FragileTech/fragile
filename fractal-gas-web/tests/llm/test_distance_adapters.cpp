#include "test_framework.hpp"
#include "control/c_api.h"
#include "control/json.hpp"
#include "optimization/engine.hpp"
#include <cstring>
using namespace fg;
TEST_CASE(control_metric_configuration_and_checkpoint_roundtrip) {
  const char* scene=R"({"size":[100,100],"bodies":[{"controlled":true,"position":[50,50],"velocity":[2,1]}]})";
  void* engine=fgc_create(scene,1,1);CHECK(engine!=nullptr);if(!engine)return;
  for(const char* algorithm:{"fmc","wave-jump"}) {
    std::string settings=std::string(R"({"walkers":8,"horizon":4,"frames":2,"recording":2,"distance_metric":"cosine","algorithm":")")+algorithm+"\"}";
    CHECK(fgc_plan_begin(engine,settings.c_str(),7)==0);
    CHECK(fgc_plan_advance(engine)>=0);
    size_t n=fgc_checkpoint_size(engine);std::vector<uint8_t> checkpoint(n);
    CHECK(fgc_checkpoint_write(engine,checkpoint.data(),checkpoint.size())==0);
    const std::string bytes(checkpoint.begin(),checkpoint.end());CHECK(bytes.find("cosine")!=std::string::npos);
    CHECK(fgc_plan_advance(engine)>=0);
    const auto words = size_t(8 * fgc_info(engine,3));
    const auto* population = fgc_wave_states(engine);
    std::vector<float> expected(population, population + words);
    const std::string expected_plan = fgc_plan_result(engine);
    CHECK(fgc_checkpoint_restore(engine,checkpoint.data(),checkpoint.size())==0);
    CHECK(fgc_plan_advance(engine)>=0);
    population = fgc_wave_states(engine);
    CHECK(std::memcmp(expected.data(), population, words * sizeof(float)) == 0);
    CHECK(expected_plan == fgc_plan_result(engine));
  }
  CHECK(fgc_plan_begin(engine,R"({"distance_metric":"bogus"})",7)<0);
  CHECK(fgc_plan_begin(engine,R"({"walkers":8,"horizon":2})",7)==0);
  fgc_destroy(engine);
}
TEST_CASE(optimization_metric_settings_propagate_to_wave_graph_and_planners) {
  for(const char* algorithm:{"wave","graph","fmc","wave_jump"}) {
    std::string text=std::string(R"({"benchmark":"sphere","dimensions":2,"walkers":8,"max_walkers":32,"algorithm":")")+algorithm+R"(","distance_metric":"cosine"})";
    optimization::Session session(control::JsonReader(text).read());
    CHECK(session.settings.distance_metric==DistanceMetric::Cosine);
    CHECK(session.config_json.find("cosine")!=std::string::npos);
    session.algorithm->step();
  }
  std::string empty="{}";optimization::Settings settings(control::JsonReader(empty).read());
  CHECK(settings.distance_metric==DistanceMetric::L2);
}
