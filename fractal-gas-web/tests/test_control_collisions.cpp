#include <cstring>
#include <sstream>
#include "control/physics.hpp"
#include "test_framework.hpp"
using namespace fg::control;
namespace {
// Independent uncached SAT reference: derive every axis from world vertices.
float reference_gap(const geometry::Shape& a, const geometry::Shape& b) {
  float gap = -1e30f;
  auto axis = [&](Vec2 n) {
    if (length2(n) < 1e-16f) return;
    n = normalized(n);
    auto pa = geometry::project(a, n), pb = geometry::project(b, n);
    gap = std::max({gap, pb.first - pa.second, pa.first - pb.second});
  };
  if (!a.n && !b.n) axis(b.center - a.center);
  for (int i = 0; i < a.n; ++i) axis(perp(a.vertices[(i + 1) % a.n] - a.vertices[i]));
  for (int i = 0; i < b.n; ++i) axis(perp(b.vertices[(i + 1) % b.n] - b.vertices[i]));
  if (a.n == 2) axis(a.vertices[1] - a.vertices[0]);
  if (b.n == 2) axis(b.vertices[1] - b.vertices[0]);
  if (!a.n && b.n) {
    int k = 0;
    for (int i = 1; i < b.n; ++i)
      if (length2(b.vertices[i] - a.center) < length2(b.vertices[k] - a.center)) k = i;
    axis(b.vertices[k] - a.center);
  }
  if (!b.n && a.n) {
    int k = 0;
    for (int i = 1; i < a.n; ++i)
      if (length2(a.vertices[i] - b.center) < length2(a.vertices[k] - b.center)) k = i;
    axis(b.center - a.vertices[k]);
  }
  return gap;
}
std::shared_ptr<const Scene> floor_scene(bool polygon, float friction = .5f) {
  std::ostringstream s;
  s << R"({"size":[100,100],"environment":{"flight":true},"physics":{"substeps":4},"bodies":[{"position":[40,0.5],"mass":1,"radius":0.5,"drag":0,"angular_drag":0,"friction":)" << friction;
  if (polygon) s << R"(,"vertices":[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]])";
  s << R"(,"controlled":true,"thrust":0}]})";
  return Scene::compile(s.str());
}
}
TEST_CASE(control_cached_geometry_and_conservative_capsule) {
  BodyDef body;
  body.radius = 1.2f;
  body.vertices = {{-1,-.4f},{1,-.4f},{.7f,.6f},{-.7f,.6f}};
  Edge edge{{-3,0},{3,1}};
  auto wall = geometry::shape(edge);
  geometry::ShapeCache cache;
  uint64_t rng = 517;
  for (int i = 0; i < 2000; ++i) {
    Vec2 start{random01(rng)*10-5,random01(rng)*10-5};
    Vec2 end{random01(rng)*10-5,random01(rng)*10-5};
    float angle = random01(rng)*6, spin = random01(rng)*12-6;
    const auto& cached = cache.get(body, start, angle);
    auto fresh = geometry::shape(body, start, angle);
    Vec2 n;
    CHECK_CLOSE(geometry::separation(cached, wall, n), reference_gap(fresh, wall), 1e-6);
    CHECK(&cache.get(body, start, angle) == &cached);
    bool near = geometry::swept_near_edge(start, end, edge, body.radius + .002f);
    for (int j = 0; j <= 16; ++j) {
      float t = j / 16.f;
      auto moving = geometry::shape(body, start + (end-start)*t, angle+spin*t);
      if (reference_gap(moving, wall) <= .0001f) CHECK(near);
    }
    cache.valid = false;
    CHECK_CLOSE(geometry::separation(cache.get(body, start, angle), wall, n), reference_gap(fresh, wall), 1e-6);
  }
  Edge horizontal{{0,0},{10,0}};
  CHECK(geometry::swept_near_edge({-1,1},{11,1},horizontal,1.002f));
  CHECK(geometry::swept_near_edge({-1,-1},{1,1},horizontal,.01f));
  CHECK(!geometry::swept_near_edge({-5,3},{15,3},horizontal,1.002f));
  BodyDef circle;
  Vec2 n;
  for (float x : {-4.f,0.f,4.f}) {
    auto c = geometry::shape(circle,{x,.2f},2.f);
    CHECK_CLOSE(geometry::separation(c,wall,n),reference_gap(c,wall),1e-6);
  }
}
TEST_CASE(control_resting_wall_contact_stays_stable) {
  for (bool polygon : {false,true}) {
    auto s = floor_scene(polygon);
    Physics physics(s);
    StateBatch a(1,*s), b(1,*s);
    a.reset(*s,7);
    float actions[2] = {}; int32_t frames = 1; StepResult result;
    for (int i = 0; i < 600; ++i) {
      physics.step(a,nullptr,actions,&frames,b,&result);
      CHECK(position(b.row(0),s->layout,0).y >= .499f);
      CHECK(position(b.row(0),s->layout,0).y < .51f);
      CHECK(std::abs(velocity(b.row(0),s->layout,0).y) < .1f);
      CHECK(result.collisions <= 4); // One floor contact per substep.
      std::swap(a,b);
    }
    velocity(a.row(0),s->layout,0,{0,5});
    frames = 6;
    physics.step(a,nullptr,actions,&frames,b,&result);
    CHECK(position(b.row(0),s->layout,0).y > .9f);
  }
}
TEST_CASE(control_wall_proximity_is_not_lethal_contact) {
  auto s = Scene::compile(R"({"size":[20,20],"physics":{"lethal_walls":true},"bodies":[{"position":[10,0.501],"radius":0.5,"controlled":true}]})");
  Physics p(s); StateBatch a(1,*s),b(1,*s); a.reset(*s,7);
  float action[2] = {}; int32_t frames=1; StepResult result;
  p.step(a,nullptr,action,&frames,b,&result);
  CHECK(result.collisions==0); CHECK(!result.dead);
}
TEST_CASE(control_sliding_wall_friction_and_restitution) {
  for (float friction : {0.f,.8f}) {
    auto s = floor_scene(true,friction);
    Physics p(s);StateBatch a(1,*s),b(1,*s);a.reset(*s,7);
    velocity(a.row(0),s->layout,0,{2,0});
    float action[2]={};int32_t frames=60;StepResult result;
    p.step(a,nullptr,action,&frames,b,&result);
    CHECK(position(b.row(0),s->layout,0).y > .49f);
    if (friction==0) CHECK_CLOSE(velocity(b.row(0),s->layout,0).x,2,1e-4);
    else CHECK(std::abs(velocity(b.row(0),s->layout,0).x)<1);
  }
  auto s=Scene::compile(R"({"size":[100,100],"physics":{"substeps":1},"bodies":[{"position":[50,2],"velocity":[0,-100],"radius":0.5,"restitution":1,"friction":0,"drag":0,"controlled":true}]})");
  Physics p(s);StateBatch a(1,*s),b(1,*s);a.reset(*s,7);
  float action[2]={};int32_t frames=1;StepResult result;
  p.step(a,nullptr,action,&frames,b,&result);
  CHECK_CLOSE(velocity(b.row(0),s->layout,0).y,100,1e-4);
  CHECK(result.collisions==1);
}
TEST_CASE(control_contact_cache_batch_restore_and_threads) {
  auto s=floor_scene(true);Physics one(s,1),many(s,4);
  StateBatch root(8,*s),a(8,*s),b(8,*s),tmp(8,*s);root.reset(*s,7);
  std::vector<float> actions(16,0);std::vector<int32_t> frames(8,12);
  std::vector<StepResult> results(8);
  one.step(root,nullptr,actions.data(),frames.data(),a,results.data());
  many.step(root,nullptr,actions.data(),frames.data(),b,results.data());
  CHECK(std::memcmp(a.row(0),b.row(0),a.bytes())==0);
  std::vector<uint8_t> snapshot(root.serialized_size());root.serialize(snapshot.data(),snapshot.size());
  root.deserialize(snapshot.data(),snapshot.size());
  std::fill(frames.begin(),frames.end(),1);
  for(int i=0;i<12;++i){many.step(root,nullptr,actions.data(),frames.data(),tmp,results.data());std::swap(root,tmp);}
  CHECK(std::memcmp(a.row(0),root.row(0),a.bytes())==0);
}
TEST_CASE(control_slopes_corners_and_rotating_wall_contacts) {
  const char* scenes[] = {
    R"({"size":[100,100],"environment":{"flight":true},"boundary":[[0,0],[100,25],[100,100],[0,100]],"bodies":[{"position":[40,10.516],"angle":0.24497866,"vertices":[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]],"friction":0.8,"controlled":true}]})",
    R"({"size":[100,100],"environment":{"flight":true},"bodies":[{"position":[0.5,0.5],"velocity":[-0.1,0],"radius":0.5,"controlled":true}]})",
    R"({"size":[100,100],"environment":{"flight":true},"bodies":[{"position":[40,2],"omega":30,"vertices":[[-0.8,-0.2],[0.8,-0.2],[0.8,0.2],[-0.8,0.2]],"controlled":true}]})"
  };
  for (const auto* json : scenes) {
    auto s=Scene::compile(json);Physics p(s);StateBatch a(1,*s),b(1,*s);a.reset(*s,7);
    float actions[2]={};int32_t frames=1;StepResult result;
    for(int i=0;i<240;++i){
      p.step(a,nullptr,actions,&frames,b,&result);
      auto hull=geometry::shape(s->bodies[0],position(b.row(0),s->layout,0),angle(b.row(0),s->layout,0));
      for(const auto& edge:s->edges){
        Vec2 normal;float gap=geometry::separation(hull,geometry::shape(edge),normal);
        CHECK(gap > -.01f);
      }
      CHECK(s->inside(hull.center));
      std::swap(a,b);
    }
  }
}
TEST_CASE(control_resting_rock_releases_under_tether_load) {
  auto s=Scene::compile(R"({"size":[100,100],"environment":{"flight":true},"bodies":[{"position":[50,10],"velocity":[0,5],"controlled":true,"mass":10},{"position":[50,0.5],"radius":0.5,"mass":0.1}],"tethers":[{"a":0,"b":1,"rest_length":5,"stiffness":20,"damping":5,"break_force":100000}]})");
  Physics p(s);StateBatch a(1,*s),b(1,*s);a.reset(*s,7);
  float action[2]={};int32_t frames=12;StepResult result;
  p.step(a,nullptr,action,&frames,b,&result);
  CHECK(position(b.row(0),s->layout,1).y > 1);
  CHECK(word(b.row(0),s->layout.joints)==2);
}
