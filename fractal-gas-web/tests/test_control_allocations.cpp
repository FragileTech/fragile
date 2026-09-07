// Instrument ordinary allocation after the persistent pool and scratch warm up.
#include <atomic>
#include <cstdlib>
#include <iostream>

#include "control/physics.hpp"
static std::atomic<size_t> allocations{0};
void* operator new(size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  ++allocations;
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void* operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }
using namespace fg::control;
int main() {
  auto scene = Scene::compile(
      R"({"rewards":{"distance_squared":1},"bodies":[{"position":[10,10],"controlled":true},{"position":[13,10],"cargo":true}],"tethers":[{"a":0,"b":1}],"pickups":[{"position":[10,10]}]})");
  for (int threads : {1, 4}) {
    size_t small = 0;
    for (int worlds : {16, 256}) {
      Physics physics(scene, threads);
      StateBatch a(worlds, *scene), b(worlds, *scene);
      a.reset(*scene, 7);
      std::vector<float> actions(worlds * 2, .2f);
      std::vector<int32_t> frames(worlds, 6);
      std::vector<StepResult> result(worlds);
      physics.step(a, nullptr, actions.data(), frames.data(), b, result.data());
      allocations = 0;
      physics.step(a, nullptr, actions.data(), frames.data(), b, result.data());
      size_t count = allocations.load();
      std::cout << worlds << " worlds, " << threads << " threads: " << count
                << " allocations per batch\n";
      if (worlds == 16)
        small = count;
      else if (count != small || count > size_t(threads + 2))
        return 1;
    }
  }
}
