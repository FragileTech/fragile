#pragma once
#include <cstdint>
#include <stdexcept>
#ifdef __EMSCRIPTEN__
#include <malloc.h>
#include <emscripten/heap.h>
#endif

namespace fg::arcade_memory {
// Only Arcade configures this limit. Other users of the shared algorithms
// retain their existing allocation policy. All arithmetic must survive 4 GiB.
inline uint64_t limit = 0;
inline void require(uint64_t extra) {
#ifdef __EMSCRIPTEN__
  if (!limit) return;
  const auto info = mallinfo();
  const uint64_t end = *emscripten_get_sbrk_ptr();
  const uint64_t fixed = end >= info.arena ? end - info.arena : 0;
  const uint64_t used = fixed + info.uordblks;
  constexpr uint64_t reserve = 16ULL * 1024 * 1024;
  if (used + extra + reserve > limit)
    throw std::runtime_error("Main engine memory limit reached. Reduce walkers or retained Graph history, or increase the engine memory limit and reset.");
#else
  (void)extra;
#endif
}
}  // namespace fg::arcade_memory
