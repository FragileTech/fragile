# Fractal Populations

A population combines independent Wave instances on the same task. Each member owns
its RNG, parameters, backend resources, and movement/adaptation state. Only selected
walker snapshots cross member boundaries. `population.hpp` remains the row-storage
container; `populations.hpp` implements the multi-swarm controller.

Open **Optimization → Fractal Populations** (`web/optimization/populations.html`) for
the browser workspace. The default is four swarms, 256 walkers and five elites per
swarm, exchanging every step. Change the swarm count and per-member fields before
resetting. Empty fields inherit shared defaults; seeds derive from the experiment
seed and stable member IDs. Execution slots limit simultaneous steps independently
of the number of members. The browser uses one WASM worker per member and a separate
C++/WASM coordinator. Its one-GiB admission estimate includes 32 MiB per member plus
configured walker storage; it is an admission bound, not a measured browser heap cap.

## Configuration

```json
{
  "seed": 7,
  "exchange_every": 1,
  "global_elites": 20,
  "concurrency": 2,
  "max_evaluations": 100000,
  "defaults": {
    "algorithm": "wave",
    "benchmark": "rastrigin",
    "dimensions": 2,
    "walkers": 256,
    "elites": 5,
    "boundary": "periodic",
    "controller_enabled": true
  },
  "members": [
    {"id": "broad", "settings": {"perturbation_std": 0.4}},
    {"id": "local", "settings": {"perturbation_std": 0.02, "seed": 42}},
    {"id": "reward", "settings": {"reward_coef": 2}},
    {"id": "geometry", "settings": {"distance_coef": 2}}
  ]
}
```

The `members` list defines the count. Omit it to obtain the four-swarm preset.
`exchange_count` on an entry optionally overrides its initial elite count. Different
walker counts, elite counts, coefficients, perturbations, durations, and seeds are
supported. All members must have the same task, dimensions, bounds, objective
orientation, and periodic geometry. Unsupported algorithm families fail explicitly.
One member runs without imports. Membership, task, and shared-budget changes require
reset. Native `settings` requests and browser `updateMember` support safe live tuning;
resolved settings are included in status and recordings. The browser workspace exposes
live scale changes and individual restart requests.

`max_evaluations` is one shared budget, including initialization, steps, validation,
and restarts. Zero means unlimited (automatic restarts require a positive budget).
Complete rounds are admitted using conservative upper bounds; unused final budget
may be insufficient for another complete round. Budget exhaustion pauses execution.
A worker or step failure stops the experiment until reset; completed work is not
rolled back and no partially staged exchange is committed.

## Black-box C++ contract

`PopulationMember` exposes `advance`, `describe`, `export_walker`, `stage`, `commit`,
and `discard`. `describe` reports score, removal score, eligibility, and protected
elite slots. `WalkerPacket` owns its bytes, compatibility key, candidate identity,
and source provenance. Scores must have a common higher-is-better interpretation.
A member adapter must outlive its controller operation; recreate it after replacing
its underlying algorithm, as Optimization does after restarts.

```cpp
#include "fractal/populations.hpp"
#include "fractal_gas.hpp"

// a and b are initialized FractalGas instances with independent environments.
// The caller's key identifies the same ROM/task, state ABI, and reward semantics.
auto first = a.population_member("first", "my-task-state-v1");
auto second = b.population_member("second", "my-task-state-v1");
fg::fractal::PopulationController population(7, 1, 20);
fg::ThreadPool executor(2); // distinct from either backend's worker pool
population.advance({first.get(), second.get()}, &executor);
```

`PackedWave::population_member(id, count)` uses the Control scene fingerprint as its
compatibility key. Imported ancestry owns its original root snapshot. Control replay
and Arcade trajectory extraction resolve the correct root for each branch. Control
checkpoint version **4** stores imported roots and rejects older checkpoint versions.

Reference-backed environments must provide both owning save/load codecs to
`FractalGas::population_member`. LLM explicitly rejects raw snapshot exchange without
these codecs. The host must export/retain immutable token records and materialize or
remap their IDs in the receiving archive; model/task/scoring and embedding semantics
belong in the compatibility key. Copying an ID into an unrelated archive is invalid.
LLM's zero historical elite reinjection remains unchanged; use an explicit exchange
count to export eligible active frontier candidates. Native LLM tests use a shared
immutable record namespace; this release adds no LLM browser population controls.

## Exchange strategies and staging

`ExchangeStrategies` contains independent `exports`, `imports`, and `donors` callables.
Replace one without modifying Wave or the scheduling barrier. Defaults export local
protected elites (or the best eligible frontier when protection is disabled), mark
the worst unprotected non-export rows, and sample foreign entries without replacement.
Replacement uses the same retention ordering as shrinking: virtual reward when
available, otherwise cumulative reward, with nonfinite scores worst and earlier
indices retained on ties.

All members finish their step before exports are frozen. Donor selection uses its own
seeded RNG and stable member order, independent of completion order. Import/export
flags are boundary-local metadata in controller frames, not inheritable walker traits.
The latest export pool replaces the previous one. A recipient lacking enough valid
foreign exports skips the whole exchange and reports its requested shortfall.

Imports carry backend state, observations, actions/root actions, rewards, fitness,
durations, terminal metadata and remapped ancestry. They add no evaluations. All
allocation and validation happens in `stage`; `commit` is non-throwing. A load codec
must make any reference ownership acquired during staging safe to discard. RNGs,
configuration and adaptive movement models are never imported.

## Archives and APIs

The controller retains up to `global_elites` best exported candidates seen so far,
deduplicated by unchanged candidate identity. Stable ties prefer prior entries.
Historical elites are separate from the current export pool and are not import donors.

Optimization merges each member's completed basin-round evidence exactly once into
one archive (capacity 64), using deterministic member order and unique event identities.
Members receive that archive before subsequent rounds; existing restart, avoidance,
focused placement and geometry reuse rules then use shared evidence. Synchronization
within the active population preserves validation; ordinary imported basin files still
use the existing revalidation workflow.

`optimization::PopulationExperiment(config)` owns native sessions and an executor.
`step()` advances a complete round; `status()` reports resolved member settings,
evaluations, exchange provenance, global elites and basins. Its additive C ABI is:

- `fgp_create(config, remote)`: native owning mode (`0`) or worker coordinator (`1`).
- `fgp_request(handle, request)`: `config`, `status`, native `step`, `snapshot`
  (`member` index), and `settings` (`member`, `patch`).
- Remote lifecycle: `initialize` reports, `prepare` archive, worker synchronization,
  `admit` updated reports, worker steps, `finish` reports, stage/commit all member plans,
  then coordinator `commit`. `refresh` updates reports after live settings; `fail`
  stops the experiment. Reports are trusted messages from its own worker sessions.
- `fgo_exchange(session, request)`: `configure` (`id`, `count`), `capture`, `sync`
  (`archive`), `stage` (`imports`), `commit`, and `discard`.
- `fgp_destroy(handle)` releases the controller. Session and controller handles remain
  separate. JSON pointers are borrowed until the next call to the same API function.

Population recording format **1** stores resolved configuration, per-round member
snapshots, flags/provenance and archive summaries. It supports viewing and playback,
not resuming numerical execution. Single-swarm recordings and APIs remain separate.

## Validation

Native suites cover 1/2/4/7 members, heterogeneous settings and sizes, shared budgets,
exchange intervals, shortfalls, serial/parallel equivalence, basin deduplication,
full-state transfer, incompatible payloads, replay and checkpoint continuation, and
LLM reference ownership. WASM protocol tests cover frozen staging, out-of-order
reports and recordings; `npm run test:populations-browser` exercises actual concurrent
workers, per-swarm settings, exchange provenance, live updates and mobile layout.
