"""Script to fix test_adaptive_kinetics.py for new KineticOperator API."""

# Read the test file
with open("tests/test_adaptive_kinetics.py", encoding="utf-8") as f:
    content = f.read()

# Pattern 1: Remove fitness_operator from KineticOperator initialization
content = content.replace(
    "params, potential=potential, fitness_operator=fitness_operator, device=device, dtype=dtype",
    "params, potential=potential, device=device, dtype=dtype",
)
content = content.replace(
    "params, potential=None, fitness_operator=fitness_operator, device=device, dtype=dtype",
    "params, potential=None, device=device, dtype=dtype",
)
content = content.replace(
    "params, potential=potential, fitness_operator=None, device=device, dtype=dtype",
    "params, potential=potential, device=device, dtype=dtype",
)

# Pattern 2: Update test signatures to include companions
content = content.replace(
    "def test_fitness_force_only(device, dtype, fitness_operator, simple_state, simple_rewards_alive):",
    "def test_fitness_force_only(device, dtype, fitness_operator, simple_state, simple_rewards_alive):",
)
content = content.replace(
    "    rewards, alive = simple_rewards_alive",
    "    rewards, alive, companions = simple_rewards_alive",
)

# Pattern 3: Update apply() calls - need to compute gradients/Hessians first
# This is complex - will do manually for each test

print("Basic replacements done. Manual fixes needed for apply() calls.")
print("Writing intermediate file...")

with open("tests/test_adaptive_kinetics.py.tmp", "w", encoding="utf-8") as f:
    f.write(content)

print("Done! Check test_adaptive_kinetics.py.tmp")
