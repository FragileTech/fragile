"""Script to fix test_operators.py for new compute_fitness API."""

# Read the test file
with open("tests/test_operators.py", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Update TestComputeFitness fixture to compute companions
content = content.replace(
    """    @pytest.fixture
    def simple_swarm(self, device):
        \"\"\"Create a simple test swarm.\"\"\"
        N = 20
        d = 3
        positions = torch.randn(N, d, device=device)
        velocities = torch.randn(N, d, device=device)
        rewards = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        companion_selection = CompanionSelection(method="uniform")

        return positions, velocities, rewards, alive, companion_selection""",
    """    @pytest.fixture
    def simple_swarm(self, device):
        \"\"\"Create a simple test swarm.\"\"\"
        N = 20
        d = 3
        positions = torch.randn(N, d, device=device)
        velocities = torch.randn(N, d, device=device)
        rewards = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        companion_selection = CompanionSelection(method="uniform")

        # Compute companions using the selection strategy
        companions = companion_selection(positions, velocities, alive)

        return positions, velocities, rewards, alive, companions""",
)

# Fix 2: Update test method signatures in TestComputeFitness
content = content.replace(
    """        positions, velocities, rewards, alive, companion_selection = simple_swarm""",
    """        positions, velocities, rewards, alive, companions = simple_swarm""",
)

# Fix 3: Update compute_fitness calls to use companions tensor
content = content.replace(
    """        fitness, distances, companions = compute_fitness(
            positions, velocities, rewards, alive, companion_selection
        )""",
    """        fitness, distances, companions_out = compute_fitness(
            positions, velocities, rewards, alive, companions
        )""",
)

content = content.replace(
    """        fitness, _, _ = compute_fitness(
            positions,
            velocities,
            rewards,
            alive,
            companion_selection,""",
    """        fitness, _, _ = compute_fitness(
            positions,
            velocities,
            rewards,
            alive,
            companions,""",
)

content = content.replace(
    """        fitness, _, _ = compute_fitness(positions, velocities, rewards, alive, companion_selection)""",
    """        fitness, _, _ = compute_fitness(positions, velocities, rewards, alive, companions)""",
)

content = content.replace(
    """        _fitness1, _distances1, companions = compute_fitness(
            positions_same, velocities, rewards, alive, companion_selection, lambda_alg=0.0
        )

        # Force same companion selection for comparison
        pos_diff = positions_same - positions_same[companions]
        vel_diff = velocities_diff - velocities_diff[companions]""",
    """        _fitness1, _distances1, companions_out = compute_fitness(
            positions_same, velocities, rewards, alive, companions, lambda_alg=0.0
        )

        # Force same companion selection for comparison
        pos_diff = positions_same - positions_same[companions_out]
        vel_diff = velocities_diff - velocities_diff[companions_out]""",
)

content = content.replace(
    """        fitness, _, _ = compute_fitness(
            positions, velocities, rewards, alive, companion_selection, alpha=alpha, beta=beta
        )""",
    """        fitness, _, _ = compute_fitness(
            positions, velocities, rewards, alive, companions, alpha=alpha, beta=beta
        )""",
)

# Fix 4: Update TestCloneWalkers fixture
content = content.replace(
    """    @pytest.fixture
    def simple_swarm(self, device):
        \"\"\"Create a simple test swarm.\"\"\"
        N = 20
        d = 3
        positions = torch.randn(N, d, device=device)
        velocities = torch.randn(N, d, device=device)
        rewards = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        companion_selection = CompanionSelection(method="uniform")

        # Compute fitness
        fitness, _distances, companions = compute_fitness(
            positions, velocities, rewards, alive, companion_selection
        )

        return positions, velocities, fitness, companions, alive""",
    """    @pytest.fixture
    def simple_swarm(self, device):
        \"\"\"Create a simple test swarm.\"\"\"
        N = 20
        d = 3
        positions = torch.randn(N, d, device=device)
        velocities = torch.randn(N, d, device=device)
        rewards = torch.randn(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        companion_selection = CompanionSelection(method="uniform")

        # Select companions using the selection strategy
        companions = companion_selection(positions, velocities, alive)

        # Compute fitness with the selected companions
        fitness, _distances, _companions_out = compute_fitness(
            positions, velocities, rewards, alive, companions
        )

        return positions, velocities, fitness, companions, alive""",
)

print("Transformations applied!")
print("Writing to tests/test_operators.py...")

with open("tests/test_operators.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done!")
