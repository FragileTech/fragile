"""Refactor test_momentum_conservation.py to use new API."""

import re


with open("tests/test_momentum_conservation.py", encoding="utf-8") as f:
    content = f.read()

# Remove CloningParams and CloningOperator creation patterns
# Pattern: params = CloningParams(...) followed by op = CloningOperator(...)
pattern1 = r'params = CloningParams\([^)]+\)\s+op = CloningOperator\(params, torch\.device\("cpu"\), torch\.float64\)'


def replacement1(match):
    # Extract alpha_restitution from the match
    alpha_match = re.search(r"alpha_restitution=([\d.]+)", match.group())
    if alpha_match:
        alpha = alpha_match.group(1)
        return f"alpha_restitution = {alpha}"
    return "alpha_restitution = 0.5  # default"


content = re.sub(pattern1, replacement1, content)

# Replace op._inelastic_collision_velocity(state, companions) with
# inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)
content = re.sub(
    r"op\._inelastic_collision_velocity\(state, companions\)",
    "inelastic_collision_velocity(v, companions, will_clone, alpha_restitution)",
    content,
)

# Remove state = SwarmState(x, v) lines
content = re.sub(r"\s+state = SwarmState\(x, v\)\n", "\n", content)


# Add will_clone = torch.ones(...) after companions = torch.tensor(...)
def add_will_clone(match):
    companions_line = match.group()
    # Extract N from the tensor
    n_match = re.search(r"\[([^\]]+)\]", companions_line)
    if n_match:
        values = n_match.group(1).split(",")
        N = len(values)
        return f"{companions_line}\n        will_clone = torch.ones({N}, dtype=torch.bool)  # All walkers clone"
    return companions_line


content = re.sub(r"companions = torch\.tensor\(\[[^\]]+\]\)", add_will_clone, content)

# Remove references to SwarmState in imports - already removed

# Remove unnecessary x = torch.randn(...) lines where x is not used
# This is trickier, so let's skip for now

with open("tests/test_momentum_conservation.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✓ Refactored test_momentum_conservation.py")
