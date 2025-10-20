"""Import test directly from pytest to match environment exactly."""

import sys
sys.path.insert(0, "tests")

import torch
from test_operators import TestInelasticCollisionVelocity

# Create instance and run test
test_instance = TestInelasticCollisionVelocity()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    test_instance.test_non_cloners_unchanged(device)
    print("✓ Test passed")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
