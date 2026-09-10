"""Independent SciPy verification of the browser Davidson optimizer.

Run with: uv run python fractal-gas-web/tests/llm/ranking-reference.py
"""

import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
script = """
import {fitDavidson} from './web/llm/ranking-math.js';
const observations=[];
for(let i=0;i<5;i++)for(let j=0;j<5;j++)if(i!==j)
 for(let k=0;k<9;k++)observations.push({i,j,y:k<2?2:k<5?(i>j?0:1):0,weight:.5});
console.log(JSON.stringify({observations,fit:fitDavidson(['a','b','c','d','e'],observations,{draws:0})}));
"""
data = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", script], cwd=ROOT))
obs = data["observations"]


def loss(theta):
    value = np.sum((theta / np.array([2, 2, 2, 2, 2, 1.5, 1])) ** 2) / 2
    for row in obs:
        a, b = theta[row["i"]], theta[row["j"]]
        logits = np.array([a + theta[6] / 2, b - theta[6] / 2, theta[5] + (a + b) / 2])
        value += row["weight"] * (logsumexp(logits) - logits[row["y"]])
    return value


fit = minimize(loss, np.zeros(7), method="BFGS", options={"gtol": 1e-7})
actual = np.array(data["fit"]["theta"])
np.testing.assert_allclose(actual, fit.x, atol=2e-5, rtol=0)
assert abs(loss(actual) - fit.fun) < 1e-8
print(f"SciPy reference agrees: maximum parameter difference {np.max(np.abs(actual - fit.x)):.3g}")
