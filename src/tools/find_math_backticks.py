#!/usr/bin/env python3
import re

with open('/home/guillem/FractalAI/clean_build/source/03_cloning.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

latex_commands = [
    'alpha', 'beta', 'gamma', 'delta', 'varepsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau',
    'upsilon', 'phi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon',
    'Phi', 'Psi', 'Omega',
    'approx', 'leq', 'geq', 'in', 'notin', 'cup', 'cap', 'times', 'div',
    'sum', 'prod', 'int', 'infty', 'partial', 'nabla'
]

count = 0
for line_num, line in enumerate(lines, 1):
    # Find backtick expressions
    backticks = re.findall(r'`([^`]+)`', line)
    for expr in backticks:
        # Check if it contains LaTeX commands
        if any(f'\\{cmd}' in expr for cmd in latex_commands):
            count += 1
            if count <= 50:
                print(f"{line_num}: `{expr}`")

print(f"\nTotal: {count} expressions")
