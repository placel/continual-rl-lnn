from pathlib import Path
import os

import numpy as np

PWD = Path(__file__).resolve().parent

envs = ['Empty', 'Key', 'Unlock']

print(envs)

envs *= 3
print(envs)

print(0 // 3)
print(1 // 3)
print(2 // 3)
print(3 // 3)
print(4 // 3)
print(5 // 3)
print(6 // 3)