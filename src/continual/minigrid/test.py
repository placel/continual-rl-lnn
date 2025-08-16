from pathlib import Path
import os

import numpy as np

PWD = Path(__file__).resolve().parent

# # weighting = [0.15, 0.25, 0.6]
# # x = np.array([0.0, 0.10, 0.95])
perf_matrix = np.array([[0.96, 0.0, 0.0],
         [0.0, 0.96, 0.0],
         [0.0, 0.0, 0.96]])

# np.save(f'{PWD}/save.npy', x)
perf_matrix_path = PWD / 'perf_matrix.npy'
# perf_matrix_path = model_dir / 'perf_matrix.npy'


weights = np.array([0.6, 0.25, 0.15])
# Iterate over each task in the sequence, and multiply results by the weights for that task.
# np.roll(weights, i) will shift weights to the right i number of times
# [0.6, 0.25, 0.15] will transform to [0.15, 0.6, 0.25] etc. 
# This will focus on the recently trained task to promote learning of all tasks
# While also gaining from potential forward or backward transfer
print(perf_matrix)
print()
for i in range(len([0, 0, 0]) - 1): 
    perf_matrix[i] *= np.roll(weights, i)

print(perf_matrix)