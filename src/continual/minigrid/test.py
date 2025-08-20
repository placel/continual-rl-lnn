from pathlib import Path
import os

import numpy as np

PWD = Path(__file__).resolve().parent

import numpy as np

perf_matrix = np.array([[0.96, 0.00, 0.00],
               [0.95, 0.96, 0.00],
               [0.93, 0.00, 0.93]])

M = perf_matrix.copy()                 # shape (T, T)
T = M.shape[0]
final = M[T-1]                         # R_i^(T)

print('FINAL')
print(final)

# Per-task forgetting F_i = max_{k<T} R_i^(k) - R_i^(T)
F = np.max(M[:T-1, :], axis=0) - final
FGT = np.mean(F)                       # overall forgetting

# Nice visuals
import matplotlib.pyplot as plt
plt.imshow(M, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Return / Accuracy')
plt.xlabel('Task i'); plt.ylabel('After task k')
plt.title('Performance matrix')
plt.show()

plt.bar(np.arange(T), F)
plt.xlabel('Task i'); plt.ylabel('Forgetting F_i')
plt.title(f'Per-task forgetting (mean={FGT:.3f})')
plt.show()
