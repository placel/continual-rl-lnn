import numpy as np

# Critic CfC
acc_vals = np.array([
    0.47, 
    0.47,
    0.24,
    0.23,
    0.33
])

fwt_vals = np.array([
    0.0,
    0.01,
    -0.12,
    0.13,
    0.0
])

bwt_vals = np.array([
    -0.54,
    -0.64,
    -0.32,
    -0.77,
    -0.0
])

wall_vals = np.array([
    331.7881746292114,
    339.60935258865356,
    344.18259167671204,
    334.86193585395813,
    324.888578414917
])

acc_mean, acc_std = acc_vals.mean(), acc_vals.std(ddof=1)
fwt_mean, fwt_std = fwt_vals.mean(), fwt_vals.std(ddof=1)
bwt_mean, bwt_std = bwt_vals.mean(), bwt_vals.std(ddof=1)
wall_mean, wall_std = wall_vals.mean(), wall_vals.std(ddof=1)

avg_minutes = wall_mean / 60
std_minutes = wall_std / 60

print(f"ACC: {acc_mean:.2f} ± {acc_std:.2f}")
print(f"FWT: {fwt_mean:.2f} ± {fwt_std:.2f}")
print(f"BWT: {bwt_mean:.2f} ± {bwt_std:.2f}")
print(f"Wall-time (min): {avg_minutes:.2f} ± {std_minutes:.2f}")