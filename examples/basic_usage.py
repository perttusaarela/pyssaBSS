import numpy as np
from pyssaBSS import SPSSA_COMB
from pyssaBSS.kernels import ScaledBallKernel
from pyssaBSS.spatial import partition_coordinates
# data: (n_signals, n_samples), coords: (n_samples, 2)
data   = np.random.randn(5, 400)
coords = np.random.uniform(0, 20, (400, 2))

# Partition the spatial domain
partition = partition_coordinates(coords, 3, 3, 20)

# Add nonstationarity in the mean to the last signal
for part in partition:
    data[-1, part] += np.random.uniform(low=-5, high=5)

# Decompose
model = SPSSA_COMB(data, coords, partition=partition, kernel=ScaledBallKernel(2.2))

# Estimate rank of non-stationary subspace
q = model.estimate_rank()
print(f"Estimated rank: {q}")

# Extract subspaces
stationary, non_stationary = model.subspaces(q)
print(f"Stationary subspace:     {stationary.shape}")
print(f"Non-stationary subspace: {non_stationary.shape}")