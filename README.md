# pyssaBSS
[![CI](https://github.com/perttusaarela/pyssaBSS/actions/workflows/publish.yml/badge.svg)](https://github.com/perttusaarela/pyssaBSS/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/pyssaBSS/badge/?version=latest)](https://pyssaBSS.readthedocs.io)

Stationary Subspace Analysis (SSA) for temporal and spatial data.

`pyssaBSS` implements methods for separating stationary and nonstationary subspaces
in multivariate data, based on the joint eigen-decomposition of scatter operators.  It provides `SSA` for
time series and `SPSSA` for spatial fields.

Note that the C++ code used for joint diagonalization is taken from the CRAN R-package JADE under the GPL license. 
The original package can be found at: https://cran.r-project.org/package=JADE

---

## Installation

```bash
pip install pyssaBSS
```

Pre-built wheels are available for Windows, macOS, and Linux (Python 3.8–3.13).
No compiler required.

### Building from source

```bash
git clone https://github.com/perttusaarela/pyssaBSS
cd pyssaBSS
pip install -e ".[dev]"
```

Requires CMake ≥ 3.15 and a C++14 compiler.

---

## Quick example

```python
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
```

---

## Overview

| Class       | Data type         | Varies across       |
|-------------|-------------------|---------------------|
| `SSA`       | Time series       | Time partitions     |
| `SPSSA`     | Spatial fields    | Spatial segments    |

Convenience constructors (`SSA_SIR`, `SPSSA_SIR`, `SPSSA_LCOR`, `SPSSA_COMB`, ...)
bundle a scatter operator and rank estimator into a single call.

---

## License

GLP 3.0 — see [LICENSE.txt](LICENSE.txt).
