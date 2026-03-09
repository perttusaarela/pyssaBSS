Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install pyssaBSS

Quick Example
-------------

.. code-block:: python

    import numpy as np
    from pyssaBSS import SSA_COMB

    # data: (n_signals, n_samples), coords: (n_samples, 2)
    data   = np.random.randn(5, 400)

    # Partition the time domain
    partition = np.split(np.arange(400), 10)

    # Add nonstationarity in the mean to the last signal
    for part in partition:
        data[-1, part] += np.random.uniform(low=-5, high=5)

    # Decompose
    model = SSA_COMB(data, coords, partition=partition, kernel=ScaledBallKernel(2.2))

    # Estimate rank of non-stationary subspace
    q = model.estimate_rank()
    print(f"Estimated rank: {q}")

    # Extract subspaces
    stationary, non_stationary = model.subspaces(q)
    print(f"Stationary subspace:     {stationary.shape}")
    print(f"Non-stationary subspace: {non_stationary.shape}")