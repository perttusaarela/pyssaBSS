import numpy as np
from .joint_diag import jdc

def do_python_jdc(X, kpmaxit, w, eps):
    X = np.asarray(X, dtype=np.float64)
    kpmaxit = np.asarray(kpmaxit, dtype=np.int32)
    w = np.asarray(w, dtype=np.float64)
    eps = np.asarray([eps], dtype=np.float64)

    return jdc(X, kpmaxit, w, eps)


def joint_diagonalization(X, weight=None, maxiter=1000, eps=1e-6):

    kp, p = X.shape
    k = kp // p
    assert k * p == kp

    if weight is None:
        weight = np.ones(k, dtype=float)

    res = do_python_jdc(X.flatten(order="F"), [k, p, maxiter], weight, [eps])

    iter = res[-1]
    assert iter < maxiter, "maxiter reached without convergence"

    V = np.asarray(res[:-1]).reshape([p,p]).transpose()
    D = []
    for i in range(k):
        matrix = X[i * p:(i + 1) * p, :]
        D.append(V.T @ (matrix.T @ V))

    return V, D, iter
