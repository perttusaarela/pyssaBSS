import numpy as np
from .joint_diag import jdc

def _do_python_jdc(X, kpmaxit, w, eps):
    X = np.asarray(X, dtype=np.float64)
    kpmaxit = np.asarray(kpmaxit, dtype=np.int32)
    w = np.asarray(w, dtype=np.float64)
    eps = np.asarray([eps], dtype=np.float64)

    return jdc(X, kpmaxit, w, eps)


def joint_diagonalization(X, weight=None, maxiter=1000, eps=1e-6):
    """
    Approximate joint diagonalization

    Parameters
    ----------
    X : ndarray of shape (kp, p)
        A concatenation of the k p-by-p matrices to be jointly diagonalized
    weight : list(float), optional
        A list of weights given to each matrix, prior to diagonalization
    maxiter : int
        maximum number of iteration in the algorithm before finishing.
        An assertion is raised if no convergence is found in this time.
    eps : float
        convergence tolerance

    Returns
    -------
    V : ndarray of shape (p, p)
        joint diagonalized
    D : list of ndarrays of shape (p, p)
        Gives the "diagonalized" matrices from the procedure in a list
    iter : int
        total number of iterations 
    """

    kp, p = X.shape
    k = kp // p
    assert k * p == kp

    if weight is None:
        weight = np.ones(k, dtype=float)

    res = _do_python_jdc(X.flatten(order="F"), [k, p, maxiter], weight, [eps])

    iter = res[-1]
    assert iter < maxiter, "maxiter reached without convergence"

    V = np.asarray(res[:-1]).reshape([p,p]).transpose()
    D = []
    for i in range(k):
        matrix = X[i * p:(i + 1) * p, :]
        D.append(V.T @ (matrix.T @ V))

    return V, D, iter
