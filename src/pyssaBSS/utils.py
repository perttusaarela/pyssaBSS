import numpy as np
from scipy.stats import ortho_group


def generate_random_orthogonal_matrix(n):
    """
    Generate a random orthogonal matrix of size n x n

    Parameters
    ----------
    n : int
        Number of rows and columns

    Returns
    -------
    ndarray of shape (n, n)
    """
    return ortho_group.rvs(n)


def generate_random_invertible_matrix(p):
    """
    Generate a random invertible matrix of size p x p

    Parameters
    ----------
    p : int
        Number of rows and columns

    Returns
    -------
    ndarray of shape (p, p)
    """
    candidate = np.random.rand(p, p)
    while np.linalg.matrix_rank(candidate) != p:
        candidate = np.random.rand(p, p)

    return candidate


def sample_mean(data, segment=None):
    """
    Compute the mean of the data points.

    Parameters
    ----------
    data : ndarray of shape (p, n)
        if provided, compute the covariance over the given indi
    segment : ndarray, optional

    Returns
    -------
    ndarray of shape (p,)
    """
    if segment is None:
        return np.mean(data, axis=1)

    return np.mean(data[:, segment], axis=1)


def sample_covariance(data, segment=None, seg_mean=None):
    """
    Compute the covariance of the data points.

    Parameters
    ----------
    data : ndarray of shape (p, n)
    segment : ndarray, optional
        if provided, compute the covariance over the given indices
    seg_mean : ndarray, optional
        if provided, the mean is not recomputed
    Returns
    -------
    ndarray of shape (p, p)
    """
    if segment is None:
        segment = range(data.shape[1])
    X = data[:, segment]

    if seg_mean is None:
        seg_mean = X.mean(axis=1, keepdims=True)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    centered = X - seg_mean
    cov = (centered @ centered.T) / X.shape[1]
    return cov


def sample_autocovariance(
        data: np.ndarray,
        lag: int,
        segment: np.ndarray = None,
        seg_mean: np.ndarray = None,
) -> np.ndarray:
    """
    Sample autocovariance matrix at a given lag.

    Parameters
    ----------
    data : ndarray of shape (p, n)
    lag : int
        Lag at which to compute autocovariance. Must be >= 0.
    segment : ndarray of int indices, optional
    seg_mean : ndarray of shape (p,) or (p, 1), optional
        If None, computed from the segment.

    Returns
    -------
    acov : ndarray of shape (p, p)
        Not necessarily symmetric (acov(lag) != acov(lag).T for lag > 0).
    """
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")

    if segment is None:
        segment = np.arange(data.shape[1])
    segment = np.asarray(segment)

    X = data[:, segment]                        # (p, n)

    if seg_mean is None:
        seg_mean = X.mean(axis=1, keepdims=True)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    Xc = X - seg_mean                           # (p, n)
    N  = Xc.shape[1]

    acov = (Xc[:, :N - lag] @ Xc[:, lag:].T) / (N-lag)
    return acov


def standardize_data(data):
    """
    Standardize the data.

    Afterwards, data has mean 0 and covariance I.

    Parameters
    ----------
    data : ndarray of shape (p, n)

    Returns
    -------
    white_data: ndarray of shape (p, n)
        data after standardization
    whitener: ndarray of shape (p, p)
        inverse square root of the covariance
    """
    data -= data.mean(axis=1, keepdims=True)
    cov = np.cov(data, bias=True, rowvar=True)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Compute A^{-1/2}
    sqrt_cov = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    return sqrt_cov @ data, sqrt_cov


gauss_const = 1.6448536269514722  # \Psi^{-1}(0.95)


def scaled_local_sample_covariance(data, coords, radius, segment=None, seg_mean=None):
    if segment is None:
        segment = np.arange(data.shape[1])
    segment = np.asarray(segment)

    X = data[:, segment]  # (D, N)
    C = coords[segment]  # (N, 2)
    N = X.shape[1]
    D = X.shape[0]

    # Mean
    if seg_mean is None:
        seg_mean = X.mean(axis=1, keepdims=True)

    Xc = X - seg_mean  # (D, N)

    # Pairwise distance mask
    diff = C[:, None, :] - C[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    mask = (dist2 <= radius ** 2)

    # Only consider j > i
    mask = np.triu(mask, k=1)

    l_cov = np.zeros((D, D))

    for i in range(N):
        js = np.where(mask[i])[0]
        counter = js.size
        if counter == 0:
            continue

        Xi = Xc[:, i:i + 1]  # (D, 1)
        Xj = Xc[:, js]  # (D, K)

        Sj = np.sum(Xj, axis=1, keepdims=True)  # (D, 1)

        # Sum_j (Xi*Xj^T) + Sum_j (Xj*Xi^T)
        tmp = Xi @ Sj.T + Sj @ Xi.T  # (D, D)

        l_cov += tmp / counter  # per-i normalization

    return l_cov / N  # final normalization


def ball_kernel_local_sample_covariance(data, coords, radius, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]                    # Shape: (D, N)
    C = coords[segment]                     # Shape: (N, d)
    N = X.shape[1]
    D = X.shape[0]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)  # Shape: (D, 1)
    else:
        seg_mean = seg_mean[:, np.newaxis]            # Ensure shape (D, 1)

    X_centered = X - seg_mean                         # Shape: (D, N)

    # Compute pairwise distances (squared)
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]   # (N, N, d)
    sq_dists = np.sum(diffs ** 2, axis=2)               # (N, N)

    # Create weights matrix using ball kernel
    mask = (sq_dists <= radius ** 2).astype(float)      # binary weights
    np.fill_diagonal(mask, 0.0)                     # exclude self-pairs

    # Step 3: Compute weighted covariance using matrix multiplication
    l_cov = (X_centered @ mask @ X_centered.T) / (N)

    return l_cov


def ring_kernel_local_sample_covariance(data, coords, inner_radius, outer_radius, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]                    # (D, N)
    C = coords[segment]                     # (N, d)
    N = X.shape[1]
    D = X.shape[0]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)  # (D, 1)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    X_centered = X - seg_mean               # (D, N)

    # Pairwise squared distances
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # (N, N, d)
    sq_dists = np.sum(diffs ** 2, axis=2)              # (N, N)

    # Create binary mask for ring kernel
    r2_inner = inner_radius ** 2
    r2_outer = outer_radius ** 2
    mask = ((sq_dists > r2_inner) & (sq_dists <= r2_outer)).astype(float)

    # Remove diagonal (u == u')
    np.fill_diagonal(mask, 0.0)

    # Weighted covariance calculation
    l_cov = (X_centered @ mask @ X_centered.T) / N

    return l_cov


def gaussian_kernel_local_sample_covariance(data, coords, radius, segment=None, seg_mean=None):
    if segment is None:
        segment = range(data.shape[1])
    segment = np.array(segment)

    X = data[:, segment]  # (D, N)
    C = coords[segment]  # (N, d)
    N = X.shape[1]
    D = X.shape[0]

    # Compute or reuse mean
    if seg_mean is None:
        seg_mean = np.mean(X, axis=1, keepdims=True)  # (D, 1)
    else:
        seg_mean = seg_mean[:, np.newaxis]

    X_centered = X - seg_mean  # (D, N)

    # Pairwise squared distances
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # (N, N, d)
    sq_dists = np.sum(diffs ** 2, axis=2)  # (N, N)

    # Gaussian kernel weights
    scale = gauss_const ** 2 / (2 * radius ** 2)
    weights = np.exp(-scale * sq_dists)

    # Zero diagonal (u == u')
    np.fill_diagonal(weights, 0.0)

    # Weighted covariance
    l_cov = (X_centered @ weights @ X_centered.T) / N

    return l_cov
