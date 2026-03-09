from typing import Dict, List, Optional, Union, Any, Tuple
from numpy.typing import NDArray
from .types import SSAError, SSARankSummary, RankResult
from .utils import standardize_data
from .jdc import joint_diagonalization
from .scatter import *

class SSA:
    """
    Stationary Subspace Analysis (SSA)

    Usage
    -----
    1. Initialize:
        model = SSA(data, partition=part, scatter=scatter)
    2. Optionally estimate the rank of the nonstationary subspace:
        q = model.estimate_rank()
    3. Extract stationary and nonstationary subspaces:
        ss, ns = model.subspaces(q)

    Parameters
    ----------
    data : ndarray of shape (n_signals, n_samples)
        Observed data matrix. Decomposition is performed immediately on construction.
    partition : ndarray, optional
        Partition labels for observations.
    scatter : dict, list, or scatter object
        Scatter matrices or operators for subspace analysis.
    dim_estimator : AugmentationRankEstimator, optional
        An object to compute an estimate of the nonstationary dimension.
        Currently only support one kind of estimator but could be easily
        extended to include different estimators.

    Attributes
    ----------
    whitener_ : ndarray
        Whitening matrix from data standardization.
    diagonalizer_ : ndarray
        Matrix that diagonalizes the scatter matrices.
    eigenvalues_ : ndarray
        Eigenvalues from the diagonalization.
    individual_models_ : dict[str, SSA]
        Decomposed models for individual scatters.
    estimated_rank_ : int or None
        Rank estimated by estimate_rank(), if called.
    rank_summary_ : SSARankSummary or RankResult
        Full rank estimation diagnostics, if estimate_rank() has been called.
    """

    def __init__(
            self,
            data: NDArray[np.float64],
            partition: NDArray,
            scatter: Union[Dict[str, Any], List[Any], Any],
            dim_estimator: Optional['AugmentationRankEstimator'] = None,
    ) -> None:
        self.scatters = self._validate_scatter(scatter)
        self.dim_estimator = dim_estimator
        self.estimated_rank_: Optional[int] = None

        white_data, self.whitener_ = self._prepare_data(data)
        self._partition_ = partition
        self._decompose_from_white(white_data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_rank(self, individual: bool = False) -> int:
        """
        Estimate the rank (dimension) of the nonstationary subspace.

        Parameters
        ----------
        individual : bool, default False
            If True, also estimate rank for each individual scatter and store
            full results in rank_summary_. Slower but more informative.

        Returns
        -------
        int
            Estimated rank of the nonstationary subspace.
        """
        if self.dim_estimator is None:
            raise ValueError(
                "No dimension estimator specified. "
                "Pass dim_estimator= at construction, or use a convenience "
                "constructor such as SSA_SIR()."
            )

        temp = self._clone_without_dim_estimator()
        joint_result = self.dim_estimator.estimate(temp)

        if individual and len(self.scatters) > 1:
            individual_results = {
                name: self.dim_estimator.estimate(model)
                for name, model in temp.individual_models_.items()
            }
            self.rank_summary_ = SSARankSummary(
                joint=joint_result,
                individual=individual_results,
            )
        else:
            self.rank_summary_ = joint_result

        self.estimated_rank_ = joint_result.rank
        return self.estimated_rank_

    def subspaces(self, q: Optional[int] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the stationary and nonstationary subspace projection matrices.

        Parameters
        ----------
        q : int, optional
            Dimension of the nonstationary subspace. If omitted, uses the rank
            stored by a prior call to estimate_rank().

        Returns
        -------
        stationary : ndarray of shape (n_signals - q, n_signals)
            Projection onto the stationary subspace.
        non_stationary : ndarray of shape (q, n_signals)
            Projection onto the nonstationary subspace.
        """
        if q is None:
            if self.estimated_rank_ is None:
                raise ValueError(
                    "No rank available. Either pass q= explicitly or call estimate_rank() first."
                )
            q = self.estimated_rank_

        n_components = self.diagonalizer_.shape[1]
        if not (0 <= q <= n_components):
            raise ValueError(f"q must be between 0 and {n_components}, got {q}")

        try:
            V = self.diagonalizer_
            non_stationary = V[:, :q].T @ self.whitener_
            stationary = V[:, q:].T @ self.whitener_
            return stationary, non_stationary
        except AttributeError as e:
            raise SSAError(
                "Decomposition attributes are missing. The model may not have "
                "been constructed correctly."
            ) from e

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues from the scatter decomposition (copy)."""
        return self.eigenvalues_.copy()

    @property
    def whitener(self) -> NDArray[np.float64]:
        """Whitening matrix (copy)."""
        return self.whitener_.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_scatter(scatter) -> dict:
        if not scatter:
            raise ValueError("At least one scatter must be provided")

        if isinstance(scatter, dict):
            if not all(isinstance(k, str) for k in scatter.keys()):
                raise TypeError("Scatter dict keys must be strings")
            for name, s in scatter.items():
                if not hasattr(s, 'compute'):
                    raise TypeError(f"Scatter '{name}' missing required 'compute' method")
            return scatter
        elif isinstance(scatter, list):
            if not scatter:
                raise ValueError("Scatter list cannot be empty")
            if isinstance(scatter[0], tuple):
                return dict(scatter)
            for s in scatter:
                if not hasattr(s, 'compute'):
                    raise TypeError(f"Scatter {s} missing required 'compute' method")
            return {s.__class__.__name__: s for s in scatter}
        else:
            if not hasattr(scatter, 'compute'):
                raise TypeError("Scatter missing required 'compute' method")
            return {scatter.__class__.__name__: scatter}

    def _prepare_data(
            self,
            data: NDArray[np.float64],
    ) -> Tuple[NDArray, NDArray]:
        # checks on input data and data whitening 
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data must be a numpy array, got {type(data)}")
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-D, got shape {data.shape}")
        if data.size == 0:
            raise ValueError("Data array is empty")

        white_data, whitener = standardize_data(data)

        return white_data, whitener

    def _decompose_from_white(self, white_data: NDArray[np.float64]) -> None:
        self._white_data_ = white_data

        matrices = []
        self.individual_models_ = {}  # dict for results from individual scatters

        for name, scatter in self.scatters.items():
            m = scatter.compute(white_data, self._partition_)
            matrices.append(m)

            clone = self._clone_without_dim_estimator()
            clone._fit_single(m)
            self.individual_models_[name] = clone

        if len(matrices) == 1:
            self._fit_single(matrices[0])
        else:
            self._fit_joint(matrices)

    def _fit_single(self, m_mat: NDArray[np.float64]) -> None:
        # eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(m_mat)
        perm = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[perm]
        self.diagonalizer_ = eigvecs[:, perm]

    def _fit_joint(self, matrices: List[NDArray[np.float64]]) -> None:
        # joint approximate diagonalization
        X = np.concatenate(matrices, axis=0)
        V, D, _ = joint_diagonalization(X)
        diag = np.diagonal(sum(np.abs(D)))
        perm = np.argsort(diag)[::-1]
        self.eigenvalues_ = diag[perm]
        self.diagonalizer_ = V[:, perm]

    def _clone_without_dim_estimator(self):
        """Internal clone used during rank estimation — bypasses __init__ decomposition."""
        clone = object.__new__(type(self))
        clone.scatters = self.scatters
        clone.dim_estimator = None
        clone.estimated_rank_ = None
        clone._white_data_ = self._white_data_
        clone._partition_ = self._partition_
        clone.whitener_ = self.whitener_
        clone.diagonalizer_ = getattr(self, 'diagonalizer_', None)
        clone.eigenvalues_ = getattr(self, 'eigenvalues_', None)
        clone.individual_models_ = getattr(self, 'individual_models_', {})
        return clone


# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------

def SSA_SIR(data, partition, s=10, r=10, **kwargs) -> SSA:
    return SSA(
        data, partition,
        scatter=SIRScatter(),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SSA_SAVE(data, partition, s=10, r=10, **kwargs) -> SSA:
    return SSA(
        data, partition,
        scatter=SAVEScatter(),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SSA_COR(data, partition, lag=1, s=10, r=10, **kwargs) -> SSA:
    return SSA(
        data, partition,
        scatter=CORScatter(lag),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SSA_COMB(data, partition, lag=1, s=10, r=10, **kwargs) -> SSA:
    return SSA(
        data, partition,
        scatter=[
            ("sir",  SIRScatter()),
            ("save", SAVEScatter()),
            ("cor",  LCORScatter(lag)),
        ],
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def normalized_scree(eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute normalized scree values from eigenvalues.

    Parameters
    ----------
    eigenvalues : ndarray

    Returns
    -------
    ndarray
        Normalized scree values in [0, 1].
    """
    abs_eigs = np.abs(eigenvalues)
    cumulative = np.cumsum(abs_eigs)
    phi = np.ones_like(abs_eigs)
    phi[1:] = abs_eigs[1:] / cumulative[1:]
    return phi

class AugmentationRankEstimator:
    """
    Rank estimator based on data augmentation.

    Parameters
    ----------
    noise_dim : int
        Number of Gaussian noise components to append.
    num_rep : int
        Number of augmentation trials.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
            self,
            noise_dim: int,
            num_rep: int = 10,
            random_state: Optional[int] = None,
    ) -> None:
        if noise_dim <= 0:
            raise ValueError("noise_dim must be positive")
        if num_rep <= 0:
            raise ValueError("num_rep must be positive")

        self.noise_dim = noise_dim
        self.num_rep = num_rep
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def _augment(self, white_data: NDArray[np.float64]) -> NDArray[np.float64]:
        p, n = white_data.shape
        noise = self._rng.standard_normal((self.noise_dim, n))
        return np.vstack([white_data, noise])

    def estimate(self, model: SSA) -> RankResult:
        """
        Estimate rank by combining measurements of norms of augmented eigenvectors 
        and normalised scree plot of eigenvalues.

        Parameters
        ----------
        model : SSA
            An SSA model to compute the eigendecompositions.

        Returns
        -------
        RankResult
        """
        if not hasattr(model, '_white_data_'):
            raise RuntimeError("Model must be decomposed before rank estimation.")

        white_data = model._white_data_
        p, _ = white_data.shape

        # compute norms of v^{AUG} for self.num_rep repetitions
        norms_all = np.zeros((self.num_rep, p))
        for s in range(self.num_rep):
            aug_data = self._augment(white_data)
            temp = model._clone_without_dim_estimator()
            temp._decompose_from_white(aug_data)

            V = temp.diagonalizer_
            v_aug = V[p:, :p]
            norms_all[s] = np.einsum('ij,ij->j', v_aug, v_aug)

        # mean data of augmented eigenvectors
        f_vec = norms_all.mean(axis=0)
        cum_f_vec = np.cumsum(f_vec)

        # scree plot of eigenvalues
        phi = normalized_scree(model.eigenvalues_)

        # combined estimate
        g_vec = cum_f_vec + phi

        return RankResult(
            rank=int(np.argmin(g_vec)),
            f_vec=f_vec,
            phi=phi,
            g_vec=g_vec,
            cum_f_vec=cum_f_vec,
        )
