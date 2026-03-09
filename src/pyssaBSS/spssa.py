from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from numpy.typing import NDArray
from .types import SSARankSummary, RankResult
from .scatter import *
from .ssa import SSA, AugmentationRankEstimator


class SPSSA(SSA):
    """
    Spatial Stationary Subspace Analysis (SPSSA)

    Usage
    -----
    1. Initialize:
        model = SPSSA(data, coords, scatter=scatter, partition=segs)
    2. Optionally estimate the rank of the nonstationary subspace:
        q = model.estimate_rank()
    3. Extract stationary and nonstationary subspaces:
        ss, ns = model.subspaces(q)

    Parameters
    ----------
    data : ndarray of shape (n_features, n_samples)
        Observed data matrix. Decomposition is performed immediately on construction.
    coords : ndarray, optional
        Spatial coordinates passed to the segmentation function.
        Required when no pre-computed partition are provided.
    partition : ndarray, optional
        Pre-computed segment labels.
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
    individual_models_ : dict[str, SPSSA]
        Decomposed models for individual scatters.
    estimated_rank_ : int or None
        Rank estimated by estimate_rank(), if called.
    rank_summary_ : SSARankSummary or RankResult
        Full rank estimation diagnostics, if estimate_rank() has been called.
    """

    def __init__(
            self,
            data: NDArray[np.float64],
            coords: NDArray,
            partition: NDArray,
            scatter: Union[Dict[str, Any], List[Any], Any],
            dim_estimator: Optional['AugmentationRankEstimator'] = None,
    ) -> None:
        self.scatters      = self._validate_scatter(scatter)
        self.dim_estimator = dim_estimator
        self.estimated_rank_: Optional[int] = None

        white_data, self.whitener_ = self._prepare_data(data)
        self._coords_   = coords
        self._partition_ = partition
        self._decompose_from_white(white_data)

    # Public API is inherited from SSA

    def _decompose_from_white(self, white_data: NDArray[np.float64]) -> None:
        self._white_data_ = white_data

        matrices = []
        self.individual_models_ = {}

        for name, scatter in self.scatters.items():
            # Only difference to SSA is here, the scatters also take coordinates
            m = scatter.compute(white_data, self._partition_, self._coords_)
            matrices.append(m)

            clone = self._clone_without_dim_estimator()
            clone._fit_single(m)
            self.individual_models_[name] = clone

        if len(matrices) == 1:
            self._fit_single(matrices[0])
        else:
            self._fit_joint(matrices)

    def _clone_without_dim_estimator(self):
        clone = super()._clone_without_dim_estimator()
        clone._coords_ = self._coords_
        return clone


# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------

def SPSSA_SIR(data, coords, partition, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data, coords, partition,
        scatter=SIRScatter(),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SPSSA_SAVE(data, coords, partition, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data, coords, partition,
        scatter=SAVEScatter(),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SPSSA_LCOR(data, coords, partition, kernel=None, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data, coords, partition,
        scatter=LCORScatter(kernel),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SPSSA_COMB(data, coords, partition, kernel=None, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data, coords, partition,
        scatter=[
            ("sir",  SIRScatter()),
            ("save", SAVEScatter()),
            ("cor",  LCORScatter(kernel)),
        ],
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )
