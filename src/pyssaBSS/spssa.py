from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from numpy.typing import NDArray
from .types import SSARankSummary, RankResult
from .scatter import *
from .ssa import SSA, AugmentationRankEstimator


class SPSSA(SSA):
    """
    Spatial Stationary Subspace Analysis (SPSSA)

    
    Examples
    --------
    Initialize the model::

        model = SPSSA(data, coords, scatter=scatter, partition=partition)

    Optionally estimate the rank of the nonstationary subspace::

        q = model.estimate_rank()

    Extract stationary and nonstationary subspaces::

        ss, ns = model.subspaces(q)


    Parameters
    ----------
    data : ndarray of shape (n_signals, n_samples)
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
    """

    def __init__(
            self,
            data: Union[NDArray[np.float64], 'pd.DataFrame'],
            partition: NDArray,
            scatter: Union[Dict[str, Any], List[Any], Any],
            coords: Optional[NDArray] = None,
            coord_cols: Optional[list] = None,
            data_cols: Optional[list] = None,
            dim_estimator: Optional['AugmentationRankEstimator'] = None,
    ) -> None:
        self.scatters      = self._validate_scatter(scatter)
        self.dim_estimator = dim_estimator
        self.estimated_rank_: Optional[int] = None

        white_data, self.whitener_, coords = self._prepare_data(data, data_cols, coords, coord_cols)
        self._coords_   = coords
        self._partition_ = partition
        self._decompose_from_white(white_data)

    # Public API is inherited from SSA

    # Overridden private functions

    def _prepare_data(self, data, data_cols, coords, coord_cols):
        # check input data for dataframe
        if hasattr(data, 'to_numpy'):
            if coord_cols is None:  # need to specify which columns are coordinates
                raise ValueError(
                    "When passing a DataFrame, coord_cols must specify which "
                    "columns contain coordinates, e.g. coord_cols=['x', 'y']"
                )
            coords = data[coord_cols].to_numpy(dtype=np.float64)  # extract coords
            if data_cols is not None:  # check if data columns provided
                data = data[data_cols].to_numpy(dtype=np.float64)
            else:
                # if not, everything but the coordinate columns will be used
                data = data.drop(columns=coord_cols).to_numpy(dtype=np.float64)
        else:
            if coords is None:
                raise ValueError("coords must be provided when data is a numpy array")

        white_data, whitener = super()._prepare_data(data)
        return white_data, whitener, coords

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

def SPSSA_SIR(data, partition, coords=None, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data=data, partition=partition, coords=coords,
        scatter=SIRScatter(),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SPSSA_SAVE(data, partition, coords=None, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data=data, partition=partition, coords=coords,
        scatter=SAVEScatter(),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SPSSA_LCOR(data, partition, coords=None, kernel=None, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data=data, partition=partition, coords=coords,
        scatter=LCORScatter(kernel),
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )


def SPSSA_COMB(data, partition, coords=None, kernel=None, s=10, r=10, **kwargs) -> SPSSA:
    return SPSSA(
        data=data, partition=partition, coords=coords,
        scatter=[
            ("sir",  SIRScatter()),
            ("save", SAVEScatter()),
            ("cor",  LCORScatter(kernel)),
        ],
        dim_estimator=AugmentationRankEstimator(noise_dim=r, num_rep=s),
        **kwargs,
    )
