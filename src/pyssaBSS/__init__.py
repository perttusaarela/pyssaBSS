from importlib.metadata import version, PackageNotFoundError

from .jdc import joint_diagonalization
from . import kernels
from .polygon import PolygonDrawer
from . import scatter
from .ssa import SSA, AugmentationRankEstimator, SSA_SIR, SSA_COMB, SSA_SAVE, SSA_COR
from .spssa import SPSSA_COMB, SPSSA_SIR, SPSSA_LCOR, SPSSA_SAVE, SPSSA
from . import spatial
from . import utils
from . import types

try:
    __version__ = version("pyssaBSS")
except PackageNotFoundError:
    # Package not installed (e.g. running directly from source)
    __version__ = "unknown"

__all__ = [
    "joint_diagonalization",
    "kernels",
    "PolygonDrawer",
    "AugmentationRankEstimator",
    "scatter",
    "spatial",
    "utils",
    "types",
    "SSA",
    "SSA_SAVE",
    "SSA_COR",
    "SSA_SIR",
    "SSA_COMB",
    "SPSSA",
    "SPSSA_SAVE",
    "SPSSA_LCOR",
    "SPSSA_SIR",
    "SPSSA_COMB",
]