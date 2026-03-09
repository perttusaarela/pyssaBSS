from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy._typing import NDArray


class SSAError(Exception):
    """Base exception for statspace package."""
    pass


@dataclass
class RankResult:
    """Result of rank estimation for a single model."""
    rank: int
    f_vec: NDArray[np.float64]
    phi: NDArray[np.float64]
    g_vec: NDArray[np.float64]
    cum_f_vec: NDArray[np.float64]


@dataclass
class SSARankSummary:
    """Complete rank estimation results including joint and individual models."""
    joint: RankResult
    individual: Dict[str, RankResult]



