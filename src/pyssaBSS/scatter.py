import numpy as np
from .utils import sample_mean, sample_covariance, sample_autocovariance
class ScatterOperator:
    """
    Base class for all SSA scatter matrices.
    """

    def compute(self, data, segments, coords=None):
        """
        Returns
        -------
        m_mat : ndarray (p, p)
            Scatter matrix
        """
        raise NotImplementedError


class SIRScatter(ScatterOperator):

    def compute(self, data, segments, coords=None):
        p, n = data.shape
        m_mat = np.zeros((p, p))

        for segment in segments:
            mean_vec = sample_mean(data, segment)
            m_mat += (len(segment) / n) * np.outer(mean_vec, mean_vec)

        return m_mat


class SAVEScatter(ScatterOperator):

    def compute(self, data, segments, coords=None):
        p, n = data.shape
        m_mat = np.zeros((p, p))

        for segment in segments:
            cov_mat = np.eye(p) - sample_covariance(data, segment)
            m_mat += (len(segment) / n) * (cov_mat @ cov_mat)

        return m_mat


class CORScatter(ScatterOperator):

    def __init__(self, lag=1):
        self.lag = lag

    def compute(self, data, segments):
        p, n = data.shape
        m_mat = np.zeros((p, p))

        full_auto_cov = sample_autocovariance(data, self.lag)

        for segment in segments:
            cov_mat = sample_autocovariance(data, self.lag, segment)
            diff = full_auto_cov - cov_mat
            m_mat += (len(segment) / n) * (diff @ diff)

        return m_mat



class LCORScatter(ScatterOperator):

    def __init__(self, kernel):
        self.kernel = kernel

    def compute(self, data, segments, coords):
        p, n = data.shape
        m_mat = np.zeros((p, p))

        full_auto_cov = self.kernel.global_covariance(data, coords)

        for segment in segments:
            cov_mat = self.kernel.local_covariance(data, coords, segment)
            diff = full_auto_cov - cov_mat
            m_mat += (len(segment) / n) * (diff @ diff)

        return m_mat
