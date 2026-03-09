from .utils import (gaussian_kernel_local_sample_covariance, scaled_local_sample_covariance,
                   ball_kernel_local_sample_covariance, ring_kernel_local_sample_covariance)
class BaseKernel(object):
    def global_covariance(self, data, coords):
        raise NotImplementedError()

    def local_covariance(self, data, coords, segment):
        raise NotImplementedError()


class BallKernel(BaseKernel):
    def __init__(self, radius):
        self.radius = radius

    def global_covariance(self, data, coords):
        return ball_kernel_local_sample_covariance(data, coords, self.radius)

    def local_covariance(self, data, coords, segment):
        return ball_kernel_local_sample_covariance(data, coords, self.radius, segment)


class ScaledBallKernel(BaseKernel):

    def __init__(self, radius):
        self.radius = radius

    def global_covariance(self, data, coords):
        return scaled_local_sample_covariance(data, coords, self.radius)

    def local_covariance(self, data, coords, segment):
        return scaled_local_sample_covariance(data, coords, self.radius, segment)


class RingKernel(BaseKernel):
    def __init__(self, inner_radius, outer_radius):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def global_covariance(self, data, coords):
        return ring_kernel_local_sample_covariance(data, coords, self.inner_radius, self.outer_radius)

    def local_covariance(self, data, coords, segment):
        return ring_kernel_local_sample_covariance(data, coords, self.inner_radius, self.outer_radius, segment)


class GaussianKernel(BaseKernel):
    def __init__(self, radius):
        self.radius = radius

    def global_covariance(self, data, coords):
        return gaussian_kernel_local_sample_covariance(data, coords, self.radius)

    def local_covariance(self, data, coords, segment):
        return gaussian_kernel_local_sample_covariance(data, coords, self.radius, segment=segment)