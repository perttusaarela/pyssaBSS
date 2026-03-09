import numpy as np
from sklearn.gaussian_process.kernels import Matern


# ----------------------------------------------------------------------
# Functions for generating spatial data
# ----------------------------------------------------------------------

def generate_coordinates(num_data_points: int, hi: float = 1.0):
    """
    Generates uniformly random 2-D coordinates from [0, hi) x [0, hi)

    Parameters
    ----------
    num_data_points : int
        Number of data points to generate
    hi : float
        Upper bound for the sampling area.

    Returns
    -------
    coordinate: ndarray of shape (num_data_points, 2)
    """
    return hi * np.random.rand(num_data_points, 2)


def spatial_data_from_cholesky(cholesky):
    """
    Spatial data via the Cholesky method.

    Parameters
    ----------
    cholesky : ndarray of shape (n, n)


    Returns
    -------
    spatial_data: ndarray of shape (n, 1)
        zero-mean spatial data with covariance L^T L
    """

    n = cholesky.shape[0]
    gaussian_data = np.random.multivariate_normal(np.zeros(n), np.eye(n))   # zero mean vector, Cov = I_n
    spatial_data = cholesky @ gaussian_data

    return spatial_data


def generate_spatial_data(covariance_matrix, mean=None):
    """
    Generate spatial data from covariance matrix using the Cholesky decomposition.

    Parameters
    ----------
    covariance_matrix : ndarray of shape (n, n)
        a positive-semidefinite covariance matrix (must be cholesky decomposable)
    mean : ndarray of shape (n, 1), optional

    Returns
    -------
    spatial_data: ndarray of shape (n, 1)
        spatial data with mean 'mean' and covariance 'covariance_matrix'
    """
    if mean is None:
        mean = np.zeros(covariance_matrix.shape[1])  # if mean is not specified, it is assumed to be zero

    cholesky = np.linalg.cholesky(covariance_matrix)  # compute the cholseky decomp. of Cov
    spatial_data = spatial_data_from_cholesky(cholesky) + mean

    return spatial_data


def matern_covariance(points, nu=1.5, phi=1.0):
    """
    Computes the covariance matrix of a set of points using sklearn Matern kernel. 
    This differs from the usual Matern Kernel by a constant.

    Parameters
    ----------
    points : ndarray of shape (n, 2)
    nu : float
        smoothness parameter
    phi : float
        range parameter

    Returns
    -------
    mat: ndarray of shape (n, n)
        spatial covariance matrix
    """

    matern = Matern(length_scale=phi, nu=nu)
    mat = matern(points)
    return mat


def ssa_matern_covariance(points, nu=0.5, phi=1.0, sigma=1.0):
    """
    Computes the usual Matern covariance matrix of a set of points using sklearn Matern kernel.

    Parameters
    ----------
    points : ndarray of shape (n, 2)
    nu : float
        smoothness parameter
    phi : float
        range parameter
    sigma : float
        variance parameter

    Returns
    -------
    mat: ndarray of shape (n, n)
        spatial covariance matrix
    """
    return sigma * matern_covariance(points, nu=nu, phi=phi * np.sqrt(2 * nu))


def params_to_block_vector(params, segments):
    """
    Computes a vector where the indices of segments[i] has the value params[i]

    Parameters
    ----------
    params : list of values
    segments : list of list of indices 

    Returns
    -------
    result: ndarray of shape (n, 1)
    """

    # Find maximum index to size the array
    max_index = max(max(seg) for seg in segments)
    result = np.zeros(max_index + 1, dtype=float)

    # Assign each segment's parameter to its indices
    for param, seg in zip(params, segments):
        result[np.array(seg)] = param

    return result


# ----------------------------------------------------------------------
# Functions for partitioning spatial data
# ----------------------------------------------------------------------


def is_in_rectangle_mask(points, corner, height, width):
    """
    Vectorized check of which points are contains in the box outlined by 
    corner, height, and width

    Parameters
    ----------
    points : ndarray of shape (n, 2)
    corner : tuple(float, float)
        indicates the bottom left corner of a rectangle
    height : float
        indicates the height of the rectangle
    width : float
        indicates the width of the rectangle

    Returns
    -------
    mask: ndarray of booleans of shape (num_data_points, 2)
    """

    x, y = corner
    mask_x = (points[:, 0] >= x) & (points[:, 0] < x + width)
    mask_y = (points[:, 1] >= y) & (points[:, 1] < y + height)
    return mask_x & mask_y

def partition_coordinates(coordinates, num_x_segments: int, num_y_segments: int, side_length: float=1.0):
    """
    Grid partition of coordinates. Coordinates lying in a box [0, side_length) x [0, side_lenght)
    are partitioned by a grid given by the number of of cuts in x and y direnctions

    Parameters
    ----------
    coordinates : ndarray of shape (n, 2)
        spatial coordinates in [0, side_length) x [0, side_lenght)
    num_x_segments : int
        number of cuts on along the x-axis
    num_y_segments : int
        number of cuts on along the y-axis
    side_length : float
        Length of the sides of the bounding square

    Returns
    -------
    partition: list(list(int))
        A list of lists of indices. Each list is a part. Each part is a list of indices of coordinates
        lying inside that rectangle. 
    """

    coordinates = np.asarray(coordinates)
    unif_height = side_length / num_y_segments  # so far only uniform partitioning is possible but this could be extended
    unif_width = side_length / num_x_segments

    partition = []
    for iy in range(num_y_segments):
        for ix in range(num_x_segments):
            x0 = ix * unif_width
            y0 = iy * unif_height
            corner = (x0, y0)

            mask = is_in_rectangle_mask(coordinates, corner, unif_height, unif_width)
            indices = np.nonzero(mask)[0]

            partition.append(indices.tolist())

    return partition


def points_in_polygon(points, polygon):
    """
    Vectorized check of which points are contained in the given polygon

    Parameters
    ----------
    points : ndarray of shape (n, 2)
    polygon : ndarray of shape (p, 2)
        a polygon is represented by the coordinates of its vertices in a fixed order

    Returns
    -------
    mask: ndarray of booleans of shape (num_data_points, 2)
    """
    points = np.asarray(points)
    polygon = np.asarray(polygon)

    x = points[:, 0]
    y = points[:, 1]

    x1 = polygon[:, 0]
    y1 = polygon[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)

    dy = y2 - y1
    non_horizontal = dy != 0

    # Only consider non-horizontal edges
    x1 = x1[non_horizontal]
    y1 = y1[non_horizontal]
    x2 = x2[non_horizontal]
    y2 = y2[non_horizontal]
    dy = dy[non_horizontal]

    # Ray casting condition
    cond = ((y1[:, None] > y) != (y2[:, None] > y))

    xinters = (x2[:, None] - x1[:, None]) * (y - y1[:, None]) / dy[:, None] + x1[:, None]

    crossings = cond & (x < xinters)
    return np.sum(crossings, axis=0) % 2 == 1


def partition_points_by_polygons(points, polygons):
    """
    Partition of coordinates based on the first polygon which contains them.

    Parameters
    ----------
    coordinates : ndarray of shape (n, 2)
    polygons : list of ndarrays of shape (num_vertices, 2) 

    Returns
    -------
    partition: list(list(int))
        A list of lists of indices. Each list is a part. Each part is a list of indices of coordinates
        lying inside that rectangle. 
    unassigned: list(int)
        list of coordinates that were in no polygon
    """
    points = np.asarray(points)

    partitions = []
    assigned = np.zeros(len(points), dtype=bool)

    for poly in polygons:
        mask = points_in_polygon(points, poly) & (~assigned)
        idx = np.nonzero(mask)[0]
        partitions.append(idx)
        assigned |= mask

    unassigned = points[~assigned]
    return partitions, unassigned

