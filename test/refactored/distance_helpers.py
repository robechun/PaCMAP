import numba
import numpy as np

from enum import Enum


# Enum for the different types of distances
class DistanceType(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    ANGULAR = "angular"
    HAMMING = "hamming"


@numba.njit("f4(f4[:])", cache=True)
def l2_norm(x):
    """
    L2 norm of a vector.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def manhattan_dist(x1, x2):
    """
    Manhattan distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i] - x2[i])
    return result


@numba.njit("f4(f4[:],f4[:])", cache=True)
def angular_dist(x1, x2):
    """
    Angular (i.e. cosine) distance between two vectors.
    """
    x1_norm = np.maximum(l2_norm(x1), 1e-20)
    x2_norm = np.maximum(l2_norm(x2), 1e-20)
    result = 0.0
    for i in range(x1.shape[0]):
        result += x1[i] * x2[i]
    return np.sqrt(2.0 - 2.0 * result / x1_norm / x2_norm)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def hamming_dist(x1, x2):
    """
    Hamming distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@numba.njit(cache=True)
def calculate_dist(x1, x2, distance_type: DistanceType):
    """
    Calculate distance between two vectors, based on the distance type.
    """
    if distance_type == DistanceType.EUCLIDEAN:
        return euclid_dist(x1, x2)
    if distance_type == DistanceType.MANHATTAN:
        return manhattan_dist(x1, x2)
    if distance_type == DistanceType.ANGULAR:
        return angular_dist(x1, x2)
    if distance_type == DistanceType.HAMMING:
        return hamming_dist(x1, x2)


@numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True, cache=True)
def scale_dist(knn_distance, sig, nbrs):
    """Scale the distance"""
    n, num_neighbors = knn_distance.shape
    scaled_dist = np.zeros((n, num_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(num_neighbors):
            scaled_dist[i, j] = knn_distance[i, j] ** 2 / sig[i] / sig[nbrs[i, j]]
    return scaled_dist
