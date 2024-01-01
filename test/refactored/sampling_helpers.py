import numba
import numpy as np

from .distance_helpers import DistanceType, calculate_dist


@numba.njit("i4[:](i4,i4,i4[:])", nogil=True, cache=True)
def sample_FP(n_samples, maximum, reject_ind):
    """
    TODO fill in details
    """

    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(maximum)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(reject_ind.shape[0]):
                if j == reject_ind[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


def sample_labeled_dist(X, y, n_label_neighbors):
    n = X.shape[0]
    distances = np.empty((n, n_label_neighbors), dtype=np.float32)

    for i in range(n):
        # Grab the label associated with sample i
        label = y[i]

        # get all the points with the same label but make sure not to include itself
        y_label = y[y == label]
        y_label = y_label[y_label.index != i]

        print("y label length")
        print(len(y_label))
        if len(y_label) < n_label_neighbors:
            print("not enough neighbors")
            print(y_label)

        sampled = y_label.sample(n_label_neighbors)
        print("== sampled ===")
        print(sampled)

        jj = 0
        for sample_idx, label in sampled.items():
            distances[i, jj] = calculate_dist(
                X[i], X[sample_idx], distance_type=DistanceType.EUCLIDEAN
            )

    return distances


# y is i -> [i, label]
# TODO @robert: figure out how to sample from the labeled data
# But also make it such that it doesn't collide with NN or MN or FP?
def sample_labeled_pair(X, y, n_label_neighbors):
    # We don't care to discard ones from NN, because if it's both near AND same label,
    # it's even more important to preserve.
    n = X.shape[0]

    # Each sample will have n_label_neighbors neighbors.
    pair_labeled = np.empty((n * n_label_neighbors, 2), dtype=np.int32)

    for i in range(n):
        # Grab the label associated with sample i
        label = y[i]

        # get all the points with the same label but make sure not to include itself
        y_label = y[y == label]
        y_label = y_label[y_label.index != i]

        print("y label length")
        print(len(y_label))
        if len(y_label) < n_label_neighbors:
            print("not enough neighbors")
            print(y_label)

        sampled = y_label.sample(n_label_neighbors)
        print("== sampled ===")
        print(sampled)

        jj = 0
        for sample_idx, label in sampled.items():
            pair_labeled[i * n_label_neighbors + jj][0] = i
            pair_labeled[i * n_label_neighbors + jj][1] = sample_idx
            jj += 1

    print("== pair_labeled ===")
    print(pair_labeled)
    return pair_labeled


@numba.njit(
    "i4[:,:](f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True, cache=True
)
def sample_neighbors_pair(
    X: np.ndarray, scaled_dist: np.ndarray, nbrs: np.ndarray, n_neighbors: int
):
    print("------------------ sampling neighbors")
    n = X.shape[0]
    pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i * n_neighbors + j][0] = i
            pair_neighbors[i * n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit(
    "i4[:,:](i4,f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True, cache=True
)
def sample_neighbors_pair_basis(
    n_basis: int,
    X: np.ndarray,
    scaled_dist: np.ndarray,
    nbrs: np.ndarray,
    n_neighbors: int,
):
    """
    Sample Nearest Neighbor pairs for additional data.
    """

    n = X.shape[0]
    pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i * n_neighbors + j][0] = n_basis + i
            pair_neighbors[i * n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](f4[:,:],i4,i4,i4)", nogil=True, cache=True)
def sample_MN_pair(
    X: np.ndarray,
    n_MN: int,
    random_state: int | None,
    distance_type: int
):
    """
    Sample Mid Near pairs.
    """

    n = X.shape[0]
    pair_MN = np.empty((n * n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            if random_state is not None:
                # Shifting the seed to prevent sampling the same pairs
                np.random.seed(random_state + i * n_MN + jj)

            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_type=DistanceType.EUCLIDEAN
                )
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i * n_MN + jj][0] = i
            pair_MN[i * n_MN + jj][1] = picked
    return pair_MN


# TODO make sure it doesn't collide with labeled ones
@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP, random_state):
    """
    Sample Further pairs.
    """

    print("------------------ sampling further pairs")
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            if random_state is not None:
                np.random.seed(random_state + i * n_FP + k)

            # TODO robert check this is correct
            neighbor_pairs_for_i = pair_neighbors[
                i * n_neighbors : i * n_neighbors + n_neighbors
            ]
            already_sampled_points = np.array(
                [pair[1] for pair in neighbor_pairs_for_i]
            )

            FP_index = sample_FP(n_FP, n, already_sampled_points)
            pair_FP[i * n_FP + k][0] = i
            pair_FP[i * n_FP + k][1] = FP_index[k]
    return pair_FP
