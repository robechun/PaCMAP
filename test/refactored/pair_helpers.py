import numpy as np

from annoy import AnnoyIndex

from .sampling_helpers import (
    sample_neighbors_pair,
    sample_MN_pair,
    sample_FP_pair,
    sample_neighbors_pair_basis,
    sample_labeled_dist,
    sample_labeled_pair,
)
from .distance_helpers import DistanceType, scale_dist
from .util import print_verbose


def sample_pairs(
    X: np.ndarray,
    y,
    pair_neighbors: np.ndarray | None,
    pair_label_neighbors: np.ndarray | None,
    pair_MN: np.ndarray | None,
    pair_FP: np.ndarray | None,
    n_near_neighbors: int,
    # n_num_label_neighbors: int,
    n_MN: int,
    n_FP: int,
    distance_type: DistanceType,
    tree,
    save_tree: bool,
    verbose: bool,
):
    """
    Sample PaCMAP pairs from the dataset.

    Parameters
    ---------
    X: numpy.ndarray
        The high-dimensional dataset that is being projected.

    y: i -> [i, label]

    save_tree: bool
        Whether to save the annoy index tree after finding the nearest neighbor pairs.
    """
    generated_pair_neighbors = None
    generated_pair_label_neighbors = None
    generated_pair_MN = None
    generated_pair_FP = None
    generated_tree = None

    # Creating pairs
    print_verbose("Finding pairs", verbose)
    if pair_neighbors is None:
        (
            generated_pair_neighbors,
            generated_pair_label_neighbors,
            generated_pair_MN,
            generated_pair_FP,
            generated_tree,
        ) = generate_pairs(
            X,
            y,
            n_near_neighbors,
            n_MN,
            n_FP,
            distance_type,
            verbose,
        )
        print_verbose("Pairs sampled successfully.", verbose)
    elif pair_MN is None and pair_FP is None:
        print_verbose("Using user provided nearest neighbor pairs.", verbose)
        assert pair_neighbors.shape == (
            X.shape[0] * n_near_neighbors,
            2,
        ), "The shape of the user provided nearest neighbor pairs is incorrect."
        (
            generated_pair_neighbors,
            generated_pair_MN,
            generated_pair_FP,
        ) = generate_pairs_no_neighbors(
            X,
            n_near_neighbors,
            n_MN,
            n_FP,
            pair_neighbors,
            distance_type,
            verbose,
        )
        print_verbose("Pairs sampled successfully.", verbose)
    else:
        print_verbose("Using stored pairs.", verbose)

    if not save_tree:
        generated_tree = None

    return (
        generated_pair_neighbors,
        generated_pair_label_neighbors,
        generated_pair_MN,
        generated_pair_FP,
        generated_tree,
    )


def generate_pairs(
    X: np.ndarray,
    y,
    n_near_neighbors: int,
    n_MN: int,
    n_FP: int,
    distance_type: DistanceType = DistanceType.EUCLIDEAN,
    random_state=None,
    verbose=True,
):
    """
    Generate pairs for the dataset.
    """

    n, dim = X.shape
    # sample more neighbors than needed
    n_neighbors_extra = min(n_near_neighbors + 50, n - 1)

    #
    # Get the nearest neighbors
    #
    tree = AnnoyIndex(dim, metric=distance_type.value)
    if random_state is not None:
        tree.set_seed(random_state)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    neighbors = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_neighbors_extra), dtype=np.float32)

    for i in range(n):
        # Get NNs for a given vector (the actual vector as the neighbor).
        nbrs_ = tree.get_nns_by_item(i, n_neighbors_extra + 1)
        neighbors[i, :] = nbrs_[1:]

        # i is row index. Get distances for each neighbor.
        for j in range(n_neighbors_extra):
            knn_distances[i, j] = tree.get_distance(i, neighbors[i, j])

    print_verbose("Found nearest neighbors", verbose)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    print_verbose(f"sig: {sig}", verbose)
    scaled_dist = scale_dist(knn_distances, sig, neighbors)
    print_verbose("Found scaled dist", verbose)
    pair_neighbors = sample_neighbors_pair(X, scaled_dist, neighbors, n_near_neighbors)

    # Sample labeled pairs, if label information exists.
    # TODO n_label_neighbors should be parameter passed in.
    # Right now, same as n_neighbors default value
    if y is not None:
        sampled_labeled_dist = sample_labeled_dist(X, y, n_near_neighbors)
        sig_label = np.maximum(np.mean(sampled_labeled_dist[:, 3:6], axis=1), 1e-10)
        scaled_dist = scale_dist(sampled_labeled_dist, sig_label, neighbors)

        #
        # TODO @greg You can see here that I set the "pair_neighbors" to the sampling of the
        # LABELED pairs. This is to "trick" the algorithm into thinking that the labeled pairs
        # are the nearest neighbors.
        # If you want to try with the extra labeled_neighbors, simply rename the following line of
        # pair_neighbors to pair_label_neighbors, and comment the `pair_label_neighbors = np.zeros(...)` line.
        # (also make sure to uncomment the `ll in range` code block)
        # the `pair_label_neighbors = np.zeros(...)` does _nothing_ at the moment, but is only there to
        # match numba types.
        #
        pair_neighbors = sample_labeled_pair(X, y, 10)
        pair_label_neighbors = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    else:
        pair_label_neighbors = None

    #
    # Get the mid and far pairs.
    #
    # TODO im just using 0 here cause im lazy to figure out the numba types
    pair_MN = sample_MN_pair(X, n_MN, random_state, 0)
    pair_FP = sample_FP_pair(X, pair_neighbors, n_near_neighbors, n_FP, random_state)
    print_verbose("Found mid and far pairs", verbose)


    return pair_neighbors, pair_label_neighbors, pair_MN, pair_FP, tree


def generate_pairs_no_neighbors(
    X: np.ndarray,
    n_near_neighbors: int,
    n_MN: int,
    n_FP: int,
    pair_neighbors: np.ndarray,
    distance_type: DistanceType = DistanceType.EUCLIDEAN,
    random_state=None,
    verbose=True,
):
    """
    Generate mid-near pairs and further pairs for a given dataset.
    This function is useful when the nearest neighbors comes from a given set.
    """

    pair_MN = sample_MN_pair(X, n_MN, random_state, distance_type)
    pair_FP = sample_FP_pair(X, pair_neighbors, n_near_neighbors, n_FP, random_state)
    print_verbose("Found mid and far pairs but no neighbors", verbose)

    return pair_neighbors, pair_MN, pair_FP


def generate_extra_pair_basis(
    basis,
    X: np.ndarray,
    n_neighbors: int,
    tree: AnnoyIndex | None,
    distance_type=DistanceType.EUCLIDEAN,
    random_state=None,
    verbose=True,
):
    """Generate pairs that connects the extra set of data to the fitted basis."""
    npr, dimp = X.shape

    assert (
        basis is not None or tree is not None
    ), "If the annoyindex is not cached, the original dataset must be provided."

    # Build the tree again if not cached
    if tree is None:
        n, dim = basis.shape
        assert (
            dimp == dim
        ), "The dimension of the original dataset is different from the new one's."
        tree = AnnoyIndex(dim, metric=distance_type.value)
        if random_state is not None:
            tree.set_seed(random_state)
        for i in range(n):
            tree.add_item(i, basis[i, :])
        tree.build(20)
    else:
        n = tree.get_n_items()

    n_neighbors_extra = min(n_neighbors + 50, n - 1)
    nbrs = np.zeros((npr, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((npr, n_neighbors_extra), dtype=np.float32)

    for i in range(npr):
        nbrs[i, :], knn_distances[i, :] = tree.get_nns_by_vector(
            X[i, :], n_neighbors_extra, include_distances=True
        )

    print_verbose("Found nearest neighbor", verbose)
    # sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    # print_verbose("Calculated sigma", verbose)

    # Debug
    # print_verbose(f"Sigma is of the scale of {sig.shape}", verbose)
    # print_verbose(f"KNN dist is of shape scale of {knn_distances.shape}", verbose)
    # print_verbose(f"nbrs max: {nbrs.max()}", verbose)

    # scaling the distances is not possible since we don't always track the basis
    # scaled_dist = scale_dist(knn_distances, sig, nbrs)
    print_verbose("Found scaled dist", verbose)

    pair_neighbors = sample_neighbors_pair_basis(n, X, knn_distances, nbrs, n_neighbors)
    return pair_neighbors
