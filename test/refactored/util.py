import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA


def print_verbose(msg, verbose, **kwargs):
    if verbose:
        print(msg, **kwargs)


# TODO this feels wrong because the if statements look flipped
def preprocess_X(
    X: np.ndarray,
    distance_type,
    should_apply_pca,
    verbose,
    random_state,
    x_dim,
    projection_dim,
):
    """
    Preprocess a dataset. This will transform X and normalize it, either through PCA or TruncatedSVD.
    """
    tsvd = None
    pca_solution = False
    if distance_type != "hamming" and x_dim > 100 and should_apply_pca:
        xmin = 0  # placeholder
        xmax = 0  # placeholder
        xmean = np.mean(X, axis=0)
        X -= np.mean(X, axis=0)
        tsvd = TruncatedSVD(n_components=100, random_state=random_state)
        X = tsvd.fit_transform(X)
        pca_solution = True
        print_verbose("Applied PCA, the dimensionality becomes 100", verbose)
    else:
        xmin, xmax = (np.min(X), np.max(X))
        X -= xmin
        X /= xmax
        xmean = np.mean(X, axis=0)
        X -= xmean
        tsvd = PCA(
            n_components=projection_dim, random_state=random_state
        )  # for init only
        tsvd.fit(X)
        print_verbose("X is normalized", verbose)
    return X, pca_solution, tsvd, xmin, xmax, xmean


def preprocess_X_new(X, distance, xmin, xmax, xmean, tsvd, should_apply_pca, verbose):
    """
    Preprocess a new dataset, given the information extracted from the basis.
    """
    _, high_dim = X.shape
    if distance != "hamming" and high_dim > 100 and should_apply_pca:
        X -= xmean  # original xmean
        X = tsvd.transform(X)
        print_verbose(
            "Applied PCA, the dimensionality becomes 100 for new dataset.", verbose
        )
    else:
        X -= xmin
        X /= xmax
        X -= xmean
        print_verbose("X is normalized.", verbose)
    return X


def decide_num_pairs(
    n: int,
    n_near_neighbors: int,
    MN_ratio: float,
    FP_ratio: float,
):
    """
    Decide the number of pairs to sample.
    """
    if n_near_neighbors is None:
        if n <= 10000:
            n_near_neighbors = 10
        else:
            n_near_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
    n_MN = int(round(n_near_neighbors * MN_ratio))
    n_FP = int(round(n_near_neighbors * FP_ratio))
    if n_near_neighbors < 1:
        raise ValueError("The number of nearest neighbors can't be less than 1")
    if n_FP < 1:
        raise ValueError("The number of further points can't be less than 1")

    return n_near_neighbors, n_MN, n_FP
