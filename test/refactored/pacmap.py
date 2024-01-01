import numba
import math
import datetime
import time
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD, PCA

from .util import print_verbose

###
# Helpers
###


def find_weight(w_MN_init, itr):
    """
    Find the corresponding weight given the index of an iteration
    """

    if itr < 100:
        w_MN = (1 - itr / 100) * w_MN_init + itr / 100 * 3.0
        w_labeled = 8.0
        w_neighbors = 2.0
        w_FP = 1.0
    elif itr < 200:
        w_MN = 3.0
        w_labeled = 8
        w_neighbors = 3
        w_FP = 1
    else:
        w_MN = 0.0
        w_labeled = 4
        w_neighbors = 1.0
        w_FP = 1.0
    return w_MN, w_labeled, w_neighbors, w_FP


@numba.njit(
    "f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4[:,:],i4[:,:],f4,f4,f4,f4)",
    parallel=True,
    nogil=True,
    cache=True,
)
def pacmap_grad(
    Y: np.ndarray,
    pair_neighbors: np.ndarray,
    pair_label_neighbors: np.ndarray,
    pair_MN: np.ndarray,
    pair_FP: np.ndarray,
    w_neighbors: float,
    w_labeled: float,
    w_MN: float,
    w_FP: float,
):
    """
    Calculate the gradient for pacmap embedding given the particular set of weights.
    """

    n, dim = Y.shape
    grad = np.zeros((n + 1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(4, dtype=np.float32)

    # TODO @greg: I was playing around with weights of the "labeled" pairs
    # and found that it actually did make a slight difference. I just used the same
    # formula as NN and only adjusted the weight, but I didn't get to trying out
    # perhaps even different formulas for this. I have no idea how to come up with such an
    # algorithm / formula, but I think it could be worth exploring.
    # I've commented  this for now so that it doesn't interfere, but I'm leaving it here for reference.

    # LN
    """ for ll in range(pair_neighbors.shape[0]): """
    """     i = pair_label_neighbors[ll, 0] """
    """     j = pair_label_neighbors[ll, 1] """
    """     d_ij = 1.0 """
    """     for d in range(dim): """
    """         y_ij[d] = Y[i, d] - Y[j, d] """
    """         d_ij += y_ij[d] ** 2 """
    """     loss[0] += w_labeled * (d_ij / (2.0 + d_ij)) """
    """     w1 = w_labeled * (20.0 / (10.0 + d_ij) ** 2) """
    """     for d in range(dim): """
    """         grad[i, d] += w1 * y_ij[d] """
    """         grad[j, d] -= w1 * y_ij[d] """

    # NN
    for t in range(pair_neighbors.shape[0]):
        i = pair_neighbors[t, 0]
        j = pair_neighbors[t, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[0] += w_neighbors * (d_ij / (10.0 + d_ij))
        w1 = w_neighbors * (20.0 / (10.0 + d_ij) ** 2)
        for d in range(dim):
            grad[i, d] += w1 * y_ij[d]
            grad[j, d] -= w1 * y_ij[d]

    # MN
    for tt in range(pair_MN.shape[0]):
        i = pair_MN[tt, 0]
        j = pair_MN[tt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i][d] - Y[j][d]
            d_ij += y_ij[d] ** 2
        loss[1] += w_MN * d_ij / (10000.0 + d_ij)
        w = w_MN * 20000.0 / (10000.0 + d_ij) ** 2
        for d in range(dim):
            grad[i, d] += w * y_ij[d]
            grad[j, d] -= w * y_ij[d]

    # FP
    for ttt in range(pair_FP.shape[0]):
        i = pair_FP[ttt, 0]
        j = pair_FP[ttt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[2] += w_FP * 1.0 / (1.0 + d_ij)
        w1 = w_FP * 2.0 / (1.0 + d_ij) ** 2
        for d in range(dim):
            grad[i, d] -= w1 * y_ij[d]
            grad[j, d] += w1 * y_ij[d]

    grad[-1, 0] = loss.sum()
    return grad


@numba.njit("f4[:,:](f4[:,:],i4[:,:],f4)", parallel=True, nogil=True, cache=True)
def pacmap_grad_fit(Y: np.ndarray, pair_XP, w_neighbors: float):
    """
    Calculate the gradient for pacmap embedding given the particular set of weights.
    """

    n, dim = Y.shape
    grad = np.zeros((n + 1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(4, dtype=np.float32)
    # For case where extra samples are added to the dataset
    for tx in range(pair_XP.shape[0]):
        i = pair_XP[tx, 0]
        j = pair_XP[tx, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[3] += w_neighbors * (d_ij / (10.0 + d_ij))
        w1 = w_neighbors * (20.0 / (10.0 + d_ij) ** 2)
        # w1 = 1. * 2./(1. + d_ij) ** 2 # original formula
        for d in range(dim):
            # just compute the gradient for new point
            grad[i, d] += w1 * y_ij[d]

    grad[-1, 0] = loss.sum()
    return grad


@numba.njit(
    "Tuple((f4[:,:],f4[:,:],f4[:,:]))(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4,f4,i4)",
    parallel=True,
    nogil=True,
    cache=True,
)
def update_embedding_adam(
    Y: np.ndarray,
    grad,
    m: np.ndarray,
    v: np.ndarray,
    beta1: float,
    beta2: float,
    learning_rate: float,
    itr: int,
):
    """
    Update the embedding with the gradient
    """
    Y_copy = np.copy(Y).astype(np.float32)
    m_copy = np.copy(m).astype(np.float32)
    v_copy = np.copy(v).astype(np.float32)

    n, dim = Y.shape
    lr_t = (
        learning_rate * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
    )
    for i in numba.prange(n):
        for d in numba.prange(dim):
            m_copy[i][d] += (1 - beta1) * (grad[i][d] - m[i][d])
            v_copy[i][d] += (1 - beta2) * (grad[i][d] ** 2 - v[i][d])
            Y_copy[i][d] -= lr_t * m[i][d] / (math.sqrt(v[i][d]) + 1e-7)

    return Y_copy, m_copy, v_copy


#
# PACMAP
#
def pacmap(
    X: np.ndarray,
    y,
    n_dims: int,
    pair_neighbors: np.ndarray,
    pair_label_neighbors: np.ndarray | None,
    pair_MN: np.ndarray,
    pair_FP: np.ndarray,
    learning_rate: float,
    num_iters: int,
    Yinit: str,
    verbose: bool,
    should_use_intermediate: bool,
    inter_snapshots: list[int],
    using_pca_solution: bool,
    random_state: int | None,
    tsvd: TruncatedSVD | PCA,
):
    start_time = time.time()
    n, _ = X.shape

    if should_use_intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32
        )
    else:
        intermediate_states = None

    # Initialize the embedding
    if Yinit == "pca":
        if using_pca_solution:
            Y = 0.01 * X[:, :n_dims]
        else:
            Y = 0.01 * tsvd.transform(X).astype(np.float32)
    elif Yinit == "random":  # random or hamming distance
        if random_state is not None:
            np.random.seed(random_state)
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    # TODO the yInit is kinda wrong
    else:  # user_supplied matrix
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = scaler.transform(Yinit) * 0.0001

    # Initialize parameters for optimizer
    w_MN_init = 1000.0
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    itr_ind = 0
    if (
        should_use_intermediate
        and intermediate_states is not None
        and inter_snapshots[0] == 0
    ):
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose((pair_neighbors.shape, pair_MN.shape, pair_FP.shape), verbose)

    for itr in range(num_iters):
        w_MN, w_labeled, w_neighbors, w_FP = find_weight(w_MN_init, itr)

        grad = pacmap_grad(
            Y,
            pair_neighbors,
            pair_label_neighbors,
            pair_MN,
            pair_FP,
            w_neighbors,
            w_labeled,
            w_MN,
            w_FP,
        )

        # We've put the loss sum at the -1, 0 position in the above calculation.
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")

        Y, m, v = update_embedding_adam(Y, grad, m, v, beta1, beta2, learning_rate, itr)

        if should_use_intermediate and intermediate_states is not None:
            if (itr + 1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = time.time() - start_time
    print_verbose(f"Elapsed time: {elapsed:.2f}s", verbose)

    return (
        Y,
        intermediate_states,
        pair_neighbors,
        pair_label_neighbors,
        pair_MN,
        pair_FP,
    )


def pacmap_fit(
    X: np.ndarray,
    # TODO
    embedding,
    n_dims: int,
    pair_XP: np.ndarray,
    learning_rate: float,
    num_iters: int,
    # TODO
    Yinit,
    verbose: bool,
    should_use_intermediate: bool,
    inter_snapshots: list[int],
    random_state: int | None,
    tsvd: TruncatedSVD | PCA,
    using_pca_solution=False,
):
    """
    PaCMAP optimization for new data.
    """

    start_time = time.time()
    n, _ = X.shape

    if should_use_intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32
        )
    else:
        intermediate_states = None

    # Initialize the embedding
    if Yinit is None or Yinit == "pca":
        if using_pca_solution:
            Y = np.concatenate([embedding, 0.01 * X[:, :n_dims]])
        else:
            Y = np.concatenate([embedding, 0.01 * tsvd.transform(X).astype(np.float32)])

    elif Yinit == "random":
        if random_state is not None:
            np.random.seed(random_state)
        Y = np.concatenate(
            [
                embedding,
                0.0001 * np.random.normal(size=[X.shape[0], n_dims]).astype(np.float32),
            ]
        )
    else:  # user_supplied matrix
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = np.concatenate([embedding, scaler.transform(Yinit) * 0.0001])

    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    itr_ind = 0
    if (
        should_use_intermediate
        and intermediate_states is not None
        and inter_snapshots[0] == 0
    ):
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(pair_XP.shape, verbose)

    for itr in range(num_iters):
        if itr < 100:
            w_neighbors = 2.0
        elif itr < 200:
            w_neighbors = 3.0
        else:
            w_neighbors = 1.0

        grad = pacmap_grad_fit(Y, pair_XP, w_neighbors)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        Y, m, v = update_embedding_adam(Y, grad, m, v, beta1, beta2, learning_rate, itr)

        if should_use_intermediate and intermediate_states is not None:
            if (itr + 1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
    print_verbose("Elapsed time: %s" % (elapsed), verbose)
    return Y, intermediate_states
