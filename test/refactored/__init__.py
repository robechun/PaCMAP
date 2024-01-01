from sklearn.base import BaseEstimator
import warnings
import numpy as np
from .util import print_verbose, preprocess_X, preprocess_X_new, decide_num_pairs
from .pacmap import pacmap, pacmap_fit
from .pair_helpers import sample_pairs, generate_extra_pair_basis
from .distance_helpers import DistanceType

global _RANDOM_STATE
_RANDOM_STATE = None


class PaCMAP(BaseEstimator):
    """Pairwise Controlled Manifold Approximation.

    Maps high-dimensional dataset to a low-dimensional embedding.
    This class inherits the sklearn BaseEstimator, and we tried our best to
    follow the sklearn api. For details of this method, please refer to our publication:
    https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf

    Parameters
    ---------
    n_components: int, default=2
        Dimensions of the embedded space. We recommend to use 2 or 3.

    n_near_neighbors: int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.

    MN_ratio: float, default=0.5
        Ratio of mid near pairs to nearest neighbor pairs (e.g. n_near_neighbors=10, MN_ratio=0.5 --> 5 Mid near pairs)
        Mid near pairs are used for global structure preservation.

    FP_ratio: float, default=2.0
        Ratio of further pairs to nearest neighbor pairs (e.g. n_near_neighbors=10, FP_ratio=2 --> 20 Further pairs)
        Further pairs are used for both local and global structure preservation.

    pair_neighbors: numpy.ndarray, optional
        Nearest neighbor pairs constructed from a previous run or from outside functions.

    pair_MN: numpy.ndarray, optional
        Mid near pairs constructed from a previous run or from outside functions.

    pair_FP: numpy.ndarray, optional
        Further pairs constructed from a previous run or from outside functions.

    distance_type: string, default="euclidean"
        Distance metric used for high-dimensional space. Allowed metrics include euclidean, manhattan, angular, hamming.

    learning_rate: float, default=1.0
        Learning rate of the Adam optimizer for embedding.

    num_iters: int, default=450
        Number of iterations for the optimization of embedding.
        Due to the stage-based nature, we suggest this parameter to be greater than 250 for all three stages to be utilized.

    verbose: bool, default=False
        Whether to print additional information during initialization and fitting.

    should_apply_pca: bool, default=True
        Whether to apply PCA on the data before pair construction.

    intermediate: bool, default=False
        Whether to return intermediate state of the embedding during optimization.
        If True, returns a series of embedding during different stages of optimization.

    intermediate_snapshots: list[int], optional
        The index of step where an intermediate snapshot of the embedding is taken.
        If intermediate sets to True, the default value will be [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]

    random_state: int, optional
        Random state for the pacmap instance.
        Setting random state is useful for repeatability.

    save_tree: bool, default=False
        Whether to save the annoy index tree after finding the nearest neighbor pairs.
        Default to False for memory saving. Setting this option to True can make `transform()` method faster.
    """

    n_components: int
    n_near_neighbors: int
    MN_ratio: float
    FP_ratio: float
    pair_neighbors: np.ndarray | None

    def __init__(
        self,
        n_components=2,
        n_near_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        pair_neighbors=None,
        pair_MN=None,
        pair_FP=None,
        distance_type=DistanceType.EUCLIDEAN,
        learning_rate=1.0,
        num_iters=450,
        verbose=False,
        should_apply_pca=True,
        intermediate=False,
        intermediate_snapshots=[
            0,
            10,
            30,
            60,
            100,
            120,
            140,
            170,
            200,
            250,
            300,
            350,
            450,
        ],
        random_state=None,
        save_tree=False,
    ):
        self.n_components = n_components
        self.n_near_neighbors = n_near_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.pair_neighbors = pair_neighbors

        # TODO this should be parameter passed in?
        self.num_label_neighbors = None
        self.pair_label_neighbors = None

        self.pair_MN = pair_MN
        self.pair_FP = pair_FP
        self.distance_type = distance_type
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.should_apply_pca = should_apply_pca
        self.verbose = verbose
        self.intermediate = intermediate
        self.tree = None
        self.save_tree = save_tree
        self.intermediate_snapshots = intermediate_snapshots

        global _RANDOM_STATE
        if random_state is not None:
            assert isinstance(random_state, int)
            self.random_state = random_state
            _RANDOM_STATE = random_state  # Set random state for numba functions
            warnings.warn(f"Warning: random state is set to {_RANDOM_STATE}")
        else:
            try:
                if _RANDOM_STATE is not None:
                    warnings.warn(f"Warning: random state is removed")
            except NameError:
                pass
            self.random_state = 0
            _RANDOM_STATE = None  # Reset random state

        if self.n_components < 2:
            raise ValueError("The number of projection dimensions must be at least 2.")
        if self.learning_rate <= 0:
            raise ValueError("The learning rate must be larger than 0.")
        if self.distance_type == "hamming" and should_apply_pca:
            warnings.warn(
                "should_apply_pca = True for Hamming distance. This option will be ignored."
            )
        if not self.should_apply_pca:
            warnings.warn(
                "Running ANNOY Indexing on high-dimensional data. Nearest-neighbor search may be slow!"
            )

    def fit(self, X: np.ndarray, y, init: str = "pca", save_pairs=True):
        """Projects a high dimensional dataset into a low-dimensional embedding, without returning the output.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected.
            An embedding will get created based on parameters of the PaCMAP instance.

        # TODO robert shape here
        y: i -> [i, label]

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        """

        # Preprocess the dataset
        X = np.copy(X).astype(np.float32)
        # n is # of "points" or "vectors". Dim is the dimension of each vector
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
        (
            X,
            self.using_pca_solution,
            self.tsvd_transformer,
            self.xmin,
            self.xmax,
            self.xmean,
        ) = preprocess_X(
            X,
            self.distance_type,
            self.should_apply_pca,
            self.verbose,
            self.random_state,
            dim,
            self.n_components,
        )

        #
        # Deciding the number of pairs
        #
        self.n_near_neighbors, self.n_MN, self.n_FP = decide_num_pairs(
            n, self.n_near_neighbors, self.MN_ratio, self.FP_ratio
        )
        print_verbose(
            "PaCMAP(n_near_neighbors={}, n_MN={}, n_FP={}, distance_type={}, "
            "learning_rate={}, n_iters={}, should_apply_pca={}, opt_method='adam', "
            "verbose={}, intermediate={}, seed={})".format(
                self.n_near_neighbors,
                self.n_MN,
                self.n_FP,
                self.distance_type,
                self.learning_rate,
                self.num_iters,
                self.should_apply_pca,
                self.verbose,
                self.intermediate,
                _RANDOM_STATE,
            ),
            self.verbose,
        )

        #
        # Sample pairs
        #
        (
            self.pair_neighbors,
            self.pair_label_neighbors,
            self.pair_MN,
            self.pair_FP,
            self.tree,
        ) = sample_pairs(
            X,
            y,
            self.pair_neighbors,
            self.pair_label_neighbors,
            self.pair_MN,
            self.pair_FP,
            self.n_near_neighbors,
            self.n_MN,
            self.n_FP,
            self.distance_type,
            self.tree,
            self.save_tree,
            self.verbose,
        )
        self.num_instances = X.shape[0]
        self.num_dimensions = X.shape[1]
        print_verbose("Done sampling pairs", self.verbose)

        #
        # Initialize and Optimize the embedding
        #
        (
            self.embedding_,
            self.intermediate_states,
            self.pair_neighbors,
            self.pair_label_neighbors,
            self.pair_MN,
            self.pair_FP,
        ) = pacmap(
            X=X,
            y=y,
            n_dims=self.n_components,
            pair_neighbors=self.pair_neighbors,
            pair_label_neighbors=self.pair_label_neighbors,
            pair_MN=self.pair_MN,
            pair_FP=self.pair_FP,
            learning_rate=self.learning_rate,
            num_iters=self.num_iters,
            Yinit=init,
            verbose=self.verbose,
            should_use_intermediate=self.intermediate,
            inter_snapshots=self.intermediate_snapshots,
            using_pca_solution=self.using_pca_solution,
            random_state=_RANDOM_STATE,
            tsvd=self.tsvd_transformer,
        )

        if not save_pairs:
            self.del_pairs()

        return self

    def fit_transform(self, X: np.ndarray, y=None, init: str = "pca", save_pairs=True):
        """Projects a high dimensional dataset into a low-dimensional embedding and return the embedding.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected.
            An embedding will get created based on parameters of the PaCMAP instance.

        y: i -> [i, label]

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        """

        self.fit(X, y, init, save_pairs)
        if self.intermediate:
            return self.intermediate_states
        else:
            return self.embedding_

    def transform(self, X: np.ndarray, basis=None, init=None, save_pairs=True):
        """Projects a high dimensional dataset into existing embedding space and return the embedding.
        Warning: In the current version of implementation, the `transform` method will treat the input as an
        additional dataset, which means the same point could be mapped into a different place.

        Parameters
        ---------
        X: numpy.ndarray
            The new high-dimensional dataset that is being projected.
            An embedding will get created based on parameters of the PaCMAP instance.

        basis: numpy.ndarray
            The original dataset that have already been applied during the `fit` or `fit_transform` process.
            If `save_tree == False`, then the basis is required to reconstruct the ANNOY tree instance.
            If `save_tree == True`, then it's unnecessary to provide the original dataset again.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            The PCA instance will be the same one that was applied to the original dataset during the `fit` or `fit_transform` process.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        """

        # Preprocess the data
        X = np.copy(X).astype(np.float32)
        X = preprocess_X_new(
            X,
            self.distance_type,
            self.xmin,
            self.xmax,
            self.xmean,
            self.tsvd_transformer,
            self.should_apply_pca,
            self.verbose,
        )
        if basis is not None and self.tree is None:
            basis = np.copy(basis).astype(np.float32)
            basis = preprocess_X_new(
                basis,
                self.distance_type,
                self.xmin,
                self.xmax,
                self.xmean,
                self.tsvd_transformer,
                self.should_apply_pca,
                self.verbose,
            )

        #
        # Sample pairs
        #
        self.pair_XP = generate_extra_pair_basis(
            basis, X, self.n_near_neighbors, self.tree, self.distance_type, self.verbose
        )
        if not save_pairs:
            self.pair_XP = None

        #
        # Initialize and Optimize the embedding
        #
        Y, intermediate_states = pacmap_fit(
            X,
            self.embedding_,
            self.n_components,
            self.pair_XP,
            self.learning_rate,
            self.num_iters,
            init,
            self.verbose,
            self.intermediate,
            self.intermediate_snapshots,
            _RANDOM_STATE,
            self.tsvd_transformer,
            self.using_pca_solution,
        )
        if self.intermediate:
            return intermediate_states
        else:
            return Y[self.embedding_.shape[0] :, :]

    def del_pairs(self):
        """
        Delete stored pairs.
        """
        self.pair_neighbors = None
        self.pair_MN = None
        self.pair_FP = None
        return self
