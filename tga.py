import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import fast_dot, norm
from sklearn.utils import as_float_array, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted


def _trimmed_mean_1d(arr, k):
    """Calculate trimmed mean on a 1d array.

    Trim values largest than the k'th largest value or smaller than the k'th
    smallest value

    Parameters
    ----------
    arr: ndarray, shape (n,)
        The one-dimensional input array to perform trimmed mean on

    k: int
        The thresholding order for trimmed mean

    Returns
    -------
    trimmed_mean: float
        The trimmed mean calculated
    """
    kth_smallest = np.partition(arr, k)[k-1]
    kth_largest = -np.partition(-arr, k)[k-1]

    cnt = 0
    summation = 0.0
    for elem in arr:
        if elem >= kth_smallest and elem <= kth_largest:
            cnt += 1
            summation += elem
    return summation / cnt


def _trimmed_mean(X, trim_proportion):
    """Calculate trimmed mean on each column of input matrix

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_features)
        The input matrix to perform trimmed mean on

    trim_proportion: float
        The proportion of trim. Largest and smallest 'trim_proportion' are
        trimmed when calculating the mean.

    Returns
    -------
    trimmed_mean: ndarray, shape (n_features,)
        The trimmed mean calculated on each column
    """
    n_samples, n_features = X.shape
    n_trim = int(n_samples * trim_proportion)
    return np.apply_along_axis(_trimmed_mean_1d, 0, X, k=n_trim)


def _reorth(basis, target, rows=None, alpha=0.5):
    """Reorthogonalize a vector using iterated Gram-Schmidt

    Parameters
    ----------
    basis: ndarray, shape (n_features, n_basis)
        The matrix whose rows are a set of basis to reorthogonalize against

    target: ndarray, shape (n_features,)
        The target vector to be reorthogonalized

    rows: {array-like, None}, default None
        Indices of rows from basis to use. Use all if None

    alpha: float, default 0.5
        Parameter for determining whether to do a second reorthogonalization.

    Returns
    -------
    reorthed_target: ndarray, shape (n_features,)
        The reorthogonalized vector
    """
    if rows is not None:
        basis = basis[rows]
    norm_target = norm(target)

    norm_target_old = 0
    n_reorth = 0

    while norm_target < alpha * norm_target_old or n_reorth == 0:
        for row in basis:
            t = fast_dot(row, target)
            target = target - t * row

        norm_target_old = norm_target
        norm_target = norm(target)
        n_reorth += 1

        if n_reorth > 4:
            # target in span(basis) => accpet target = 0
            target = np.zeros(basis.shape[0])
            break

    return target


class TGA(BaseEstimator, TransformerMixin):
    """Trimmed Grassmann Average as robust PCA

    Implementation of Trimmed Grassmann Average by Hauberg S et al.

    Parameters
    ----------
    n_components: int, optional, default None
        Maximum number of components to keep. When not given or None, this
        is set to n_features (the second dimension of the training data).

    trim_proportion: float, default 0.5
        The proportion with resepct to n_samples to trim when calculating

    copy: bool, default True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    tol: float, default 1e-5
        Tolerance for stopping criterion of grassmann average

    centering: {'mean', 'median'}, default 'mean'
        Whether to center the data with empirical mean or median

    random_state: int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.

    Attributes
    ----------
    components_: array, [n_components, n_features]
        Top `n_components` principle components.

    center_: array, [n_features]
        Per-feature empirical mean or median according to value of 'centering',
        estimated from the training set.

    References
    ----------
    Hauberg, Soren, Aasa Feragen, and Michael J. Black. "Grassmann averages
        for scalable robust PCA." Computer Vision and Pattern Recognition
        (CVPR), 2014 IEEE Conference on. IEEE, 2014.
    """
    def __init__(self, n_components=None, trim_proportion=0.5, copy=True,
                 tol=1e-5, centering='mean', random_state=None):
        self.n_components = n_components
        self.trim_proportion = trim_proportion
        self.copy = copy
        self.tol = tol
        self.centering = centering
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit(X)
        if self.copy and self.center_ is not None:
            X = X - self.center_
        return fast_dot(X, self.components_.T)

    def _fit(self, X):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        """
        if self.trim_proportion < 0 or self.trim_proportion > 0.5:
            raise ValueError('`trim_proportion` must be between 0 and 0.5,'
                             ' got %s.' % self.trim_proportion)

        rng = check_random_state(self.random_state)
        X = check_array(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        # Center data
        if self.centering == 'mean':
            self.center_ = np.mean(X, axis=0)
        elif self.centering == 'median':
            self.center_ = np.median(X, axis=0)
        else:
            raise ValueError("`centering` must be 'mean' or 'median', "
                             "got %s" % self.centering)
        X -= self.center_

        if self.n_components is None:
            n_components = X.shape[1]
        elif not 0 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (self.n_components, n_features))
        else:
            n_components = self.n_components

        self.components_ = np.empty((n_components, n_features))
        for k in range(n_components):
            # compute k'th principle component
            mu = rng.rand(n_features) - 0.5
            mu = mu / norm(mu)

            # initialize using a few EM iterations
            for i in range(3):
                dots = fast_dot(X, mu)
                mu = fast_dot(dots.T, X)
                mu = mu / norm(mu)

            # grassmann average
            for i in range(n_samples):
                prev_mu = mu
                dot_signs = np.sign(fast_dot(X, mu))
                mu = _trimmed_mean(X * dot_signs[:, np.newaxis],
                                   self.trim_proportion)
                mu = mu / norm(mu)

                if np.max(np.abs(mu - prev_mu)) < self.tol:
                    break

            # store the estimated vector and possibly re-orthonormalize
            if k > 0:
                mu = _reorth(self.components_[:k-1], mu)
                mu = mu / norm(mu)

            self.components_[k] = mu

            if k < n_components - 1:
                X = X - fast_dot(fast_dot(X, mu)[:, np.newaxis],
                                 mu[np.newaxis, :])

    def transform(self, X, y=None):
        """Apply dimensionality reduction on X.

        X is projected on the principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'center_')

        X = check_array(X)
        if self.center_ is not None:
            X = X - self.center_
        X_transformed = fast_dot(X, self.components_.T)
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space, i.e.,
        return an input X_original whose transform would be X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original: array-like, shape (n_samples, n_features)
        """
        check_is_fitted(self, 'center_')

        X_original = fast_dot(X, self.components_)
        if self.center_ is not None:
            X_original = X_original + self.center_
        return X_original
