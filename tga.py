import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import fast_dot, norm
from sklearn.utils import as_float_array, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted


def trimmed_mean():
    pass


def reorth():
    pass


class TGA(BaseEstimator, TransformerMixin):
    """Trimmed Grassmann Average as robust PCA"""
    def __init__(self, n_components=None, trim_proportion=0.5, copy=True,
                 tol=1e-5, random_state=None):
        self.n_components = n_components
        self.trim_proportion = trim_proportion
        self.copy = copy
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
        X = self._fit(X)
        return fast_dot(X, self.components_.T)

    def _fit(self, X):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        X : ndarray, shape (n_samples, n_features)
            The input data, copied, centered and whitened when requested.
        -------
        """
        if self.trim_proportion < 0 or self.trim_proportion > 0.5:
            raise ValueError('trim_proportion must be between 0 and 0.5,'
                             ' got %s.' % self.trim_proportion)

        rng = check_random_state(self.random_state)
        X = check_array(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        self.components_ = np.empty(n_features, n_components)
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
                mu = trimmed_mean(X * dot_signs, self.trim_proportion)
                mu = mu / norm(mu)

                if np.max(np.abs(mu - prev_mu)) < self.tol:
                    break

            # store the estimated vector and possibly re-orthonormalize
            if k > 0:
                mu = reorth(self.components_[:, :k-1], mu, 1)
                mu = mu / norm(mu)

            self.components_[:, k] = mu

            if k < n_components - 1:
                X = X - fast_dot(fast_dot(X, mu), mu.T)

        return X

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
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = fast_dot(X, self.components_.T)
        return X_transformed
