import numpy as np

from tga import TGA
from sklearn import datasets
from sklearn.utils.testing import assert_array_almost_equal, assert_equal, \
    assert_raises, assert_almost_equal


iris = datasets.load_iris()


def test_tga():
    # TGA on dense arrays
    tga = TGA(n_components=2, random_state=1)
    X = iris.data
    X_r = tga.fit(X).transform(X)
    assert_equal(X_r.shape[1], 2)

    tga = TGA(n_components=2, random_state=1)
    X_r2 = tga.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)

    tga = TGA(random_state=1)
    tga.fit(X)
    X_r = tga.transform(X)
    tga = TGA(random_state=1)
    X_r2 = tga.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)


def test_tga_median():
    # TGA on dense arrays
    tga = TGA(n_components=2, random_state=1, centering='median')
    X = iris.data
    X_r = tga.fit(X).transform(X)
    assert_equal(X_r.shape[1], 2)

    tga = TGA(n_components=2, random_state=1, centering='median')
    X_r2 = tga.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)

    tga = TGA(random_state=1, centering='median')
    tga.fit(X)
    X_r = tga.transform(X)
    tga = TGA(random_state=1, centering='median')
    X_r2 = tga.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)


def test_pca_error():
    X = [[0, 1], [1, 0]]
    for n_components in [-1, 3]:
        assert_raises(ValueError, TGA(n_components).fit, X)
    for trim_p in [-0.1, 0.6]:
        assert_raises(ValueError, TGA(trim_proportion=trim_p).fit, X)
    assert_raises(ValueError, TGA(centering='invalid').fit, X)


def test_tga_check_projection():
    # Test that the projection by RandomizedPCA on dense data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * .1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])

    Yt = TGA(n_components=2, random_state=0).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())

    assert_almost_equal(np.abs(Yt[0][0]), 1., 1)


def test_tga_inverse():
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= .00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    tga = TGA(n_components=2).fit(X)
    Y = tga.transform(X)
    Y_inverse = tga.inverse_transform(Y)
    assert_almost_equal(X, Y_inverse, decimal=3)


def test_tga_check_list():
    # Test that the projection by TGA on list data is correct
    X = [[1.0, 0.0], [0.0, 1.0]]
    X_transformed = TGA(n_components=1,
                        random_state=0).fit(X).transform(X)
    assert_equal(X_transformed.shape, (2, 1))
    assert_almost_equal(X_transformed.mean(), 0.00, 2)
    assert_almost_equal(X_transformed.std(), 0.71, 2)


def test_tga_dim():
    # Check automated dimensionality setting
    rng = np.random.RandomState(0)
    n, p = 100, 5
    X = rng.randn(n, p) * .1
    tga = TGA().fit(X)
    assert_equal(tga.components_.shape[0], 5)
