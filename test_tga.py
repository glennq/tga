from tga import TGA
from sklearn import datasets
from sklearn.utils.testing import assert_array_almost_equal, assert_equal


iris = datasets.load_iris()


def test_TGA():
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
