"""External utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Pool

import numpy as np
import scipy.spatial.distance
from sklearn import preprocessing


class PointSet(object):
    """A set of points.

    Args:
        points: A NumPy array of shape (`N`, `dx`). A list of `dx`-dim points.
    """

    def __init__(self, points):
        self.points = np.array(points)

    def inside(self, x):
        """Returns ``True`` if `x` is in this set of points, otherwise, returns
        ``False``.

        Args:
            x: A NumPy array. A single point, or a list of points.

        Returns:
            If `x` is a single point, returns ``True`` or ``False``. If `x` is a list of
                points, returns a list of ``True`` or ``False``.
        """
        if x.ndim == 1:
            # A single point
            return np.any(np.all(np.isclose(x, self.points), axis=1))
        if x.ndim == 2:
            # A list of points
            return np.any(
                np.all(np.isclose(x[:, np.newaxis, :], self.points), axis=-1),
                axis=-1,
            )

    def values_to_func(self, values, default_value=0):
        """Convert the pairs of points and values to a callable function.

        Args:
            values: A NumPy array of shape (`N`, `dy`). `values[i]` is the `dy`-dim
                function value of the `i`-th point in this point set.
            default_value (float): The function value of the points not in this point
                set.

        Returns:
            A callable function. The input of this function should be a NumPy array of
                shape (?, `dx`).
        """

        def func(x):
            pt_equal = np.all(np.isclose(x[:, np.newaxis, :], self.points), axis=-1)
            not_inside = np.logical_not(np.any(pt_equal, axis=-1, keepdims=True))
            return np.matmul(pt_equal, values) + default_value * not_inside

        return func


def apply(func, args=None, kwds=None):
    """Launch a new process to call the function.

    This can be used to clear Tensorflow GPU memory after model execution:
    https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def standardize(X_train, X_test):
    """Standardize features by removing the mean and scaling to unit variance.

    The mean and std are computed from the training data `X_train` using
    `sklearn.preprocessing.StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_,
    and then applied to the testing data `X_test`.

    Args:
        X_train: A NumPy array of shape (n_samples, n_features). The data used to
            compute the mean and standard deviation used for later scaling along the
            features axis.
        X_test: A NumPy array.

    Returns:
        scaler: Instance of ``sklearn.preprocessing.StandardScaler``.
        X_train: Transformed training data.
        X_test: Transformed testing data.
    """
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return scaler, X_train, X_test


def uniformly_continuous_delta(X, Y, eps):
    """Compute the supremum of delta in uniformly continuous.

    Args:
        X: N x d, equispaced points.
    """
    if X.shape[1] == 1:
        # 1d equispaced points
        dx = np.linalg.norm(X[1] - X[0])
        n = len(Y)
        k = 1
        while True:
            if np.any(np.linalg.norm(Y[: n - k] - Y[k:], ord=np.inf, axis=1) >= eps):
                return (k - 0.5) * dx
            k += 1
    else:
        dX = scipy.spatial.distance.pdist(X, "euclidean")
        dY = scipy.spatial.distance.pdist(Y, "chebyshev")
        delta = np.min(dX)
        dx = delta / 2
        while True:
            if np.max(dY[dX <= delta]) >= eps:
                return delta - dx / 2
            delta += dx
