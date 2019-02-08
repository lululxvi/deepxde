from __future__ import division

import numpy as np
import scipy.spatial.distance


def cross_entropy(p, q):
    return -np.sum(p * np.log(q), axis=1)


def softmax(X):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(X - np.max(X, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


def uniformly_continuous_delta(X, Y, eps):
    """Compute the supremum of delta in uniformly continuous
    X: N x d, equispaced points
    """
    if X.shape[1] == 1:
        # 1d equispaced points
        dx = np.linalg.norm(X[1] - X[0])
        n = len(Y)
        k = 1
        while True:
            if np.any(np.linalg.norm(Y[:n-k] - Y[k:], ord=np.inf, axis=1) >= eps):
                return (k - 0.5) * dx
            k += 1
    else:
        dX = scipy.spatial.distance.pdist(X, 'euclidean')
        dY = scipy.spatial.distance.pdist(Y, 'chebyshev')
        delta = np.min(dX)
        dx = delta / 2
        while True:
            if np.max(dY[dX <= delta]) >= eps:
                return delta - dx / 2
            delta += dx
