__all__ = [
    "FunctionSpace",
    "PowerSeries",
    "Chebyshev",
    "GRF",
    "GRF_KL",
    "GRF2D",
    "wasserstein2",
]

import abc

import numpy as np
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp

from .. import config


class FunctionSpace(abc.ABC):
    """Function space base class.

    Example:

        .. code-block:: python

            space = dde.data.GRF()
            feats = space.random(10)
            xs = np.linspace(0, 1, num=100)[:, None]
            y = space.eval_batch(feats, xs)
    """

    @abc.abstractmethod
    def random(self, size):
        """Generate feature vectors of random functions.

        Args:
            size (int): The number of random functions to generate.

        Returns:
            A NumPy array of shape (`size`, n_features).
        """

    @abc.abstractmethod
    def eval_one(self, feature, x):
        """Evaluate the function at one point.

        Args:
            feature: The feature vector of the function to be evaluated.
            x: The point to be evaluated.

        Returns:
            float: The function value at `x`.
        """

    @abc.abstractmethod
    def eval_batch(self, features, xs):
        """Evaluate a list of functions at a list of points.

        Args:
            features: A NumPy array of shape (n_functions, n_features). A list of the
                feature vectors of the functions to be evaluated.
            xs: A NumPy array of shape (n_points, dim). A list of points to be
                evaluated.

        Returns:
            A NumPy array of shape (n_functions, n_points). The values of
            different functions at different points.
        """


class PowerSeries(FunctionSpace):
    r"""Power series.

    p(x) = \sum_{i=0}^{N-1} a_i x^i

    Args:
        N (int)
        M (float): `M` > 0. The coefficients a_i are randomly sampled from [-`M`, `M`].
    """

    def __init__(self, N=100, M=1):
        self.N = N
        self.M = M

    def random(self, size):
        return 2 * self.M * np.random.rand(size, self.N) - self.M

    def eval_one(self, feature, x):
        return np.dot(feature, x ** np.arange(self.N))

    def eval_batch(self, features, xs):
        mat = np.ones((self.N, len(xs)))
        for i in range(1, self.N):
            mat[i] = np.ravel(xs ** i)
        return np.dot(features, mat)


class Chebyshev(FunctionSpace):
    r"""Chebyshev polynomial.

    p(x) = \sum_{i=0}^{N-1} a_i T_i(x),
    where T_i is Chebyshev polynomial of the first kind.
    Note: The domain of x is scaled from [-1, 1] to [0, 1].

    Args:
        N (int)
        M (float): `M` > 0. The coefficients a_i are randomly sampled from [-`M`, `M`].
    """

    def __init__(self, N=100, M=1):
        self.N = N
        self.M = M

    def random(self, size):
        return 2 * self.M * np.random.rand(size, self.N) - self.M

    def eval_one(self, feature, x):
        return np.polynomial.chebyshev.chebval(2 * x - 1, feature)

    def eval_batch(self, features, xs):
        return np.polynomial.chebyshev.chebval(2 * np.ravel(xs) - 1, features.T)


class GRF(FunctionSpace):
    """Gaussian random field (Gaussian process) in 1D.

    The random sampling algorithm is based on Cholesky decomposition of the covariance
    matrix.

    Args:
        T (float): `T` > 0. The domain is [0, `T`].
        kernel (str): Name of the kernel function. "RBF" (radial-basis function kernel,
            squared-exponential kernel, Gaussian kernel), "AE"
            (absolute exponential kernel), or "ExpSineSquared" (Exp-Sine-Squared kernel,
            periodic kernel).
        length_scale (float): The length scale of the kernel.
        N (int): The size of the covariance matrix.
        interp (str): The interpolation to interpolate the random function. "linear",
            "quadratic", or "cubic".
    """

    def __init__(self, T=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        elif kernel == "ExpSineSquared":
            K = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=T)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size):
        u = np.random.randn(self.N, size)
        return np.dot(self.L, u).T

    def eval_one(self, feature, x):
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), feature)
        f = interpolate.interp1d(
            np.ravel(self.x), feature, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_batch(self, features, xs):
        if self.interp == "linear":
            return np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])
        res = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(xs).T,
            features,
        )
        return np.vstack(list(res)).astype(config.real(np))


class GRF_KL(FunctionSpace):
    """Gaussian random field (Gaussian process) in 1D.

    The random sampling algorithm is based on truncated Karhunen-Loeve (KL) expansion.

    Args:
        T (float): `T` > 0. The domain is [0, `T`].
        kernel (str): The kernel function. "RBF" (radial-basis function) or "AE"
            (absolute exponential).
        length_scale (float): The length scale of the kernel.
        num_eig (int): The number of eigenfunctions in KL expansion to be kept.
        N (int): Each eigenfunction is discretized at `N` points in [0, `T`].
        interp (str): The interpolation to interpolate the random function. "linear",
            "quadratic", or "cubic".
    """

    def __init__(
        self, T=1, kernel="RBF", length_scale=1, num_eig=10, N=100, interp="cubic"
    ):
        if not np.isclose(T, 1):
            raise ValueError("GRF_KL only supports T = 1.")

        self.num_eig = num_eig
        if kernel == "RBF":
            kernel = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            kernel = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        eigval, eigvec = eig(kernel, num_eig, N, eigenfunction=True)
        eigvec *= eigval ** 0.5
        x = np.linspace(0, T, num=N)
        self.eigfun = [
            interpolate.interp1d(x, y, kind=interp, copy=False, assume_sorted=True)
            for y in eigvec.T
        ]

    def bases(self, sensors):
        """Evaluate the eigenfunctions at a list of points `sensors`."""
        return np.array([np.ravel(f(sensors)) for f in self.eigfun])

    def random(self, size):
        return np.random.randn(size, self.num_eig)

    def eval_one(self, feature, x):
        eigfun = [f(x) for f in self.eigfun]
        return np.sum(eigfun * feature)

    def eval_batch(self, features, xs):
        eigfun = np.array([np.ravel(f(xs)) for f in self.eigfun])
        return np.dot(features, eigfun)


class GRF2D(FunctionSpace):
    """Gaussian random field in [0, 1]x[0, 1].

    The random sampling algorithm is based on Cholesky decomposition of the covariance
    matrix.

    Args:
        kernel (str): The kernel function. "RBF" (radial-basis function) or "AE"
            (absolute exponential).
        length_scale (float): The length scale of the kernel.
        N (int): The size of the covariance matrix.
        interp (str): The interpolation to interpolate the random function. "linear" or
            "splinef2d".

    Example:

        .. code-block:: python

            space = dde.data.GRF2D(length_scale=0.1)
            features = space.random(3)
            x = np.linspace(0, 1, num=500)
            y = np.linspace(0, 1, num=500)
            xv, yv = np.meshgrid(x, y)
            sensors = np.vstack((np.ravel(xv), np.ravel(yv))).T
            u = space.eval_batch(features, sensors)
            for ui in u:
                plt.figure()
                plt.imshow(np.reshape(ui, (len(y), len(x))))
                plt.colorbar()
            plt.show()
    """

    def __init__(self, kernel="RBF", length_scale=1, N=100, interp="splinef2d"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        xv, yv = np.meshgrid(self.x, self.y)
        self.X = np.vstack((np.ravel(xv), np.ravel(yv))).T
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.X)
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N ** 2))

    def random(self, size):
        u = np.random.randn(self.N ** 2, size)
        return np.dot(self.L, u).T

    def eval_one(self, feature, x):
        y = np.reshape(feature, (self.N, self.N))
        return interpolate.interpn((self.x, self.y), y, x, method=self.interp)[0]

    def eval_batch(self, features, xs):
        points = (self.x, self.y)
        ys = np.reshape(features, (-1, self.N, self.N))
        res = map(lambda y: interpolate.interpn(points, y, xs, method=self.interp), ys)
        return np.vstack(list(res))


def wasserstein2(space1, space2):
    """Compute 2-Wasserstein (W2) metric to measure the distance between two ``GRF``."""
    return (
        np.trace(space1.K + space2.K - 2 * linalg.sqrtm(space1.K @ space2.K)) ** 0.5
        / space1.N ** 0.5
    )


def eig(kernel, num, Nx, eigenfunction=True):
    """Compute the eigenvalues and eigenfunctions of a kernel function in [0, 1]."""
    h = 1 / (Nx - 1)
    c = kernel(np.linspace(0, 1, num=Nx)[:, None])[0] * h
    A = np.empty((Nx, Nx))
    for i in range(Nx):
        A[i, i:] = c[: Nx - i]
        A[i, i::-1] = c[: i + 1]
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5

    if not eigenfunction:
        return np.flipud(np.sort(np.real(np.linalg.eigvals(A))))[:num]

    eigval, eigvec = np.linalg.eig(A)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    idx = np.flipud(np.argsort(eigval))[:num]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    for i in range(num):
        eigvec[:, i] /= np.trapz(eigvec[:, i] ** 2, dx=h) ** 0.5
    return eigval, eigvec
