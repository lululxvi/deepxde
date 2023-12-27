"""Compute Jacobian matrix."""

from abc import ABC, abstractmethod

from ..backend import backend_name


class Jacobian(ABC):
    """Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J with J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    It is lazy evaluation, i.e., it only computes J[i, j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
    """

    def __init__(self, ys, xs):
        self.ys = ys
        self.xs = xs

        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            self.dim_y = ys.shape[1]
        elif backend_name == "jax":
            # For backend jax, a tuple of a jax array and a callable is passed as one of
            # the arguments, since jax does not support computational graph explicitly.
            # The array is used to control the dimensions and the callable is used to
            # obtain the derivative function, which can be used to compute the
            # derivatives.
            self.dim_y = ys[0].shape[1]
        self.dim_x = xs.shape[1]

        self.J = {}

    @abstractmethod
    def __call__(self, i=None, j=None):
        """Returns (`i`, `j`)th entry J[`i`, `j`].

        - If `i` is ``None``, returns the jth column J[:, `j`].
        - If `j` is ``None``, returns the ith row J[`i`, :], i.e., the gradient of y_i.
        - `i` and `j` cannot be both ``None``.
        """
        if i is None and j is None:
            raise ValueError("i and j cannot be both None.")
        if i is not None and not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))


class Jacobians:
    """Compute multiple Jacobians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self, JacobianClass):
        self.JacobianClass = JacobianClass
        self.Js = {}

    def __call__(self, ys, xs, i=None, j=None):
        # For backend tensorflow and pytorch, self.Js cannot be reused across iteration.
        # For backend pytorch, we need to reset self.Js in each iteration to avoid
        # memory leak.
        #
        # For backend tensorflow, in each iteration, self.Js is reset to {}.
        #
        # Example:
        #
        # mydict = {}
        #
        # @tf.function
        # def f(x):
        #     tf.print(mydict)  # always {}
        #     y = 1 * x
        #     tf.print(hash(y.ref()), hash(x.ref()))  # Doesn't change
        #     mydict[(y.ref(), x.ref())] = 1
        #     tf.print(mydict)
        #
        # for _ in range(2):
        #     x = np.random.random((3, 4))
        #     f(x)
        #
        #
        # For backend pytorch, in each iteration, ys and xs are new tensors
        # converted from np.ndarray, so self.Js will increase over iteration.
        #
        # Example:
        #
        # mydict = {}
        #
        # def f(x):
        #     print(mydict)
        #     y = 1 * x
        #     print(hash(y), hash(x))
        #     mydict[(y, x)] = 1
        #     print(mydict)
        #
        # for i in range(2):
        #     x = np.random.random((3, 4))
        #     x = torch.from_numpy(x)
        #     x.requires_grad_()
        #     f(x)
        if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
            key = (ys.ref(), xs.ref())
        elif backend_name in ["pytorch", "paddle"]:
            key = (ys, xs)
        elif backend_name == "jax":
            key = (id(ys[0]), id(xs))
        if key not in self.Js:
            self.Js[key] = self.JacobianClass(ys, xs)
        return self.Js[key](i, j)

    def clear(self):
        """Clear cached Jacobians."""
        self.Js = {}


def jacobian(ys, xs, i=None, j=None):
    """Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J with J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int or None): `i`th row. If `i` is ``None``, returns the `j`th column
            J[:, `j`].
        j (int or None): `j`th column. If `j` is ``None``, returns the `i`th row
            J[`i`, :], i.e., the gradient of y_i.

    Returns:
        (`i`, `j`)th entry J[`i`, `j`]. `i` and `j` cannot be both ``None``.
    """
    return jacobian._Jacobians(ys, xs, i=i, j=j)


jacobian._Jacobians = Jacobians(Jacobian)
