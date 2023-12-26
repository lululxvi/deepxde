"""Compute gradients using forward-mode autodiff."""

__all__ = ["jacobian", "hessian"]

from ..backend import backend_name, jax


class Jacobian:
    """Compute Jacobian matrix J: J[i][j] = dy_i/dx_j, where i = 0, ..., dim_y-1 and
    j = 0, ..., dim_x - 1.

    It is lazy evaluation, i.e., it only computes J[i][j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
    """

    def __init__(self, ys, xs):
        self.ys = ys
        self.xs = xs

        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            self.dim_y = ys.shape[1]
            # TODO: Other backends
            raise NotImplementedError(
                "Backend f{backend_name} doesn't support forward-mode autodiff."
            )
        elif backend_name == "jax":
            # For backend jax, a tuple of a jax array and a callable is passed as one of
            # the arguments, since jax does not support computational graph explicitly.
            # The array is used to control the dimensions and the callable is used to
            # obtain the derivative function, which can be used to compute the
            # derivatives.
            self.dim_y = ys[0].shape[1]
        self.dim_x = xs.shape[1]

        self.J = {}

    def __call__(self, i=0, j=None):
        """Returns J[`i`][`j`]. If `j` is ``None``, returns the gradient of y_i, i.e.,
        J[i]. If `i` is ``None``, returns J[:, j]. `i` and `j` cannot be both ``None``.
        """
        if i is None and j is None:
            raise ValueError("i and j cannot be both None.")
        if i is not None and not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))
        # Computing gradient is not supported in forward mode, unless there is only one input.
        if j is None:
            if self.dim_x == 1:
                j = 0
            else:
                raise NotImplementedError(
                    "Forward-mode autodiff doesn't support computing gradient."
                )
        # Compute J[:, j]
        if j not in self.J:
            if backend_name == "jax":
                # Here, we use jax.jvp to compute the gradient of a function. This is
                # different from TensorFlow and PyTorch that the input of a function is
                # no longer a batch. Instead, it is a single point. Formally, backend
                # jax computes gradients pointwisely and then vectorizes to batch, by
                # jax.vmap. However, computationally, this is in fact done batchwisely
                # and efficiently. It is very important to note that, without jax.vmap,
                # this can only deal with functions whose output is a scalar and input
                # is a single point.
                tangent = jax.numpy.zeros(self.dim_x).at[j].set(1)
                grad_fn = lambda x: jax.jvp(self.ys[1], (x,), (tangent,))[1]
                self.J[j] = (jax.vmap(grad_fn)(self.xs), grad_fn)

        if i is None or self.dim_y == 1:
            return self.J[j]

        # Compute J[i, j]
        if (i, j) not in self.J:
            if backend_name == "jax":
                # Unlike other backends, in backend jax, a tuple of a jax array and a callable is returned, so that
                # it is consistent with the argument, which is also a tuple. This may be useful for further computation,
                # e.g. Hessian.
                self.J[i, j] = (
                    self.J[j][0][:, i : i + 1],
                    lambda x: self.J[j][1](x)[i : i + 1],
                )
        return self.J[i, j]


# TODO: Refactor duplicate code
class Jacobians:
    """Compute multiple Jacobians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Js = {}

    def __call__(self, ys, xs, i=0, j=None):
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
            self.Js[key] = Jacobian(ys, xs)
        return self.Js[key](i, j)

    def clear(self):
        """Clear cached Jacobians."""
        self.Js = {}


# TODO: Refactor duplicate code
def jacobian(ys, xs, i=0, j=None):
    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int):
        j (int or None):

    Returns:
        J[`i`][`j`] in Jacobian matrix J. If `j` is ``None``, returns the gradient of
        y_i, i.e., J[`i`].
    """
    return jacobian._Jacobians(ys, xs, i=i, j=j)


# TODO: Refactor duplicate code
jacobian._Jacobians = Jacobians()


def hessian(ys, xs, component=0, i=0, j=0):
    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j = 0,..., dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: `ys[:, component]` is used as y to compute the Hessian.
        i (int):
        j (int):

    Returns:
        H[`i`][`j`].
    """
    dys_xj = jacobian(ys, xs, i=None, j=j)
    return jacobian(dys_xj, xs, i=component, j=i)
