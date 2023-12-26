"""Compute gradients using reverse-mode autodiff."""

__all__ = ["jacobian", "hessian"]

from ..backend import backend_name, tf, torch, jax, paddle


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
        J[i].
        """
        if not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))
        # Compute J[i]
        if i not in self.J:
            if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
                y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
                self.J[i] = tf.gradients(y, self.xs)[0]
            elif backend_name == "pytorch":
                # TODO: retain_graph=True has memory leak?
                y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
                self.J[i] = torch.autograd.grad(
                    y, self.xs, grad_outputs=torch.ones_like(y), create_graph=True
                )[0]
            elif backend_name == "paddle":
                y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
                self.J[i] = paddle.grad(y, self.xs, create_graph=True)[0]
            elif backend_name == "jax":
                # Here, we use jax.grad to compute the gradient of a function. This is
                # different from TensorFlow and PyTorch that the input of a function is
                # no longer a batch. Instead, it is a single point. Formally, backend
                # jax computes gradients pointwisely and then vectorizes to batch, by
                # jax.vmap. However, computationally, this is in fact done batchwisely
                # and efficiently. It is very important to note that, without jax.vmap,
                # this can only deal with functions whose output is a scalar and input
                # is a single point.
                # Other options are jax.jacrev + jax.vmap or jax.jacfwd + jax.vmap,
                # which could be used to compute the full Jacobian matrix efficiently,
                # if needed. Also, jax.vjp, jax.jvp will bring more flexibility and
                # efficiency. jax.vjp + jax.vmap or jax.jvp + jax.vmap will be
                # implemented in the future.
                grad_fn = jax.grad(lambda x: self.ys[1](x)[i])
                self.J[i] = (jax.vmap(grad_fn)(self.xs), grad_fn)

        if j is None or self.dim_x == 1:
            return self.J[i]

        # Compute J[i, j]
        if (i, j) not in self.J:
            if backend_name in [
                "tensorflow.compat.v1",
                "tensorflow",
                "pytorch",
                "paddle",
            ]:
                self.J[i, j] = self.J[i][:, j : j + 1]
            elif backend_name == "jax":
                # Unlike other backends, in backend jax, a tuple of a jax array and a callable is returned, so that
                # it is consistent with the argument, which is also a tuple. This may be useful for further computation,
                # e.g. Hessian.
                self.J[i, j] = (
                    self.J[i][0][:, j : j + 1],
                    lambda x: self.J[i][1](x)[j : j + 1],
                )
        return self.J[i, j]


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


jacobian._Jacobians = Jacobians()


class Hessian:
    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j = 0,..., dim_x-1.

    It is lazy evaluation, i.e., it only computes H[i][j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: `ys[:, component]` is used as y to compute the Hessian.
    """

    def __init__(self, ys, xs, component=0):
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            dim_y = ys.shape[1]
        elif backend_name == "jax":
            dim_y = ys[0].shape[1]
        if component >= dim_y:
            raise ValueError(
                "The component of ys={} cannot be larger than the dimension={}.".format(
                    component, dim_y
                )
            )

        # There is no duplicate computation of grad_y.
        grad_y = jacobian(ys, xs, i=component, j=None)
        self.H = Jacobian(grad_y, xs)

    def __call__(self, i=0, j=0):
        """Returns H[`i`][`j`]."""
        return self.H(j, i)


class Hessians:
    """Compute multiple Hessians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Hs = {}

    def __call__(self, ys, xs, component=0, i=0, j=0):
        if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
            key = (ys.ref(), xs.ref(), component)
        elif backend_name in ["pytorch", "paddle"]:
            key = (ys, xs, component)
        elif backend_name == "jax":
            key = (id(ys[0]), id(xs), component)
        if key not in self.Hs:
            self.Hs[key] = Hessian(ys, xs, component=component)
        return self.Hs[key](i, j)

    def clear(self):
        """Clear cached Hessians."""
        self.Hs = {}


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
    return hessian._Hessians(ys, xs, component=component, i=i, j=j)


hessian._Hessians = Hessians()


def clear():
    """Clear cached Jacobians and Hessians."""
    jacobian._Jacobians.clear()
    hessian._Hessians.clear()
