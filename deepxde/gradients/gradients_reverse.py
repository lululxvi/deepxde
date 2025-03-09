"""Compute gradients using reverse-mode autodiff."""

__all__ = ["hessian", "jacobian"]

from .jacobian import Jacobian, Jacobians
from .. import backend as bkd
from ..backend import backend_name, tf, torch, jax, paddle


class JacobianReverse(Jacobian):
    def __call__(self, i=None, j=None):
        super().__call__(i=i, j=j)
        # Compute a column is not supported in reverse mode, unless there is only one
        # output.
        if i is None:
            if self.dim_y > 1:
                raise NotImplementedError(
                    "Reverse-mode autodiff doesn't support computing a column."
                )
            i = 0
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            ndim_y = bkd.ndim(self.ys)
        elif backend_name == "jax":
            ndim_y = bkd.ndim(self.ys[0])
        if ndim_y == 3:
            raise NotImplementedError(
                "Reverse-mode autodiff doesn't support 3D output"
            )

        # Compute J[i, :]
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
            elif backend_name == "jax":
                # Here, we use jax.grad to compute the gradient of a function. This is
                # different from TensorFlow and PyTorch that the input of a function is
                # no longer a batch. Instead, it is a single point. Formally, backend
                # jax computes gradients pointwisely and then vectorizes to batch, by
                # jax.vmap. It is very important to note that, without jax.vmap, this
                # can only deal with functions whose output is a scalar and input is a
                # single point.
                # Other option is jax.jacrev + jax.vmap, which could be used to compute
                # the full Jacobian matrix efficiently, if needed. Another option is
                # jax.vjp + jax.vmap.
                grad_fn = jax.grad(lambda x: self.ys[1](x)[i])
                self.J[i] = (jax.vmap(grad_fn)(self.xs), grad_fn)
            elif backend_name == "paddle":
                y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
                self.J[i] = paddle.grad(y, self.xs, create_graph=True)[0]

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
                # In backend jax, a tuple of a jax array and a callable is returned, so
                # that it is consistent with the argument, which is also a tuple. This
                # is useful for further computation, e.g., Hessian.
                self.J[i, j] = (
                    self.J[i][0][:, j : j + 1],
                    lambda x: self.J[i][1](x)[j : j + 1],
                )
        return self.J[i, j]


def jacobian(ys, xs, i=None, j=None):
    return jacobian._Jacobians(ys, xs, i=i, j=j)


jacobian._Jacobians = Jacobians(JacobianReverse)


class Hessian:
    """Compute `Hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_ H as
    H[i, j] = d^2y / dx_i dx_j, where i,j = 0, ..., dim_x - 1.

    It is lazy evaluation, i.e., it only computes H[i, j] when needed.

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
        self.H = JacobianReverse(grad_y, xs)

    def __call__(self, i=0, j=0):
        """Returns H[`i`, `j`]."""
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
    return hessian._Hessians(ys, xs, component=component, i=i, j=j)


hessian._Hessians = Hessians()


def clear():
    """Clear cached Jacobians and Hessians."""
    jacobian._Jacobians.clear()
    hessian._Hessians.clear()
