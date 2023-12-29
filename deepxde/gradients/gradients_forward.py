"""Compute gradients using forward-mode autodiff."""

__all__ = ["hessian", "jacobian"]

from .jacobian import Jacobian, Jacobians
from ..backend import backend_name, jax


class JacobianForward(Jacobian):
    def __call__(self, i=None, j=None):
        super().__call__(i=i, j=j)
        # Compute a row is not supported in forward mode, unless there is only one
        # input.
        if j is None:
            if self.dim_x > 1:
                raise NotImplementedError(
                    "Forward-mode autodiff doesn't support computing gradient."
                )
            j = 0

        # Compute J[:, j]
        if j not in self.J:
            if backend_name in [
                "tensorflow.compat.v1",
                "tensorflow",
                "pytorch",
                "paddle",
            ]:
                # TODO: Other backends
                raise NotImplementedError(
                    "Backend f{backend_name} doesn't support forward-mode autodiff."
                )
            elif backend_name == "jax":
                # Here, we use jax.jvp to compute the gradient of a function. This is
                # different from TensorFlow and PyTorch that the input of a function is
                # no longer a batch. Instead, it is a single point. Formally, backend
                # jax computes gradients pointwisely and then vectorizes to batch, by
                # jax.vmap. It is very important to note that, without jax.vmap, this
                # can only deal with functions whose output is a scalar and input is a
                # single point.
                # Other option is jax.jacfwd + jax.vmap, which could be used to compute
                # the full Jacobian matrix efficiently, if needed.
                tangent = jax.numpy.zeros(self.dim_x).at[j].set(1)
                grad_fn = lambda x: jax.jvp(self.ys[1], (x,), (tangent,))[1]
                self.J[j] = (jax.vmap(grad_fn)(self.xs), grad_fn)

        if i is None or self.dim_y == 1:
            return self.J[j]

        # Compute J[i, j]
        if (i, j) not in self.J:
            if backend_name == "jax":
                # In backend jax, a tuple of a jax array and a callable is returned, so
                # that it is consistent with the argument, which is also a tuple. This
                # is useful for further computation, e.g., Hessian.
                self.J[i, j] = (
                    self.J[j][0][:, i : i + 1],
                    lambda x: self.J[j][1](x)[i : i + 1],
                )
        return self.J[i, j]


def jacobian(ys, xs, i=None, j=None):
    return jacobian._Jacobians(ys, xs, i=i, j=j)


jacobian._Jacobians = Jacobians(JacobianForward)


def hessian(ys, xs, component=0, i=0, j=0):
    dys_xj = jacobian(ys, xs, i=None, j=j)
    return jacobian(dys_xj, xs, i=component, j=i)
