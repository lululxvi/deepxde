from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .backend import tf


class Jacobian(object):
    """Compute Jacobian matrix J: J[i][j] = dy_i/dx_j, where i=0,...,dim_y-1 and j=0,...,dim_x-1.

    It is lazy evaluation, i.e., it only computes J[i][j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
    """

    def __init__(self, ys, xs):
        self.ys = ys
        self.xs = xs

        self.dim_y = ys.get_shape().as_list()[1]
        self.dim_x = xs.get_shape().as_list()[1]
        self.J = {}

    def __call__(self, i=0, j=None):
        """Returns J[`i`][`j`].
        If `j` is ``None``, returns the gradient of y_i, i.e., J[i].
        """
        if not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))
        # Compute J[i]
        if i not in self.J:
            y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
            self.J[i] = tf.gradients(y, self.xs)[0]
        return self.J[i] if j is None or self.dim_x == 1 else self.J[i][:, j : j + 1]


class Hessian(object):
    """Compute Hessian matrix H: H[i][j] = d^2y/dx_idx_j, where i,j=0,...,dim_x-1.

    It is lazy evaluation, i.e., it only computes H[i][j] when needed.

    Args:
        y: Output Tensor of shape (batch_size, 1).
        xs: Input Tensor of shape (batch_size, dim_x).
        grad_y: The gradient of `y` w.r.t. `xs`. Provide `grad_y` if known to avoid duplicate computation. `grad_y` can
            be computed from ``Jacobian``.
    """

    def __init__(self, y, xs, grad_y=None):
        dim_y = y.get_shape().as_list()[1]
        if dim_y != 1:
            raise ValueError("The dimension of y is {}.".format(dim_y))

        if grad_y is None:
            grad_y = tf.gradients(y, xs)[0]
        self.H = Jacobian(grad_y, xs)

    def __call__(self, i=0, j=0):
        """Returns H[`i`][`j`].
        """
        return self.H(i, j)


# class Jacobians(object):
#     """Compute multiple Jacobian matrices J: J[i][j] = dy_i/dx_j, where i=0,...,dim_y-1 and j=0,...,dim_x-1.

#     - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
#     - It will remember the gradients that have already been computed to avoid duplicate computation.
#     """

#     def __init__(self):
#         pass

#     def __call__(self, ys, xs, i, j=None):
#         """Returns J[i][j] = dy_i/dx_j in Jacobian matrix J, where i=0,...,dim_y-1 and j=0,...,dim_x-1.
#         If j is ``None``, returns the gradient of y_i J[i], which can be used to construct the Hessian of y_i.
#         """
