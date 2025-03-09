"""Compute gradients using reverse-mode or forward-mode autodiff."""

__all__ = ["hessian", "jacobian"]

from . import gradients_forward
from . import gradients_reverse
from .. import config


def jacobian(ys, xs, i=None, j=None):
    """Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J as J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y) or (batch_size_out, batch_size,
            dim_y). Here, the `batch_size` is the same one for `xs`, and
            `batch_size_out` is the batch size for an additional/outer dimension.
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int or None): `i`th row. If `i` is ``None``, returns the `j`th column
            J[:, `j`].
        j (int or None): `j`th column. If `j` is ``None``, returns the `i`th row
            J[`i`, :], i.e., the gradient of y_i. `i` and `j` cannot be both ``None``,
            unless J has only one element, which is returned.

    Returns:
        (`i`, `j`)th entry J[`i`, `j`], `i`th row J[`i`, :], or `j`th column J[:, `j`].
        When `ys` has shape (batch_size, dim_y), the output shape is (batch_size, 1).
        When `ys` has shape (batch_size_out, batch_size, dim_y), the output shape is
        (batch_size_out, batch_size, 1).
    """
    if config.autodiff == "reverse":
        return gradients_reverse.jacobian(ys, xs, i=i, j=j)
    if config.autodiff == "forward":
        return gradients_forward.jacobian(ys, xs, i=i, j=j)


def hessian(ys, xs, component=0, i=0, j=0):
    """Compute `Hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_ H as
    H[i, j] = d^2y / dx_i dx_j, where i,j = 0, ..., dim_x - 1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y) or (batch_size_out, batch_size,
            dim_y). Here, the `batch_size` is the same one for `xs`, and
            `batch_size_out` is the batch size for an additional/outer dimension.
        xs: Input Tensor of shape (batch_size, dim_x).
        component: `ys[:, component]` is used as y to compute the Hessian.
        i (int): `i`th row.
        j (int): `j`th column.

    Returns:
        H[`i`, `j`]. When `ys` has shape (batch_size, dim_y), the output shape is
        (batch_size, 1). When `ys` has shape (batch_size_out, batch_size, dim_y),
        the output shape is (batch_size_out, batch_size, 1).
    """
    if config.autodiff == "reverse":
        return gradients_reverse.hessian(ys, xs, component=component, i=i, j=j)
    if config.autodiff == "forward":
        return gradients_forward.hessian(ys, xs, component=component, i=i, j=j)


def clear():
    """Clear cached Jacobians and Hessians."""
    if config.autodiff == "reverse":
        gradients_reverse.clear()
    elif config.autodiff == "forward":
        gradients_forward.clear()
