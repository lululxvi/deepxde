# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

from functools import wraps
from typing import Dict, Callable, Sequence, Union, Optional, Tuple, Any, Iterator

import brainstate as bst
import brainunit as u

TransformFn = Callable

__all__ = [
    'jacobian', 'hessian', 'gradient',
]


class GradientTransform(bst.util.PrettyRepr):

    def __init__(
        self,
        target: Callable,
        transform: TransformFn,
        return_value: bool = False,
        has_aux: bool = False,
        transform_params: Optional[Dict[str, Any]] = None,
    ):
        self._return_value = return_value
        self._has_aux = has_aux

        # target
        self.target = target

        # transform
        self._states_to_be_written: Tuple[bst.State, ...] = None
        _grad_setting = dict() if transform_params is None else transform_params
        if self._has_aux:
            self._transform = transform(self._fun_with_aux, has_aux=True, **_grad_setting)
        else:
            self._transform = transform(self._fun_without_aux, has_aux=True, **_grad_setting)

    def __pretty_repr__(self) -> Iterator[Union[bst.util.PrettyType, bst.util.PrettyAttr]]:
        yield bst.util.PrettyType(self.__class__.__name__)
        yield bst.util.PrettyAttr("target", self.target)
        yield bst.util.PrettyAttr("return_value", self._return_value)
        yield bst.util.PrettyAttr("has_aux", self._has_aux)
        yield bst.util.PrettyAttr("transform", self._transform)

    def _call_target(self, *args, **kwargs):
        if self._states_to_be_written is None:
            with bst.StateTraceStack() as stack:
                output = self.target(*args, **kwargs)
                self._states_to_be_written = [st for st in stack.get_write_states()]
        else:
            output = self.target(*args, **kwargs)
        return output

    def _fun_with_aux(self, *args, **kwargs):
        # Users should return the auxiliary data like::
        # >>> # 1. example of return one data
        # >>> return scalar_loss, data
        # >>> # 2. example of return multiple data
        # >>> return scalar_loss, (data1, data2, ...)
        outs = self._call_target(*args, **kwargs)
        # outputs: [0] is the value for gradient,
        #          [1] is other values for return
        assert self._states_to_be_written is not None, "The states to be written should be collected."
        return outs[0], (outs, [v.value for v in self._states_to_be_written])

    def _fun_without_aux(self, *args, **kwargs):
        # Users should return the scalar value like this::
        # >>> return scalar_loss
        out = self._call_target(*args, **kwargs)
        assert self._states_to_be_written is not None, "The states to be written should be collected."
        return out, (out, [v.value for v in self._states_to_be_written])

    def _return(self, rets):
        grads, (outputs, new_dyn_vals) = rets
        for i, val in enumerate(new_dyn_vals):
            self._states_to_be_written[i].value = val

        # check returned value
        if self._return_value:
            # check aux
            if self._has_aux:
                return grads, outputs[0], outputs[1]
            else:
                return grads, outputs
        else:
            # check aux
            if self._has_aux:
                return grads, outputs[1]
            else:
                return grads

    def __call__(self, *args, **kwargs):
        rets = self._transform(*args, **kwargs)
        return self._return(rets)


def _raw_jacrev(
    fun: Callable,
    has_aux: bool = False,
    y: str | Sequence[str] | None = None,
    x: str | Sequence[str] | None = None,
) -> Callable:
    # process only for y
    if isinstance(y, str):
        y = [y]
    if y is not None:
        fun = _format_y(fun, y, has_aux=has_aux)

    # process only for x
    if isinstance(x, str):
        x = [x]

    def transform(inputs):
        if x is not None:
            fun2, inputs = _format_x(fun, x, inputs)
            return u.autograd.jacrev(fun2, has_aux=has_aux)(inputs)
        else:
            return u.autograd.jacrev(fun, has_aux=has_aux)(inputs)

    return transform


def _raw_jacfwd(
    fun: Callable,
    has_aux: bool = False,
    y: str | Sequence[str] | None = None,
    x: str | Sequence[str] | None = None,
) -> Callable:
    # process only for y
    if isinstance(y, str):
        y = [y]
    if y is not None:
        fun = _format_y(fun, y, has_aux=has_aux)

    # process only for x
    if isinstance(x, str):
        x = [x]

    def transform(inputs):
        if x is not None:
            fun2, inputs = _format_x(fun, x, inputs)
            return u.autograd.jacfwd(fun2, has_aux=has_aux)(inputs)
        else:
            return u.autograd.jacfwd(fun, has_aux=has_aux)(inputs)

    return transform


def _raw_hessian(
    fun: Callable,
    has_aux: bool = False,
    y: str | Sequence[str] | None = None,
    xi: str | Sequence[str] | None = None,
    xj: str | Sequence[str] | None = None,
) -> Callable:
    r"""
    Physical unit-aware version of `jax.hessian <https://jax.readthedocs.io/en/latest/_autosummary/jax.hessian.html>`_,
    computing Hessian of ``fun`` as a dense array.

    H[y][xi][xj] = d^2y / dxi dxj

    Args:
      fun: Function whose Hessian is to be computed.  Its arguments at positions
        specified by ``argnums`` should be arrays, scalars, or standard Python
        containers thereof. It should return arrays, scalars, or standard Python
        containers thereof.
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the Hessian of
      ``fun``.
    """

    inner = _raw_jacrev(fun, has_aux=has_aux, y=y, x=xi)

    # process only for xj
    if isinstance(xj, str):
        xj = [xj]

    def transform(inputs):
        if xj is not None:
            fun2, inputs = _format_x(inner, xj, inputs)
            return u.autograd.jacfwd(fun2, has_aux=has_aux)(inputs)
        else:
            return u.autograd.jacfwd(inner, has_aux=has_aux)(inputs)

    return transform


def _format_x(fn, x_keys, xs):
    assert isinstance(xs, dict), 'xs must be a dictionary.'
    assert isinstance(x_keys, (tuple, list)), 'x must be a tuple or list.'
    assert all(isinstance(key, str) for key in x_keys), 'x_keys must be a tuple or list of strings.'
    others = {key: xs[key] for key in xs if key not in x_keys}
    xs = {key: xs[key] for key in x_keys}

    @wraps(fn)
    def fn_new(inputs):
        return fn({**inputs, **others})

    return fn_new, xs


def _format_y(fn, y, has_aux: bool):
    assert isinstance(y, (tuple, list)), 'y must be a tuple or list.'
    assert all(isinstance(key, str) for key in y), 'y must be a tuple or list of strings.'

    @wraps(fn)
    def fn_new(inputs):
        if has_aux:
            outs, _aux = fn(inputs)
            return {key: outs[key] for key in y}, _aux
        else:
            outs = fn(inputs)
            return {key: outs[key] for key in y}

    return fn_new


def jacobian(
    fn: Callable,
    xs: Dict,
    y: str | Sequence[str] | None = None,
    x: str | Sequence[str] | None = None,
    mode: str = 'backward',
    vmap: bool = True,
):
    """
    Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J as J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    Args:
        fn: Function to compute the gradient.
        xs: Inputs of the function.
        mode: The mode of the gradient computation. Choose between 'backward' and 'forward'.
        x (str or None): `i`th row. If `i` is ``None``, returns the `j`th column
            J[:, `j`].
        y (str or None): `j`th column. If `j` is ``None``, returns the `i`th row
            J[`i`, :], i.e., the gradient of y_i. `i` and `j` cannot be both ``None``,
            unless J has only one element, which is returned.

    Returns:
        (`i`, `j`)th entry J[`i`, `j`], `i`th row J[`i`, :], or `j`th column J[:, `j`].
    """
    # assert isinstance(xs, dict), 'xs must be a dictionary.'
    assert isinstance(mode, str), 'mode must be a string.'
    assert mode in ['backward', 'forward'], 'mode must be either backward or forward.'

    # process only for x
    if isinstance(x, str):
        x = [x]

    # process only for y
    if isinstance(y, str):
        y = [y]

    # compute the Jacobian
    if mode == 'backward':
        transform = GradientTransform(fn, _raw_jacrev, transform_params={'y': y, 'x': x})
    elif mode == 'forward':
        transform = GradientTransform(fn, _raw_jacfwd, transform_params={'y': y, 'x': x})
    else:
        raise ValueError('Invalid mode. Choose between backward and forward.')
    if vmap:
        return bst.augment.vmap(transform)(xs)
    else:
        return transform(xs)


def hessian(
    fn: Callable,
    xs: Dict,
    y: str | Sequence[str] | None = None,
    xi: str | Sequence[str] | None = None,
    xj: str | Sequence[str] | None = None,
    vmap: bool = True,
):
    """
    Compute `Hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_ H as
    H[i, j] = d^2y / dx_i dx_j, where i,j = 0, ..., dim_x - 1.

    Args:
        fn: Function to compute the gradient.
        xs: Inputs of the function.
        y (str or None): The output variable.
        xi (str or None): `i`th row. If `i` is ``None``, returns the `j`th column H[:, `j`].
        xj (str or None): `j`th column. If `j` is ``None``, returns the `i`th row
            H[`i`, :], i.e., the gradient of y_i. `i` and `j` cannot be both ``None``,
            unless H has only one element, which is returned.

    Returns:
        H[`i`, `j`].
    """
    # assert isinstance(xs, dict), 'xs must be a dictionary.'
    transform = GradientTransform(fn, _raw_hessian, transform_params={'y': y, 'xi': xi, 'xj': xj})
    if vmap:
        return bst.augment.vmap(transform)(xs)
    else:
        return transform(xs)


def gradient(
    fn: Callable,
    xs: Dict,
    y: str | Sequence[str] | None = None,
    *xi: str | Sequence[str] | None,
    order: int = 1,
):
    """
    Compute the gradient dy/dx of a function y = f(x) with respect to x.

    If order is 1, it computes the first derivative dy/dx.


    Args:
        fn: Function to compute the gradient.
        xs: Inputs of the function.
        y (str or None): The variable to differentiate.
        xi (str or None): The variable to differentiate with respect to.
        order: The order of the gradient. Default is 1.

    Returns:
        dy/dx.
    """
    assert isinstance(order, int), 'order must be an integer.'
    assert order > 0, 'order must be positive.'

    # process only for y
    if isinstance(y, str):
        y = [y]
    if y is not None:
        fn = _format_y(fn, y, has_aux=False)

    # process xi
    if len(xi) > 0:
        assert len(xi) == order, 'The number of xi must be equal to order.'
        xi = list(xi)
        for i in range(order):
            if isinstance(xi[i], str):
                xi[i] = [xi[i]]
    else:
        xi = [None] * order

    # compute the gradient
    for i, x in enumerate(xi):
        if i == 0:
            fn = _raw_jacrev(fn, y=y, x=x)
        else:
            fn = _raw_jacfwd(fn, y=None, x=x)
    return bst.augment.vmap(fn)(xs)
