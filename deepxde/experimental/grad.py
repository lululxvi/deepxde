from __future__ import annotations

from functools import wraps
from typing import Dict, Callable, Sequence, Union, Optional, Tuple, Any, Iterator

import brainstate as bst
import brainunit as u

TransformFn = Callable

__all__ = [
    "jacobian",
    "hessian",
    "gradient",
]


class GradientTransform(bst.util.PrettyRepr):
    """
    A class for transforming gradient computations.

    This class wraps a target function and applies a gradient transformation to it.
    It handles auxiliary data and state management during the transformation process.

    Attributes:
        target (Callable): The target function to be transformed.
        _transform (Callable): The transformed function.
        _return_value (bool): Flag to determine if the original function value should be returned.
        _has_aux (bool): Flag to indicate if the target function returns auxiliary data.
        _states_to_be_written (Tuple[bst.State, ...]): States that need to be updated after computation.
    """

    def __init__(
        self,
        target: Callable,
        transform: TransformFn,
        return_value: bool = False,
        has_aux: bool = False,
        transform_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GradientTransform.

        Args:
            target (Callable): The target function to be transformed.
            transform (TransformFn): The transformation function to be applied.
            return_value (bool, optional): If True, return the original function value along with the gradient. Defaults to False.
            has_aux (bool, optional): If True, the target function returns auxiliary data. Defaults to False.
            transform_params (Optional[Dict[str, Any]], optional): Additional parameters for the transformation. Defaults to None.
        """
        self._return_value = return_value
        self._has_aux = has_aux

        # target
        self.target = target

        # transform
        self._states_to_be_written: Tuple[bst.State, ...] = None
        _grad_setting = dict() if transform_params is None else transform_params
        if self._has_aux:
            self._transform = transform(
                self._fun_with_aux, has_aux=True, **_grad_setting
            )
        else:
            self._transform = transform(
                self._fun_without_aux, has_aux=True, **_grad_setting
            )

    def __pretty_repr__(
        self,
    ) -> Iterator[Union[bst.util.PrettyType, bst.util.PrettyAttr]]:
        """
        Generate a pretty representation of the GradientTransform instance.

        Returns:
            Iterator[Union[bst.util.PrettyType, bst.util.PrettyAttr]]: An iterator of pretty-formatted attributes.
        """
        yield bst.util.PrettyType(self.__class__.__name__)
        yield bst.util.PrettyAttr("target", self.target)
        yield bst.util.PrettyAttr("return_value", self._return_value)
        yield bst.util.PrettyAttr("has_aux", self._has_aux)
        yield bst.util.PrettyAttr("transform", self._transform)

    def _call_target(self, *args, **kwargs):
        """
        Call the target function and collect states to be written.

        Args:
            *args: Positional arguments for the target function.
            **kwargs: Keyword arguments for the target function.

        Returns:
            Any: The output of the target function.
        """
        if self._states_to_be_written is None:
            with bst.StateTraceStack() as stack:
                output = self.target(*args, **kwargs)
                self._states_to_be_written = [st for st in stack.get_write_states()]
        else:
            output = self.target(*args, **kwargs)
        return output

    def _fun_with_aux(self, *args, **kwargs):
        """
        Wrapper for target function when it returns auxiliary data.

        Args:
            *args: Positional arguments for the target function.
            **kwargs: Keyword arguments for the target function.

        Returns:
            Tuple: A tuple containing the main output and auxiliary data.
        """
        outs = self._call_target(*args, **kwargs)
        assert (
            self._states_to_be_written is not None
        ), "The states to be written should be collected."
        return outs[0], (outs, [v.value for v in self._states_to_be_written])

    def _fun_without_aux(self, *args, **kwargs):
        """
        Wrapper for target function when it doesn't return auxiliary data.

        Args:
            *args: Positional arguments for the target function.
            **kwargs: Keyword arguments for the target function.

        Returns:
            Tuple: A tuple containing the output and related data.
        """
        out = self._call_target(*args, **kwargs)
        assert (
            self._states_to_be_written is not None
        ), "The states to be written should be collected."
        return out, (out, [v.value for v in self._states_to_be_written])

    def _return(self, rets):
        """
        Process and return the results of the transformation.

        Args:
            rets: The results from the transformation.

        Returns:
            Tuple: Processed results based on the configuration of return_value and has_aux.
        """
        grads, (outputs, new_dyn_vals) = rets
        for i, val in enumerate(new_dyn_vals):
            self._states_to_be_written[i].value = val

        if self._return_value:
            if self._has_aux:
                return grads, outputs[0], outputs[1]
            else:
                return grads, outputs
        else:
            if self._has_aux:
                return grads, outputs[1]
            else:
                return grads

    def __call__(self, *args, **kwargs):
        """
        Call the transformed function and process its results.

        Args:
            *args: Positional arguments for the transformed function.
            **kwargs: Keyword arguments for the transformed function.

        Returns:
            Any: The processed results of the transformation.
        """
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
    assert isinstance(xs, dict), "xs must be a dictionary."
    assert isinstance(x_keys, (tuple, list)), "x must be a tuple or list."
    assert all(
        isinstance(key, str) for key in x_keys
    ), "x_keys must be a tuple or list of strings."
    others = {key: xs[key] for key in xs if key not in x_keys}
    xs = {key: xs[key] for key in x_keys}

    @wraps(fn)
    def fn_new(inputs):
        return fn({**inputs, **others})

    return fn_new, xs


def _format_y(fn, y, has_aux: bool):
    assert isinstance(y, (tuple, list)), "y must be a tuple or list."
    assert all(
        isinstance(key, str) for key in y
    ), "y must be a tuple or list of strings."

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
    mode: str = "backward",
    vmap: bool = True,
):
    """
    Compute the Jacobian matrix of a function.

    This function calculates the Jacobian matrix J as J[i, j] = dy_i / dx_j,
    where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    Args:
        fn (Callable): The function to compute the Jacobian for.
        xs (Dict): A dictionary containing the input values for the function.
        y (str | Sequence[str] | None, optional): Specifies the output variable(s) for which
            to compute the Jacobian. If None, computes for all outputs. Defaults to None.
        x (str | Sequence[str] | None, optional): Specifies the input variable(s) with respect
            to which the Jacobian is computed. If None, computes for all inputs. Defaults to None.
        mode (str, optional): The mode of gradient computation. Either 'backward' or 'forward'.
            Defaults to 'backward'.
        vmap (bool, optional): Whether to use vectorized mapping. Defaults to True.

    Returns:
        The Jacobian matrix. Depending on the inputs, it can be:
        - The full Jacobian matrix if both x and y are None or specify all variables.
        - A row vector J[i, :] if y specifies a single output and x is None.
        - A column vector J[:, j] if x specifies a single input and y is None.
        - A scalar J[i, j] if both x and y specify single variables.

    Raises:
        ValueError: If an invalid mode is specified.

    Note:
        The function uses automatic differentiation techniques to compute the Jacobian.
        The 'backward' mode is generally more efficient for functions with more outputs than inputs,
        while 'forward' mode is more efficient for functions with more inputs than outputs.
    """
    # assert isinstance(xs, dict), 'xs must be a dictionary.'
    assert isinstance(mode, str), "mode must be a string."
    assert mode in ["backward", "forward"], "mode must be either backward or forward."

    # process only for x
    if isinstance(x, str):
        x = [x]

    # process only for y
    if isinstance(y, str):
        y = [y]

    # compute the Jacobian
    if mode == "backward":
        transform = GradientTransform(
            fn, _raw_jacrev, transform_params={"y": y, "x": x}
        )
    elif mode == "forward":
        transform = GradientTransform(
            fn, _raw_jacfwd, transform_params={"y": y, "x": x}
        )
    else:
        raise ValueError("Invalid mode. Choose between backward and forward.")
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
    Compute the Hessian matrix of a function.

    This function calculates the Hessian matrix H as H[i, j] = d^2y / dx_i dx_j,
    where i, j = 0, ..., dim_x - 1.

    Args:
        fn (Callable): The function for which to compute the Hessian.
        xs (Dict): A dictionary containing the input values for the function.
        y (str | Sequence[str] | None, optional): Specifies the output variable(s) for which
            to compute the Hessian. If None, computes for all outputs. Defaults to None.
        xi (str | Sequence[str] | None, optional): Specifies the input variable(s) for the i-th
            dimension of the Hessian. If None, computes for all inputs in this dimension.
            Defaults to None.
        xj (str | Sequence[str] | None, optional): Specifies the input variable(s) for the j-th
            dimension of the Hessian. If None, computes for all inputs in this dimension.
            Defaults to None.
        vmap (bool, optional): Whether to use vectorized mapping. Defaults to True.

    Returns:
        The Hessian matrix or a part of it, depending on the specified xi and xj:
        - If both xi and xj are None, returns the full Hessian matrix.
        - If xi is specified and xj is None, returns the i-th row of the Hessian, H[i, :].
        - If xj is specified and xi is None, returns the j-th column of the Hessian, H[:, j].
        - If both xi and xj are specified, returns the specific element H[i, j].

    Note:
        xi and xj cannot both be None unless the Hessian has only one element.
    """
    # assert isinstance(xs, dict), 'xs must be a dictionary.'
    transform = GradientTransform(
        fn, _raw_hessian, transform_params={"y": y, "xi": xi, "xj": xj}
    )
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
    Compute the gradient of a function with respect to specified variables.

    This function calculates the gradient dy/dx of a function y = f(x) with respect to x.
    It supports computing higher-order gradients by specifying the 'order' parameter.

    Args:
        fn (Callable): The function for which to compute the gradient.
        xs (Dict): A dictionary containing the input values for the function.
        y (str | Sequence[str] | None, optional): Specifies the output variable(s) to differentiate.
            If None, computes for all outputs. Defaults to None.
        *xi (str | Sequence[str] | None): Variable-length argument specifying the input variable(s)
            to differentiate with respect to. The number of xi arguments should match the 'order' parameter.
        order (int, optional): The order of the gradient to compute. Default is 1 (first derivative).

    Returns:
        The computed gradient. The structure and dimensions of the output depend on the inputs:
        - For first-order gradients (order=1), returns dy/dx.
        - For higher-order gradients, returns the corresponding higher-order derivative.

    Raises:
        AssertionError: If 'order' is not a positive integer or if the number of 'xi' arguments
                        doesn't match the specified 'order'.

    Note:
        The function uses a combination of reverse-mode (for the first derivative) and
        forward-mode (for higher-order derivatives) automatic differentiation.
    """
    assert isinstance(order, int), "order must be an integer."
    assert order > 0, "order must be positive."

    # process only for y
    if isinstance(y, str):
        y = [y]
    if y is not None:
        fn = _format_y(fn, y, has_aux=False)

    # process xi
    if len(xi) > 0:
        assert len(xi) == order, "The number of xi must be equal to order."
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
