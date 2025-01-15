# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from __future__ import annotations

from typing import Dict, Sequence

import brainstate as bst

from pinnx.grad import jacobian, hessian, gradient
from .convert import DictToArray, ArrayToDict

__all__ = [
    'Model',
]


class Model(bst.nn.Module):
    """
    A neural network approximator.

    Args:
        input: The input check.
        approx: The neural network model.
        output: The output unit.

    """

    def __init__(
        self,
        input: DictToArray,
        approx: bst.nn.Module,
        output: ArrayToDict,
        *args,
    ):
        super().__init__()

        assert isinstance(input, DictToArray), "input must be an instance of DictToArray."
        self.input = input

        assert isinstance(approx, bst.nn.Module), "approx must be an instance of nn.Module."
        self.approx = approx

        assert isinstance(output, ArrayToDict), "output must be an instance of Output."
        self.output = output

    @bst.compile.jit(static_argnums=(0,))
    def update(self, x):
        return self.output(self.approx(self.input(x)))

    def jacobian(
        self,
        inputs: Dict[str, bst.typing.ArrayLike],
        y: str | Sequence[str] | None = None,
        x: str | Sequence[str] | None = None,
    ):
        """
        Compute the Jacobian of the approximation neural networks.

        Args:
            inputs: The input data.
            y: The output variables.
            x: The input variables.

        Returns:
            The Jacobian of the approximation neural networks.
        """
        return jacobian(self, inputs, y=y, x=x)

    def hessian(
        self,
        inputs: Dict[str, bst.typing.ArrayLike],
        y: str | Sequence[str] | None = None,
        xi: str | Sequence[str] | None = None,
        xj: str | Sequence[str] | None = None,
    ):
        """
        Compute the Hessian of the approximator.

        Compute: `H[y][xi][xj] = d^2y / dxi dxj = d^2y / dxj dxi`

        Args:
            inputs: The input data.
            y: The output variables.
            xi: The first input variables.
            xj: The second input variables.

        Returns:
            The Hessian of the approximator.
        """
        return hessian(self, inputs, y=y, xi=xi, xj=xj)

    def gradient(
        self,
        inputs: Dict[str, bst.typing.ArrayLike],
        order: int,
        y: str | Sequence[str] | None = None,
        *xi: str | Sequence[str] | None,
    ):
        """
        Compute the gradient of the approximator.

        Args:
            inputs: The input data.
            order: The order of the gradient.
            y: The output variables.
            xi: The input variables.

        Returns:
            The gradient of the approximator.
        """
        assert isinstance(order, int) and order >= 1, "order must be an integer greater than or equal to 1."
        return gradient(self, inputs, y, *xi, order=order)
