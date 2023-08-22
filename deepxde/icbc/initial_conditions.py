"""Initial conditions."""

__all__ = ["IC"]

from typing import Any, Callable, List, Optional, overload, Union

import numpy as np
from numpy.typing import NDArray, ArrayLike

from .boundary_conditions import npfunc_range_autocache
from .. import backend as bkd
from .. import utils
from ..geometry import Geometry
from ..types import _Tensor, _TensorOrTensors


class IC:
    """Initial conditions: y([x, t0]) = func([x, t0])."""

    def __init__(
        self,
        geom: Geometry,
        func: Callable[[NDArray[np.float_]], NDArray[np.float_]],
        on_initial: Callable[[NDArray[Any], NDArray[Any]], NDArray[np.bool_]],
        component: Union[List[int], int] = 0,
    ):
        self.geom = geom
        self.func = npfunc_range_autocache(utils.return_tensor(func))
        self.on_initial = lambda x, on: np.array(
            [on_initial(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

    def filter(self, X: NDArray[np.float_]) -> NDArray[np.bool_]:
        return X[self.on_initial(X, self.geom.on_initial(X))]

    def collocation_points(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        return self.filter(X)

    def error(
        self,
        X: NDArray[np.float_],
        inputs: _TensorOrTensors,
        outputs: _Tensor,
        beg: int,
        end: int,
        aux_var: Union[NDArray[np.float_], None] = None,
    ) -> _Tensor:
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "IC function should return an array of shape N by 1 for each component."
                "Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values
