import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, TypeVar
from numpy.typing import NDArray, ArrayLike

# Tensor from any backend
_Tensor = TypeVar("_Tensor")
_TensorOrTensors = Union[_Tensor, Sequence[_Tensor]]