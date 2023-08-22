import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, TypeVar
from numpy.typing import NDArray, ArrayLike

# Tensor from any backend
Tensor = TypeVar("Tensor")
TensorOrTensors = Union[Tensor, Sequence[Tensor]]