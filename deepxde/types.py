from __future__ import annotations
from typing import Sequence, TypeVar, Union

# Tensor from any backend
Tensor = TypeVar("Tensor")
TensorOrTensors = Union[Tensor, Sequence[Tensor]]
SparseTensor = TypeVar("SparseTensor")
dtype = TypeVar("dtype")