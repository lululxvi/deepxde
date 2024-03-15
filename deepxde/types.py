from __future__ import annotations
from typing import Sequence, TypeVar, Union

# dtype from any backend
dtype = TypeVar("dtype")

# NN from any backend (Using the `NN` from deepxde is recommended.)
NN = TypeVar("NN")

# SparseTensor from any backend
SparseTensor = TypeVar("SparseTensor")

# Tensor from any backend
Tensor = TypeVar("Tensor")
TensorOrTensors = Union[Tensor, Sequence[Tensor]]
