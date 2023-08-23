from __future__ import annotations
from typing import Sequence, TypeVar

# Tensor from any backend
Tensor = TypeVar("Tensor")
TensorOrTensors = Tensor | Sequence[Tensor]