"""Enhancing the performance of DeepONets using Zero Coordinate Shift.

Reference: https://arxiv.org/abs/2311.00860
"""

__all__ = [
    "LazyGrad",
    "Model",
    "PDEOperatorCartesianProd",
]

from .gradient import LazyGrad
from .model import Model
from .operator import PDEOperatorCartesianProd
