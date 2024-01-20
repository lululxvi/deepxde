"""Package for paddle NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "PODDeepONet",
    "FNN",
    "MsFFN",
    "PFNN",
    "STMsFFN",
]

from .deeponet import DeepONet, DeepONetCartesianProd, PODDeepONet
from .fnn import FNN, PFNN
from .msffn import MsFFN, STMsFFN
