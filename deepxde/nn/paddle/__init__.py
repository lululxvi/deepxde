"""Package for paddle NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "FNN",
    "MfNN",
    "MsFFN",
    "PFNN",
    "STMsFFN",
]

from .deeponet import DeepONet, DeepONetCartesianProd
from .fnn import FNN, PFNN
from .msffn import MsFFN, STMsFFN
from .mfnn import MfNN
