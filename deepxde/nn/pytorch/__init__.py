"""Package for pytorch NN modules."""

__all__ = [
    "DeepONetCartesianProd",
    "FNN",
    "MIONetCartesianProd",
    "NN",
    "PFNN",
    "PODDeepONet",
    "PODMIONet",
    "DeepONet",
]

from .deeponet import DeepONetCartesianProd, PODDeepONet, DeepONet
from .mionet import MIONetCartesianProd, PODMIONet
from .fnn import FNN, PFNN
from .nn import NN
