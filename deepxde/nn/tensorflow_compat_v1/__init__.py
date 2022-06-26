"""Package for tensorflow.compat.v1 NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "FNN",
    "MfNN",
    "MIONet",
    "MIONetCartesianProd",
    "MsFFN",
    "NN",
    "PFNN",
    "ResNet",
    "STMsFFN",
]

from .deeponet import DeepONet, DeepONetCartesianProd
from .fnn import FNN, PFNN
from .mfnn import MfNN
from .mionet import MIONet, MIONetCartesianProd
from .msffn import MsFFN, STMsFFN
from .nn import NN
from .resnet import ResNet
