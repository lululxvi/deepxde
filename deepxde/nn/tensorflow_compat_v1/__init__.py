"""Package for tensorflow.compat.v1 NN modules."""

from .mionet import MIONet, MIONetCartesianProd
from .deeponet import DeepONet, DeepONetCartesianProd, FourierDeepONetCartesianProd
from .fnn import FNN, PFNN
from .mfnn import MfNN
from .mfonet import MfONet
from .msffn import MsFFN, STMsFFN
from .nn import NN
from .resnet import ResNet

__all__ = [
    "MIONet",
    "MIONetCartesianProd",
    "DeepONet",
    "DeepONetCartesianProd",
    "FourierDeepONetCartesianProd",
    "FNN",
    "PFNN",
    "MfNN",
    "MfONet",
    "MsFFN",
    "NN",
    "STMsFFN",
    "ResNet",
]
