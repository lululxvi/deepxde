"""Package for tensorflow.compat.v1 NN modules."""
from __future__ import absolute_import

from .bionet import BiONet
from .deeponet import DeepONet, DeepONetCartesianProd, FourierDeepONetCartesianProd
from .fnn import FNN
from .mfnn import MfNN
from .mfonet import MfONet
from .msffn import MsFFN, STMsFFN
from .pfnn import PFNN
from .resnet import ResNet

__all__ = [
    "BiONet",
    "DeepONet",
    "DeepONetCartesianProd",
    "FourierDeepONetCartesianProd",
    "FNN",
    "MfNN",
    "MfONet",
    "MsFFN",
    "STMsFFN",
    "PFNN",
    "ResNet",
]
