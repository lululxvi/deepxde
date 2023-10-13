"""Package for tensorflow.compat.v1 NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "DeepONetStrategy",
    "FNN",
    "IndependentStrategy",
    "MfNN",
    "MIONet",
    "MIONetCartesianProd",
    "MsFFN",
    "NN",
    "PFNN",
    "ResNet",
    "SingleOutputStrategy",
    "SplitBothStrategy",
    "SplitBranchStrategy",
    "SplitTrunkStrategy",
    "STMsFFN",
]

from .deeponet import (
    DeepONet,
    DeepONetCartesianProd,
    DeepONetStrategy,
    IndependentStrategy,
    SingleOutputStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy,
)
from .fnn import FNN, PFNN
from .mfnn import MfNN
from .mionet import MIONet, MIONetCartesianProd
from .msffn import MsFFN, STMsFFN
from .nn import NN
from .resnet import ResNet
