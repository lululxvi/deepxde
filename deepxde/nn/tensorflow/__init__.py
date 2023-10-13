"""Package for tensorflow NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "DeepONetStrategy",
    "FNN",
    "IndependentStrategy",
    "NN",
    "PFNN",
    "PODDeepONet",
    "SingleOutputStrategy",
    "SplitBothStrategy",
    "SplitBranchStrategy",
    "SplitTrunkStrategy",
]

from .deeponet import (
    DeepONet,
    DeepONetCartesianProd,
    DeepONetStrategy,
    IndependentStrategy,
    PODDeepONet,
    SingleOutputStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy,
)
from .fnn import FNN, PFNN
from .nn import NN
