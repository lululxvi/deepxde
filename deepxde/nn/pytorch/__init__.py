"""Package for pytorch NN modules."""

__all__ = ["FNN", "NN", "PFNN", "DeepONetCartesianProd", "PODDeepONet"]

from .fnn import FNN, PFNN
from .nn import NN
from .deeponet import DeepONetCartesianProd, PODDeepONet
