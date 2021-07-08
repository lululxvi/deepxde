"""Package for tensorflow NN modules."""
from __future__ import absolute_import

from .deeponet import DeepONetCartesianProd
from .fnn import FNN
from .nn import NN

__all__ = ["DeepONetCartesianProd", "FNN", "NN"]
