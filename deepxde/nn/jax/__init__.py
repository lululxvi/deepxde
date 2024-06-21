"""Package for jax NN modules."""

__all__ = ["FNN", "NN", "PFNN", "SPINN"]

from .snn import SPINN
from .fnn import FNN, PFNN
from .nn import NN
