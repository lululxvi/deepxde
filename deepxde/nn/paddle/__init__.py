"""Package for paddle NN modules."""

__all__ = [
    "FNN",
    "MsFFN",
    "PFNN",
    "STMsFFN",
]

from .fnn import FNN
from .fnn import PFNN
from .msffn import MsFFN, STMsFFN
