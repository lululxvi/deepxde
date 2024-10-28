__all__ = [
    "backend",
    "callbacks",
    "data",
    "geometry",
    "grad",
    "icbc",
    "nn",
    "utils",
    "Model",
    "Variable",
    "zcs",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"

# Should import backend before importing anything else
from . import backend

from . import callbacks
from . import data
from . import geometry
from . import gradients as grad
from . import icbc
from . import nn
from . import utils
from . import zcs

from .backend import Variable
from .model import Model
from .utils import saveplot

# Backward compatibility
from .icbc import (
    DirichletBC,
    Interface2DBC,
    NeumannBC,
    OperatorBC,
    PeriodicBC,
    RobinBC,
    PointSetBC,
    PointSetOperatorBC,
    IC,
)

maps = nn
