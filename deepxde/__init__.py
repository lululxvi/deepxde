from .__about__ import __version__

# Should import backend before importing anything else
from . import backend

from . import callbacks
from . import data
from . import geometry
from . import gradients as grad
from . import icbcs
from . import nn
from . import utils

from .backend import Variable
from .icbcs import (
    DirichletBC,
    NeumannBC,
    OperatorBC,
    PeriodicBC,
    RobinBC,
    PointSetBC,
    IC,
)
from .model import Model
from .postprocessing import saveplot

# Backward compatibility
maps = nn

__all__ = [
    "backend",
    "callbacks",
    "data",
    "geometry",
    "grad",
    "icbcs",
    "maps",
    "nn",
    "utils",
    "Model",
    "Variable",
]
