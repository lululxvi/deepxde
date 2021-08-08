from __future__ import absolute_import

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
from .postprocessing import (
    plot_best_state,
    plot_loss_history,
    save_best_state,
    save_loss_history,
    saveplot,
)

# Backward compatibility
maps = nn

__all__ = [
    "callbacks",
    "data",
    "geometry",
    "grad",
    "icbcs",
    "maps",
    "nn",
    "utils",
    "Model",
]
