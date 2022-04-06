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
]

import sys
import os

from .__about__ import __version__

# Should import backend before importing anything else
from . import backend

thismod = sys.modules[__name__]

# Set up random seed
from .config import set_random_seed

if "DDE_SEED" in os.environ:
    seed = os.getenv("DDE_SEED")
    try:
        seed = int(seed)
        set_random_seed(seed)
    except ValueError:
        seed = None
    print("Using seed: %s\n" % seed, file=sys.stderr, flush=True)

setattr(thismod, "seed", seed)

from . import callbacks
from . import data
from . import geometry
from . import gradients as grad
from . import icbc
from . import nn
from . import utils

from .backend import Variable
from .model import Model
from .utils import saveplot

# Backward compatibility
from .icbc import (
    DirichletBC,
    NeumannBC,
    OperatorBC,
    PeriodicBC,
    RobinBC,
    PointSetBC,
    IC,
)

maps = nn
