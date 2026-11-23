"""The ``experimental.nn`` package contains framework-specific implementations for different
neural networks.

Users can directly import ``experimental.nn.<network_name>`` (e.g., ``experimental.nn.FNN``), and
the package will dispatch the network name to the actual implementation according to the
backend framework currently in use.

Note that there are coverage differences among frameworks. If you encounter an
``AttributeError: module 'experimental.nn.XXX' has no attribute 'XXX'`` or ``ImportError:
cannot import name 'XXX' from 'experimental.nn.XXX'`` error, that means the network is not
available to the current backend. If you wish a module to appear in DeepXDE, please
create an issue. If you want to contribute a NN module, please create a pull request.
"""

__all__ = [
    "DictToArray",
    "ArrayToDict",
    "Model",
    "NN",
    "FNN",
    "DeepONet",
    "DeepONetCartesianProd",
    "MIONetCartesianProd",
    "PFNN",
    "PODDeepONet",
    "PODMIONet",
]

from .base import NN
from .convert import DictToArray, ArrayToDict
from .deeponet import DeepONet, DeepONetCartesianProd, PODDeepONet
from .fnn import FNN, PFNN
from .mionet import MIONetCartesianProd, PODMIONet
from .model import Model
