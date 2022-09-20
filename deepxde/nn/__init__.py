"""The ``deepxde.nn`` package contains framework-specific implementations for different
neural networks.

Users can directly import ``deepxde.nn.<network_name>`` (e.g., ``deepxde.nn.FNN``), and
the package will dispatch the network name to the actual implementation according to the
backend framework currently in use.

Note that there are coverage differences among frameworks. If you encounter an
``AttributeError: module 'deepxde.nn.XXX' has no attribute 'XXX'`` or ``ImportError:
cannot import name 'XXX' from 'deepxde.nn.XXX'`` error, that means the network is not
available to the current backend. If you wish a module to appear in DeepXDE, please
create an issue. If you want to contribute a NN module, please create a pull request.
"""

import importlib
import os
import sys

from ..backend import backend_name


# To get Sphinx documentation to build, we import all
if os.environ.get("READTHEDOCS") == "True":
    # The backend should be tensorflow/tensorflow.compat.v1 to ensure backend.tf is not
    # None.
    from . import jax
    from . import paddle
    from . import pytorch
    from . import tensorflow
    from . import tensorflow_compat_v1


def _load_backend(mod_name):
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


_load_backend(backend_name.replace(".", "_"))
