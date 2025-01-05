import importlib
import os
import sys

from .config import LBFGS_options, set_LBFGS_options, NNCG_options, set_NNCG_options
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
