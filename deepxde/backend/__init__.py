# References: https://github.com/dmlc/dgl/tree/master/python/dgl/backend

import importlib
import json
import os
import sys

from . import backend
from .set_default_backend import set_default_backend

_enabled_apis = set()


def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError(
            'API "%s" is not supported by backend "%s".'
            " You can switch to other backends by setting"
            " the DDE_BACKEND environment." % (api, mod_name)
        )

    return _missing_api


def load_backend(mod_name):
    if mod_name not in [
        "tensorflow.compat.v1",
        "tensorflow",
        "pytorch",
        "jax",
        "paddle",
    ]:
        raise NotImplementedError("Unsupported backend: %s" % mod_name)

    print("Using backend: %s\n" % mod_name, file=sys.stderr, flush=True)
    mod = importlib.import_module(".%s" % mod_name.replace(".", "_"), __name__)
    thismod = sys.modules[__name__]
    # log backend name
    setattr(thismod, "backend_name", mod_name)
    for api in backend.__dict__.keys():
        if api.startswith("__"):
            # ignore python builtin attributes
            continue
        if api == "data_type_dict":
            # load data type
            if api not in mod.__dict__:
                raise ImportError(
                    'API "data_type_dict" is required but missing for backend "%s".'
                    % mod_name
                )
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

            # override data type dict function
            setattr(thismod, "data_type_dict", data_type_dict)
            setattr(
                thismod,
                "reverse_data_type_dict",
                {v: k for k, v in data_type_dict.items()},
            )
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))


def get_preferred_backend():
    backend_name = None
    config_path = os.path.join(os.path.expanduser("~"), ".deepxde", "config.json")
    if "DDE_BACKEND" in os.environ:
        backend_name = os.getenv("DDE_BACKEND")
    # Backward compatibility
    elif "DDEBACKEND" in os.environ:
        backend_name = os.getenv("DDEBACKEND")
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get("backend", "").lower()

    if backend_name in [
        "tensorflow.compat.v1",
        "tensorflow",
        "pytorch",
        "jax",
        "paddle",
    ]:
        return backend_name
    print(
        "DeepXDE backend not selected or invalid. Use tensorflow.compat.v1.",
        file=sys.stderr,
    )
    set_default_backend("tensorflow.compat.v1")
    return "tensorflow.compat.v1"


load_backend(get_preferred_backend())


def is_enabled(api):
    """Return true if the api is enabled by the current backend.

    Args:
        api (string): The api name.

    Returns:
        bool: ``True`` if the API is enabled by the current backend.
    """
    return api in _enabled_apis
