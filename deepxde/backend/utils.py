def import_tensorflow_compat_v1():
    # pylint: disable=import-outside-toplevel
    try:
        import tensorflow.compat.v1

        assert tensorflow.compat.v1  # silence pyflakes
        return True
    except ImportError:
        return False


def import_tensorflow():
    # pylint: disable=import-outside-toplevel
    try:
        import tensorflow

        assert tensorflow  # silence pyflakes
        return True
    except ImportError:
        return False


def import_pytorch():
    # pylint: disable=import-outside-toplevel
    try:
        import torch

        assert torch  # silence pyflakes
        return True
    except ImportError:
        return False


def import_jax():
    # pylint: disable=import-outside-toplevel
    try:
        import jax

        assert jax  # silence pyflakes
        return True
    except ImportError:
        return False


def import_paddle():
    # pylint: disable=import-outside-toplevel
    try:
        import paddle

        assert paddle  # silence pyflakes
        return True
    except ImportError:
        return False


def verify_backend(backend_name):
    """Verify if the backend is available. If it is available,
    do nothing, otherwise, raise RuntimeError.
    """
    import_funcs = {
        "tensorflow.compat.v1": import_tensorflow_compat_v1,
        "tensorflow": import_tensorflow,
        "pytorch": import_pytorch,
        "jax": import_jax,
        "paddle": import_paddle,
    }

    if backend_name not in import_funcs:
        raise NotImplementedError(
            f"Unsupported backend: {backend_name}.\n"
            "Please select backend from tensorflow.compat.v1, tensorflow, pytorch, jax or paddle."
        )

    if not import_funcs[backend_name]():
        raise RuntimeError(
            f"Backend is set as {backend_name}, but '{backend_name}' failed to import."
        )
