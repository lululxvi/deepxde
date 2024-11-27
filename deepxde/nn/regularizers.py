from .. import backend as bkd
from ..backend import backend_name


def get(identifier):
    """Retrieves a TensorFlow regularizer instance based on the given identifier.

    Args:
        identifier (list/tuple): Specifies the type and factor(s) of the regularizer.
            The first element should be one of "l1", "l2", or "l1l2" ("l1+l2").
            For "l1" and "l2", a single regularization factor is expected.
            For "l1l2", provide both "l1" and "l2" factors.
    """

    # TODO: other backends
    if identifier is None or not identifier:
        return None
    if not isinstance(identifier, (list, tuple)):
        raise ValueError("Identifier must be a list or a tuple.")

    name = identifier[0].lower()
    factor = identifier[1:]
    if not factor:
        raise ValueError("Regularization factor must be provided.")

    try:
        if name == "l1":
            return bkd.l1_decay(factor[0])
        if name == "l2":
            return bkd.l2_decay(factor[0])
        if name in ("l1l2", "l1+l2"):
            # TODO: only supported by 'tensorflow.compat.v1' now.
            if len(factor) < 2:
                return bkd.l1_l2_decay(factor[0], factor[1])
    except Exception:
        print(f"{name} regularization to be implemented for backend {backend_name} now.")
    raise ValueError(f"Unknown regularizer name: {name}")
