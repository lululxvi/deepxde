from ..backend import tf

REGULARIZER_DICT = {
    "l1": tf.keras.regularizers.L1,
    "l2": tf.keras.regularizers.L2,
    "l1l2": tf.keras.regularizers.L1L2,
    "l1+l2": tf.keras.regularizers.L1L2,
}
if hasattr(tf.keras.regularizers, "OrthogonalRegularizer"):
    REGULARIZER_DICT["orthogonal"] = tf.keras.regularizers.OrthogonalRegularizer


def get(identifier):
    """Retrieves a TensorFlow regularizer instance based on the given identifier.

    Args:
        identifier (list/tuple): Specifies the type of regularizer and
            regularization factor. The first element should be one of "l1", "l2",
            "orthogonal", or "l1l2" ("l1+l2"). For "l1", "l2", or "orthogonal",
            you can provide a single factor value. For "l1l2" ("l1+l2"),
            both "l1" and "l2" factors are required.
    """

    # TODO: other backends
    if identifier is None or not identifier:
        return None

    if isinstance(identifier, (list, tuple)):
        name = identifier[0].lower()
        factor = identifier[1:]
    else:
        raise ValueError("Identifier must be a non-empty list or tuple.")

    if not factor:
        raise ValueError("Regularization factor must be provided.")

    regularizer_class = REGULARIZER_DICT.get(name)
    if not regularizer_class:
        if name == "orthogonal":
            raise ValueError(
                "The 'orthogonal' regularizer is not available "
                "in your version of TensorFlow"
            )
        raise ValueError(f"Unknown regularizer name: {name}")

    regularizer_kwargs = {}
    if name == "l1":
        regularizer_kwargs["l1"] = factor[0]
    elif name == "l2":
        regularizer_kwargs["l2"] = factor[0]
    elif name == "orthogonal":
        regularizer_kwargs["factor"] = factor[0]
    elif name in ("l1l2", "l1+l2"):
        if len(factor) < 2:
            raise ValueError("L1L2 regularizer requires both L1/L2 penalties.")
        regularizer_kwargs["l1"] = factor[0]
        regularizer_kwargs["l2"] = factor[1]
    return regularizer_class(**regularizer_kwargs)
