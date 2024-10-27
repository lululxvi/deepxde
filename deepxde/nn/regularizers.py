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
        identifier (str or list/tuple): Specifies the type of regularizer and
            the optional scale values. If a string, it should be one of "l1",
            "l2", "orthogonal", or "l1l2" ("l1+l2"); default scale values will
            be used. If a list or tuple, the first element should be one of
            the above strings, followed by scale values. For "l1", "l2", or
            "orthogonal", you can provide a single scale value. For "l1l2" ("l1+l2"),
            you can provide both "l1" and "l2" scale values.
    """

    # TODO: other backends
    if identifier is None:
        return None

    if isinstance(identifier, str):
        name = identifier.lower()
        scales = []
    elif isinstance(identifier, (list, tuple)) and identifier:
        name = identifier[0].lower()
        scales = identifier[1:]
    else:
        raise ValueError("Identifier must be a string or a non-empty list or tuple.")

    regularizer_class = REGULARIZER_DICT.get(name)
    if not regularizer_class:
        if name == "orthogonal":
            raise ValueError(
                "The 'orthogonal' regularizer is not available "
                "in your version of TensorFlow"
            )
        raise ValueError(f"Unknown regularizer name: {name}")

    regularizer_kwargs = {}
    if scales:
        if name == "l1":
            regularizer_kwargs["l1"] = scales[0]
        elif name == "l2":
            regularizer_kwargs["l2"] = scales[0]
        elif name == "orthogonal":
            regularizer_kwargs["factor"] = scales[0]
        elif name in ("l1l2", "l1+l2"):
            if len(scales) < 2:
                raise ValueError("L1L2 regularizer requires both L1/L2 penalties.")
            regularizer_kwargs["l1"] = scales[0]
            regularizer_kwargs["l2"] = scales[1]
    return regularizer_class(**regularizer_kwargs)
