from ..backend import tf


def get(identifier):
    # TODO: other backends
    if identifier is None:
        return None
    name, scales = identifier[0].lower(), identifier[1:]
    return (
        tf.keras.regularizers.L1(l1=scales[0])
        if name == "l1"
        else tf.keras.regularizers.L2(l2=scales[0])
        if name == "l2"
        else tf.keras.regularizers.L1L2(l1=scales[0], l2=scales[1])
        if name in ("l1+l2", "l1l2")
        else None
    )
