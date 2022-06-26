from ..backend import tf


def get(identifier):
    # TODO: other backends
    if identifier is None:
        return None
    name, scales = identifier[0], identifier[1:]
    return (
        tf.keras.regularizers.l1(l=scales[0])
        if name == "l1"
        else tf.keras.regularizers.l2(l=scales[0])
        if name == "l2"
        else tf.keras.regularizers.l1_l2(l1=scales[0], l2=scales[1])
        if name == "l1+l2"
        else None
    )
