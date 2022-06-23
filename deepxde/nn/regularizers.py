from ..backend import backend_name, tf

def get(identifier):
    # TODO: other backends
    if identifier is None:
        return None
    if backend_name == "pytorch":
        return identifier
    elif backend_name == "tensorflow" or "tensorflow.compat.v1":
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
    else:
        # TODO: other backends
        raise NotImplementedError(
            f"regularization to be implemented for backend {backend_name}."
        )


