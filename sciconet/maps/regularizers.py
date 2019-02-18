from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get(identifier):
    if identifier is None:
        return None
    name, scales = identifier[0], identifier[1:]
    return (
        tf.contrib.layers.l1_regularizer(scales[0])
        if name == "l1"
        else tf.contrib.layers.l2_regularizer(scales[0])
        if name == "l2"
        else tf.contrib.layers.l1_l2_regularizer(scales[0], scales[1])
        if name == "l1+l2"
        else None
    )
