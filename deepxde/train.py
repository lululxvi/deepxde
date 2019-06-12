from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def is_scipy_opts(optimizer):
    scipy_opts = ["BFGS", "L-BFGS-B", "Nelder-Mead", "Powell", "CG", "Newton-CG"]
    return optimizer in scipy_opts


def get_train_op(loss, optimizer, lr=None, decay=None):
    if is_scipy_opts(optimizer):
        if lr is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            method=optimizer,
            options={
                "disp": None,
                "maxcor": 50,
                "ftol": np.finfo(float).eps,
                "gtol": 1e-5,
                "eps": 1e-8,
                "maxfun": 15000,
                "maxiter": 15000,
                "maxls": 50,
            },
        )

    if lr is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    lr, global_step = _get_learningrate(lr, decay)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = _get_optimizer(optimizer, lr).minimize(loss, global_step=global_step)
    return train_op


def _get_learningrate(lr, decay):
    if decay is None:
        return lr, None
    global_step = tf.Variable(0, trainable=False)
    return (
        {
            "inverse time": tf.train.inverse_time_decay(
                lr, global_step, decay[1], decay[2]
            ),
            "cosine": tf.train.cosine_decay(lr, global_step, decay[1], alpha=decay[2]),
        }[decay[0]],
        global_step,
    )


def _get_optimizer(name, lr):
    return {
        "sgd": tf.train.GradientDescentOptimizer(lr),
        "sgdnesterov": tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        "adagrad": tf.train.AdagradOptimizer(0.01),
        "adadelta": tf.train.AdadeltaOptimizer(),
        "rmsprop": tf.train.RMSPropOptimizer(lr),
        "adam": tf.train.AdamOptimizer(lr),
    }[name]
