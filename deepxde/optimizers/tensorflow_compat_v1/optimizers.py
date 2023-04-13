__all__ = ["get", "is_external_optimizer"]

from .scipy_optimizer import ScipyOptimizerInterface
from ..config import LBFGS_options
from ...backend import tf


def is_external_optimizer(optimizer):
    scipy_opts = ["L-BFGS", "L-BFGS-B"]
    return optimizer in scipy_opts


def get(loss, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if is_external_optimizer(optimizer):
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return ScipyOptimizerInterface(
            loss,
            method="L-BFGS-B",
            options={
                "maxcor": LBFGS_options["maxcor"],
                "ftol": LBFGS_options["ftol"],
                "gtol": LBFGS_options["gtol"],
                "maxfun": LBFGS_options["maxfun"],
                "maxiter": LBFGS_options["maxiter"],
                "maxls": LBFGS_options["maxls"],
            },
        )

    if isinstance(optimizer, tf.train.AdamOptimizer):
        optim = optimizer
        global_step = None
    else:
        if learning_rate is None:
            raise ValueError("No learning rate for {}.".format(optimizer))
        lr, global_step = _get_learningrate(learning_rate, decay)

        if optimizer == "sgd":
            optim = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "sgdnesterov":
            optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        elif optimizer == "adagrad":
            optim = tf.train.AdagradOptimizer(lr)
        elif optimizer == "adadelta":
            optim = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == "rmsprop":
            optim = tf.train.RMSPropOptimizer(lr)
        elif optimizer == "adam":
            optim = tf.train.AdamOptimizer(lr)
        else:
            raise NotImplementedError(
                f"{optimizer} to be implemented for backend tensorflow.compat.v1."
            )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(loss, global_step=global_step)
    return train_op


def _get_learningrate(lr, decay):
    if decay is None:
        return lr, None
    global_step = tf.Variable(0, trainable=False)
    if decay[0] == "inverse time":
        lr = tf.train.inverse_time_decay(lr, global_step, decay[1], decay[2])
    elif decay[0] == "cosine":
        lr = tf.train.cosine_decay(lr, global_step, decay[1], alpha=decay[2])
    else:
        raise NotImplementedError(
            f"{decay[0]} decay to be implemented for backend tensorflow.compat.v1."
        )
    return lr, global_step
