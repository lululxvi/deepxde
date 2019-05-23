from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Callback(object):
    """Callback base class.

    Properties:
        model: instance of `Model`.
            Reference of the model being trained.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """Init after setting a model."""

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""

    def on_batch_begin(self):
        """Called at the beginning of every batch."""

    def on_batch_end(self):
        """Called at the end of every batch."""

    def on_train_begin(self):
        """Called at the beginning of model training."""

    def on_train_end(self):
        """Called at the end of model training."""

    def on_predict_begin(self):
        """Called at the beginning of prediction."""

    def on_predict_end(self):
        """Called at the end of prediction."""


class CallbackList(Callback):
    """Container abstracting a list of callbacks.
    
    Args:
        callbacks: List of `Callback` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.model = None

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_predict_begin(self):
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_predict_end(self):
        for callback in self.callbacks:
            callback.on_predict_end()

    def append(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")
        self.callbacks.append(callback)


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    Args:
        filepath: string, path to save the model file.
        verbose: verbosity mode, 0 or 1.
        save_better_only: if `save_better_only=True`,
            only save a better model according to the quantity monitored.
            Model is only checked at validation step according to validation_every.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, verbose=0, save_better_only=False, period=1):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = period

        self.monitor = "train loss"
        self.monitor_op = np.less
        self.epochs_since_last_save = 0
        self.best = np.Inf

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            current = self.model.train_state.best_loss_train
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        "\nEpoch {epoch}: {} improved from {:.2e} to {:.2e}, saving model to {}-{epoch} ...".format(
                            self.monitor,
                            self.best,
                            current,
                            self.filepath,
                            epoch=self.model.train_state.epoch,
                        )
                    )
                self.best = current
                self.model.save(self.filepath, verbose=0)
        else:
            self.model.save(self.filepath, verbose=self.verbose)


class OperatorPredictor(Callback):
    """Generates operator values for the input samples.
    
    Args:
        x: The input data.
        op: The operator with inputs (x, y).
    """

    def __init__(self, x, op):
        super(OperatorPredictor, self).__init__()
        self.x = x
        self.op = op
        self.tf_op = None
        self.value = None

    def init(self):
        self.tf_op = self.op(self.model.net.inputs, self.model.net.outputs)

    def on_predict_end(self):
        self.value = self.model.sess.run(
            self.tf_op,
            feed_dict=self.model._get_feed_dict(False, False, 2, self.x, None),
        )

    def get_value(self):
        return self.value


class FirstDerivative(OperatorPredictor):
    """Generates the first order derivative of the outputs with respect to the inputs.
    
    Args:
        x: The input data.
    """

    def __init__(self, x, component_x=0, component_y=0):
        def first_derivative(x, y):
            return tf.gradients(y[:, component_y : component_y + 1], x)[0][
                :, component_x : component_x + 1
            ]

        super(FirstDerivative, self).__init__(x, first_derivative)
