from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from collections import OrderedDict

import numpy as np

from . import display
from . import losses as losses_module
from . import metrics as metrics_module
from . import train as train_module
from .backend import tf
from .callbacks import CallbackList
from .utils import get_num_args, guarantee_initialized_variables, timing


class Model(object):
    """The ``Model`` class trains a ``Map`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.maps.Map`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.optimizer = None
        self.batch_size = None

        self.losses = None
        self.totalloss = None
        self.train_op = None
        self.metrics = None

        self.sess = None
        self.saver = None
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False
        self.callbacks = None

    def close(self):
        self._close_tfsession()

    @timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String. Name of optimizer.
            lr: A Tensor or a floating point value. The learning rate.
            loss: If the same loss is used for all errors, then `loss` is a String (name of objective function) or
                objective function. If different errors use different losses, then `loss` is a list whose size is equal
                to the number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay: Tuple. Name and parameters of decay to the initial learning rate. One of the following options:

                - `inverse time decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/inverse_time_decay>`_: ("inverse time", decay_steps, decay_rate)
                - `cosine decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats)
                to weight the loss contributions. The loss value that will be minimized by the model
                will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
        """
        print("Compiling model...")

        if not self.net.built:
            self.net.build()
        self._open_tfsession()

        self.optimizer = optimizer

        loss = losses_module.get(loss)
        self.losses = self.data.losses(self.net.targets, self.net.outputs, loss, self)
        if not isinstance(self.losses, list):
            self.losses = [self.losses]
        if self.net.regularizer is not None:
            self.losses.append(tf.losses.get_regularization_loss())
        self.losses = tf.convert_to_tensor(self.losses)
        if loss_weights is not None:
            self.losses *= loss_weights
            self.losshistory.set_loss_weights(loss_weights)
        self.totalloss = tf.reduce_sum(self.losses)

        self.train_op = train_module.get_train_op(
            self.totalloss, self.optimizer, lr=lr, decay=decay
        )

        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    @timing
    def train(
        self,
        epochs=None,
        batch_size=None,
        display_every=1000,
        uncertainty=False,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        print_model=False,
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of epochs to train the model.
            batch_size: Integer or ``None``. Not fully supported yet.
            display_every: Integer. Print the loss and metrics every this steps.
            uncertainty: Boolean. If ``True``, use Monte-Carlo Dropout to estimate uncertainty.
            disregard_previous_best: If ``True``, disregard the previous saved best model.
            callbacks: List of ``deepxde.callbacks.Callback`` instances.
                List of callbacks to apply during training.
            model_restore_path: String. Path where parameters were previously saved.
                See ``save_path`` in `tf.train.Saver.restore <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore>`_.
            model_save_path: String. Prefix of filenames created for the checkpoint.
                See ``save_path`` in `tf.train.Saver.save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#save>`_.
            print_model: If ``True``, print the values of all variables.
        """
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if self.train_state.step == 0:
            print("Initializing variables...")
            self.sess.run(tf.global_variables_initializer())
        else:
            guarantee_initialized_variables(self.sess)
        if model_restore_path is not None:
            print("Restoring model from {} ...".format(model_restore_path))
            self.saver.restore(self.sess, model_restore_path)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test(uncertainty)
        self.callbacks.on_train_begin()
        if train_module.is_scipy_opts(self.optimizer):
            self._train_scipy(display_every, uncertainty)
        else:
            if epochs is None:
                raise ValueError("No epochs for {}.".format(self.optimizer))
            self._train_sgd(epochs, display_every, uncertainty)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        if print_model:
            self.print_model()
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def evaluate(self, x, y, callbacks=None):
        """Returns the loss values & metrics values for the model in test mode."""
        raise NotImplementedError(
            "Model.evaluate to be implemented. Alternatively, use Model.predict."
        )

    def predict(self, x, operator=None, callbacks=None):
        """Generates output predictions for the input samples."""
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        self.callbacks.on_predict_begin()
        if operator is None:
            y = self.sess.run(
                self.net.outputs, feed_dict=self.net.feed_dict(False, False, 2, x)
            )
        else:
            if get_num_args(operator) == 2:
                op = operator(self.net.inputs, self.net.outputs)
            elif get_num_args(operator) == 3:
                op = operator(self.net.inputs, self.net.outputs, x)
            y = self.sess.run(
                op,
                feed_dict=self.net.feed_dict(False, False, 2, x),
            )
        self.callbacks.on_predict_end()
        return y

    def _open_tfsession(self):
        if self.sess is not None:
            return
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.train_state.set_tfsession(self.sess)

    def _close_tfsession(self):
        self.sess.close()

    def _train_sgd(self, epochs, display_every, uncertainty):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self.sess.run(
                self.train_op,
                feed_dict=self.net.feed_dict(
                    True,
                    True,
                    0,
                    self.train_state.X_train,
                    self.train_state.y_train,
                    self.train_state.train_aux_vars,
                ),
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == epochs:
                self._test(uncertainty)

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_scipy(self, display_every, uncertainty):
        def loss_callback(loss_train):
            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0:
                self.train_state.loss_train = loss_train
                self.train_state.loss_test = None
                self.train_state.metrics_test = None
                self.losshistory.append(
                    self.train_state.step, self.train_state.loss_train, None, None
                )
                display.training_display(self.train_state)

        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_op.minimize(
            self.sess,
            feed_dict=self.net.feed_dict(
                True,
                True,
                0,
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            ),
            fetches=[self.losses],
            loss_callback=loss_callback,
        )
        self._test(uncertainty)

    def _test(self, uncertainty):
        self.train_state.loss_train, self.train_state.y_pred_train = self.sess.run(
            [self.losses, self.net.outputs],
            feed_dict=self.net.feed_dict(
                False,
                False,
                0,
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            ),
        )

        if uncertainty:
            # TODO: support multi outputs
            losses, y_preds = [], []
            for _ in range(1000):
                loss_one, y_pred_test_one = self.sess.run(
                    [self.losses, self.net.outputs],
                    feed_dict=self.net.feed_dict(
                        False,
                        True,
                        1,
                        self.train_state.X_test,
                        self.train_state.y_test,
                        self.train_state.test_aux_vars,
                    ),
                )
                losses.append(loss_one)
                y_preds.append(y_pred_test_one)
            self.train_state.loss_test = np.mean(losses, axis=0)
            self.train_state.y_pred_test = np.mean(y_preds, axis=0)
            self.train_state.y_std_test = np.std(y_preds, axis=0)
        else:
            self.train_state.loss_test, self.train_state.y_pred_test = self.sess.run(
                [self.losses, self.net.outputs],
                feed_dict=self.net.feed_dict(
                    False,
                    False,
                    1,
                    self.train_state.X_test,
                    self.train_state.y_test,
                    self.train_state.test_aux_vars,
                ),
            )

        if isinstance(self.net.targets, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.net.targets))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )
        display.training_display(self.train_state)

    def state_dict(self):
        """Returns a dictionary containing all variables."""
        destination = OrderedDict()
        variables_names = [v.name for v in tf.global_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            destination[k] = v
        return destination

    def print_model(self):
        """Prints all trainable variables."""
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)

    def save(self, save_path, protocol="tf.train.Saver", verbose=0):
        """Saves all variables to a disk file.

        Args:
            protocol (string): If `protocol` is "tf.train.Saver", save using `tf.train.Save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#attributes>`_.
                If `protocol` is "pickle", save using the Python pickle module.
                Only "tf.train.Saver" protocol supports ``restore()``.
        """
        if verbose > 0:
            print(
                "Epoch {}: saving model to {}-{} ...\n".format(
                    self.train_state.epoch, save_path, self.train_state.epoch
                )
            )
        if protocol == "tf.train.Saver":
            self.saver.save(self.sess, save_path, global_step=self.train_state.epoch)
        elif protocol == "pickle":
            with open("{}-{}.pkl".format(save_path, self.train_state.epoch), "wb") as f:
                pickle.dump(self.state_dict(), f)

    def restore(self, save_path, verbose=0):
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        self.saver.restore(self.sess, save_path)


class TrainState(object):
    def __init__(self):
        self.epoch, self.step = 0, 0

        self.sess = None

        # Data
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        # Results of current step
        self.y_pred_train = None
        self.loss_train, self.loss_test = None, None
        self.y_pred_test, self.y_std_test = None, None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_y, self.best_ystd = None, None
        self.best_metrics = None

    def set_tfsession(self, sess):
        self.sess = sess

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train, self.y_train, self.train_aux_vars = (
            X_train,
            y_train,
            train_aux_vars,
        )

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test, self.y_test, self.test_aux_vars = X_test, y_test, test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y, self.best_ystd = self.y_pred_test, self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf

    def packed_data(self):
        def merge_values(values):
            if values is None:
                return None
            return np.hstack(values) if isinstance(values, (list, tuple)) else values

        X_train = merge_values(self.X_train)
        y_train = merge_values(self.y_train)
        X_test = merge_values(self.X_test)
        y_test = merge_values(self.y_test)
        best_y = merge_values(self.best_y)
        best_ystd = merge_values(self.best_ystd)
        return X_train, y_train, X_test, y_test, best_y, best_ystd


class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
