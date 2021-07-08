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
from . import utils
from .backend import backend_name, tf
from .callbacks import CallbackList


class Model(object):
    """A ``Model`` trains a ``Map`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.maps.Map`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.batch_size = None
        self.callbacks = None

        self.losses = None  # Tensor or callable
        self.train_step = None  # Tensor or callable
        self.metrics = None
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False

        if backend_name == "tensorflow.compat.v1":
            self.sess = None
            self.saver = None

    @utils.timing
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
            loss: If the same loss is used for all errors, then `loss` is a String (name
                of objective function) or objective function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay: Tuple. Name and parameters of decay to the initial learning rate. One
                of the following options:

                - `inverse time decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/inverse_time_decay>`_: ("inverse time", decay_steps, decay_rate)
                - `cosine decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
        """
        print("Compiling model...")

        if backend_name == "tensorflow.compat.v1":
            if not self.net.built:
                self.net.build()
            if self.sess is None:
                self.sess = tf.Session()
                self.saver = tf.train.Saver(max_to_keep=None)

        self.opt_name = optimizer

        loss_fn = losses_module.get(loss)
        if backend_name == "tensorflow.compat.v1":
            # Data losses
            losses = self.data.losses(self.net.targets, self.net.outputs, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses.append(tf.losses.get_regularization_loss())
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
                self.losshistory.set_loss_weights(loss_weights)
            total_loss = tf.math.reduce_sum(losses)
            # Tensor: losses and train_step
            self.losses = losses
            self.train_step = train_module.get_train_op(
                total_loss, self.opt_name, lr=lr, decay=decay
            )
        elif backend_name == "tensorflow":

            def compute_losses(targets, outputs):
                # Data losses
                losses = self.data.losses(targets, outputs, loss_fn, self)
                if not isinstance(losses, list):
                    losses = [losses]
                # TODO: Regularization loss
                losses = tf.convert_to_tensor(losses)
                # TODO: Weighted losses
                return losses

            # TODO: Support different optimizers
            # TODO: Support learning rate decay
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

            @tf.function
            def train_step(inputs, targets):
                with tf.GradientTape() as tape:
                    outputs = self.net(inputs, training=True)
                    losses = compute_losses(targets, outputs)
                    total_loss = tf.math.reduce_sum(losses)
                grads = tape.gradient(total_loss, self.net.trainable_variables)
                opt.apply_gradients(zip(grads, self.net.trainable_variables))

            # Callable: losses and train_step
            self.losses = compute_losses
            self.train_step = train_step

        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    @utils.timing
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
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of epochs to train the model.
            batch_size: Integer or ``None``. If you solve PDEs via ``dde.data.PDE`` or
                ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                `dde.callbacks.PDEResidualResampler
                <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEResidualResampler>`_,
                see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
            display_every: Integer. Print the loss and metrics every this steps.
            uncertainty: Boolean. If ``True``, use Monte-Carlo Dropout to estimate
                uncertainty.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path: String. Path where parameters were previously saved.
                See ``save_path`` in `tf.train.Saver.restore <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore>`_.
            model_save_path: String. Prefix of filenames created for the checkpoint.
                See ``save_path`` in `tf.train.Saver.save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#save>`_.
        """
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if backend_name == "tensorflow.compat.v1":
            if self.train_state.step == 0:
                print("Initializing variables...")
                self.sess.run(tf.global_variables_initializer())
            else:
                utils.guarantee_initialized_variables(self.sess)

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test(uncertainty)
        self.callbacks.on_train_begin()
        if train_module.is_scipy_opts(self.opt_name):
            self._train_scipy(display_every, uncertainty)
        else:
            if epochs is None:
                raise ValueError("No epochs for {}.".format(self.opt_name))
            self._train_sgd(epochs, display_every, uncertainty)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
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
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(False, False, 2, x)
            if operator is None:
                y = self.sess.run(self.net.outputs, feed_dict=feed_dict)
            else:
                if utils.get_num_args(operator) == 2:
                    op = operator(self.net.inputs, self.net.outputs)
                elif utils.get_num_args(operator) == 3:
                    op = operator(self.net.inputs, self.net.outputs, x)
                y = self.sess.run(op, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            if operator is None:
                y = self.net(x).numpy()
            else:
                raise NotImplementedError(
                    "Model.predict for operator has not been implemented for backend tensorflow."
                )
        self.callbacks.on_predict_end()
        return y

    def _train_sgd(self, epochs, display_every, uncertainty):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            if backend_name == "tensorflow.compat.v1":
                feed_dict = self.net.feed_dict(
                    True,
                    True,
                    0,
                    self.train_state.X_train,
                    self.train_state.y_train,
                    self.train_state.train_aux_vars,
                )
                self.sess.run(self.train_step, feed_dict=feed_dict)
            elif backend_name == "tensorflow":
                # TODO: Support self.train_state.train_aux_vars, data_id
                self.train_step(self.train_state.X_train, self.train_state.y_train)

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == epochs:
                self._test(uncertainty)

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_scipy(self, display_every, uncertainty):
        # TODO: backend tensorflow
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
        feed_dict = self.net.feed_dict(
            True,
            True,
            0,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_step.minimize(
            self.sess,
            feed_dict=feed_dict,
            fetches=[self.losses],
            loss_callback=loss_callback,
        )
        self._test(uncertainty)

    def _test(self, uncertainty):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(
                False,
                False,
                0,
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )
            self.train_state.loss_train, self.train_state.y_pred_train = self.sess.run(
                [self.losses, self.net.outputs], feed_dict=feed_dict
            )
        elif backend_name == "tensorflow":
            outputs = self.net(self.train_state.X_train, training=False)
            self.train_state.loss_train = self.losses(
                self.train_state.y_train, outputs
            ).numpy()
            self.train_state.y_pred_train = outputs.numpy()

        if uncertainty:
            # TODO: support multi outputs
            # TODO: backend tensorflow
            losses, y_preds = [], []
            feed_dict = self.net.feed_dict(
                False,
                True,
                1,
                self.train_state.X_test,
                self.train_state.y_test,
                self.train_state.test_aux_vars,
            )
            for _ in range(1000):
                loss_one, y_pred_test_one = self.sess.run(
                    [self.losses, self.net.outputs], feed_dict=feed_dict
                )
                losses.append(loss_one)
                y_preds.append(y_pred_test_one)
            self.train_state.loss_test = np.mean(losses, axis=0)
            self.train_state.y_pred_test = np.mean(y_preds, axis=0)
            self.train_state.y_std_test = np.std(y_preds, axis=0)
        else:
            if backend_name == "tensorflow.compat.v1":
                feed_dict = self.net.feed_dict(
                    False,
                    False,
                    1,
                    self.train_state.X_test,
                    self.train_state.y_test,
                    self.train_state.test_aux_vars,
                )
                (
                    self.train_state.loss_test,
                    self.train_state.y_pred_test,
                ) = self.sess.run([self.losses, self.net.outputs], feed_dict=feed_dict)
            elif backend_name == "tensorflow":
                outputs = self.net(self.train_state.X_test, training=False)
                self.train_state.loss_test = self.losses(
                    self.train_state.y_test, outputs
                ).numpy()
                self.train_state.y_pred_test = outputs.numpy()

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
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
        # TODO: backend tensorflow
        destination = OrderedDict()
        variables_names = [v.name for v in tf.global_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            destination[k] = v
        return destination

    def save(self, save_path, protocol="tf.train.Saver", verbose=0):
        """Saves all variables to a disk file.

        Args:
            protocol (string): If `protocol` is "tf.train.Saver", save using
                `tf.train.Save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#attributes>`_.
                If `protocol` is "pickle", save using the Python pickle module. Only
                "tf.train.Saver" protocol supports ``restore()``.
        """
        # TODO: backend tensorflow
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
        """Restore all variables from a disk file."""
        # TODO: backend tensorflow
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        self.saver.restore(self.sess, save_path)

    def print_model(self):
        """Prints all trainable variables."""
        # TODO: backend tensorflow
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)


class TrainState(object):
    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.train_aux_vars = None
        self.X_test = None
        self.y_test = None
        self.test_aux_vars = None

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Test results
        self.loss_test = None
        self.y_pred_test = None
        self.y_std_test = None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train = np.inf
        self.best_loss_test = np.inf
        self.best_y = None
        self.best_ystd = None
        self.best_metrics = None

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train = X_train
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test = X_test
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
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
