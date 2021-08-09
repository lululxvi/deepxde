from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from collections import OrderedDict

import numpy as np

from . import display
from . import gradients as grad
from . import losses as losses_module
from . import metrics as metrics_module
from . import optimizers
from . import utils
from .backend import backend_name, tf, torch
from .callbacks import CallbackList


class Model(object):
    """A ``Model`` trains a ``NN`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.nn.NN`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.batch_size = None
        self.callbacks = None
        self.metrics = None
        self.external_trainable_variables = []
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False

        # Backend-dependent attributes
        # Tensor or callable
        self.outputs = None
        self.outputs_losses = None
        self.train_step = None
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
        external_trainable_variables=None,
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

                - `inverse time decay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: ("inverse time", decay_steps, decay_rate)
                - `cosine decay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
            external_trainable_variables: A trainable ``tf.Variable`` object or a list
                of trainable ``tf.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered. If the backend is
                tensorflow.compat.v1, `external_trainable_variables` is ignored, and all
                trainable ``tf.Variable`` objects are automatically collected.
        """
        print("Compiling model...")

        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            if backend_name == "tensorflow.compat.v1":
                print(
                    "Warning: For the backend tensorflow.compat.v1, "
                    "`external_trainable_variables` is ignored, and all trainable "
                    "``tf.Variable`` objects are automatically collected."
                )
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
            self.external_trainable_variables = external_trainable_variables

        if backend_name == "tensorflow.compat.v1":
            self._compile_tensorflow_compat_v1(lr, loss_fn, decay, loss_weights)
        elif backend_name == "tensorflow":
            self._compile_tensorflow(lr, loss_fn, decay, loss_weights)
        elif backend_name == "pytorch":
            self._compile_pytorch(lr, loss_fn, decay, loss_weights)

        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    def _compile_tensorflow_compat_v1(self, lr, loss_fn, decay, loss_weights):
        """tensorflow.compat.v1"""
        if not self.net.built:
            self.net.build()
        if self.sess is None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver(max_to_keep=None)

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

        # Tensors
        self.outputs = self.net.outputs
        self.outputs_losses = [self.net.outputs, losses]
        self.train_step = optimizers.get(
            total_loss, self.opt_name, learning_rate=lr, decay=decay
        )

    def _compile_tensorflow(self, lr, loss_fn, decay, loss_weights):
        """tensorflow"""

        # TODO: Avoid creating multiple graphs by using tf.TensorSpec.
        @tf.function
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        # TODO: Avoid creating multiple graphs by using tf.TensorSpec.
        @tf.function
        def outputs_losses(training, inputs, targets, auxiliary_vars):
            # TODO: Add training
            # self.net.training = training
            self.net.inputs = inputs
            self.net.targets = targets
            self.net.auxiliary_vars = auxiliary_vars
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = self.net(inputs, training=training)
            # Data losses
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses += [tf.math.reduce_sum(self.net.losses)]
            losses = tf.convert_to_tensor(losses)
            # TODO: Weighted losses
            if loss_weights is not None:
                raise NotImplementedError(
                    "Backend tensorflow doesn't support loss_weights"
                )
            return outputs_, losses

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses(True, inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))

        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = train_step

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """pytorch"""

        def outputs(inputs):
            # TODO: Add training
            # TODO: Use torch.no_grad() if training is False
            inputs = torch.from_numpy(inputs)
            inputs.requires_grad_()
            self.net.inputs = inputs
            return self.net(inputs)

        def outputs_losses(inputs, targets):
            outputs_ = outputs(inputs)
            # Data losses
            if targets is not None:
                targets = torch.from_numpy(targets)
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # TODO: regularization
            # TODO: Weighted losses
            if loss_weights is not None:
                raise NotImplementedError(
                    "Backend pytorch doesn't support loss_weights"
                )
            losses = torch.stack(losses)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        # Another way is using Per-parameter options
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options
        trainable_variables = (
            list(self.net.parameters()) + self.external_trainable_variables
        )
        opt = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=lr, decay=decay
        )

        def train_step(inputs, targets):
            losses = outputs_losses(inputs, targets)[1]
            total_loss = torch.sum(losses)
            opt.zero_grad()
            total_loss.backward()
            opt.step()

        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = train_step

    def _outputs(self, training, inputs):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs)
            return self.sess.run(self.outputs, feed_dict=feed_dict)
        if backend_name == "tensorflow":
            outs = self.outputs(training, inputs)
        elif backend_name == "pytorch":
            # TODO: training
            with torch.no_grad():
                outs = self.outputs(inputs)
        if isinstance(outs, (list, tuple)):
            return [out.numpy() for out in outs]
        return outs.numpy()

    def _train_step(self, inputs, targets, auxiliary_vars):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(True, inputs, targets, auxiliary_vars)
            self.sess.run(self.train_step, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            self.train_step(inputs, targets, auxiliary_vars)
        elif backend_name == "pytorch":
            # TODO: auxiliary_vars
            self.train_step(inputs, targets)

    def _run(self, fetches, training, inputs, targets, auxiliary_vars):
        """Runs one "step" of computation of tensors or callables in `fetches`."""
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs, targets, auxiliary_vars)
            return self.sess.run(fetches, feed_dict=feed_dict)
        if backend_name == "tensorflow":
            outs = fetches(training, inputs, targets, auxiliary_vars)
            return None if outs is None else [out.numpy() for out in outs]
        if backend_name == "pytorch":
            # TODO: Use torch.no_grad() in _test() and predict()
            # TODO: training, auxiliary_vars
            outs = fetches(inputs, targets)
            return None if outs is None else [out.detach().numpy() for out in outs]

    @utils.timing
    def train(
        self,
        epochs=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of iterations to train the model. Note: It is the
                number of iterations, not the number of epochs.
            batch_size: Integer or ``None``. If you solve PDEs via ``dde.data.PDE`` or
                ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                `dde.callbacks.PDEResidualResampler
                <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEResidualResampler>`_,
                see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
            display_every: Integer. Print the loss and metrics every this steps.
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
        self._test()
        self.callbacks.on_train_begin()
        if optimizers.is_external_optimizer(self.opt_name):
            self._train_scipy(display_every)
        else:
            if epochs is None:
                raise ValueError("No epochs for {}.".format(self.opt_name))
            self._train_sgd(epochs, display_every)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def _train_sgd(self, epochs, display_every):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == epochs:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_scipy(self, display_every):
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
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_step.minimize(
            self.sess,
            feed_dict=feed_dict,
            fetches=[self.outputs_losses[1]],
            loss_callback=loss_callback,
        )
        self._test()

    def _test(self):
        self.train_state.y_pred_train, self.train_state.loss_train = self._run(
            self.outputs_losses,
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._run(
            self.outputs_losses,
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

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

    def predict(self, x, operator=None, callbacks=None):
        """Generates output predictions for the input samples."""
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        self.callbacks.on_predict_begin()
        if operator is None:
            y = self._outputs(False, x)
        else:
            # TODO: predict operator with auxiliary_vars
            if backend_name == "tensorflow.compat.v1":
                if utils.get_num_args(operator) == 2:
                    op = operator(self.net.inputs, self.net.outputs)
                elif utils.get_num_args(operator) == 3:
                    op = operator(self.net.inputs, self.net.outputs, x)
                y = self._run(op, False, x, None, None)
            elif backend_name == "tensorflow":
                # TODO: avoid creating the same graph every time predict is called
                # TODO: use self._run for tensorflow
                if utils.get_num_args(operator) == 2:

                    @tf.function
                    def op(inputs):
                        y = self.net(inputs)
                        return operator(inputs, y)

                elif utils.get_num_args(operator) == 3:

                    @tf.function
                    def op(inputs):
                        y = self.net(inputs)
                        return operator(inputs, y, x)

                y = op(x).numpy()
            elif backend_name == "pytorch":
                # TODO
                raise NotImplementedError(
                    "Model.predict hasn't been implemented for backend pytorch."
                )
        self.callbacks.on_predict_end()
        return y

    # def evaluate(self, x, y, callbacks=None):
    #     """Returns the loss values & metrics values for the model in test mode."""
    #     raise NotImplementedError(
    #         "Model.evaluate to be implemented. Alternatively, use Model.predict."
    #     )

    def state_dict(self):
        """Returns a dictionary containing all variables."""
        # TODO: backend tensorflow, pytorch
        if backend_name != "tensorflow.compat.v1":
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
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
        # TODO: backend tensorflow, pytorch
        if backend_name != "tensorflow.compat.v1":
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
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
        # TODO: backend tensorflow, pytorch
        if backend_name != "tensorflow.compat.v1":
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        self.saver.restore(self.sess, save_path)

    def print_model(self):
        """Prints all trainable variables."""
        # TODO: backend tensorflow, pytorch
        if backend_name != "tensorflow.compat.v1":
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
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
