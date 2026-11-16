import os
import sys
import time

import numpy as np

from . import config
from . import gradients as grad
from . import utils
from .backend import backend_name, jax, paddle, tf, torch
import matplotlib
import matplotlib.pyplot as plt

class Callback:
    """Callback base class.

    Attributes:
        model: instance of ``Model``. Reference of the model being trained.
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
        callbacks: List of ``Callback`` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = list(callbacks)
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
        filepath (string): Prefix of filenames to save the model file.
        verbose: Verbosity mode, 0 or 1.
        save_better_only: If True, only save a better model according to the quantity
            monitored. Model is only checked at validation step according to
            ``display_every`` in ``Model.train``.
        period: Interval (number of epochs) between checkpoints.
        monitor: The loss function that is monitored. Either 'train loss' or 'test loss'.
    """

    def __init__(
        self,
        filepath,
        verbose=0,
        save_better_only=False,
        period=1,
        monitor="train loss",
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = period

        self.monitor = monitor
        self.monitor_op = np.less
        self.epochs_since_last_save = 0
        self.best = np.inf

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            current = self.get_monitor_value()
            if self.monitor_op(current, self.best):
                save_path = self.model.save(self.filepath, verbose=0)
                if self.verbose > 0:
                    print(
                        "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                            self.model.train_state.iteration,
                            self.monitor,
                            self.best,
                            current,
                            save_path,
                        )
                    )
                self.best = current
        else:
            self.model.save(self.filepath, verbose=self.verbose)

    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


class EarlyStopping(Callback):
    """Stop training when a monitored quantity (training or testing loss) has stopped improving.
    Only checked at validation step according to ``display_every`` in ``Model.train``.

    Args:
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        monitor: The loss function that is monitored. Either 'loss_train' or 'loss_test'
        start_from_epoch: Number of epochs to wait before starting
            to monitor improvement. This allows for a warm-up period in which
            no improvement is expected and thus training will not be stopped.
    """

    def __init__(
        self,
        min_delta=0,
        patience=0,
        baseline=None,
        monitor="loss_train",
        start_from_epoch=0,
    ):
        super().__init__()

        self.baseline = baseline
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.start_from_epoch = start_from_epoch

        self.monitor_op = np.less
        self.min_delta *= -1

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self):
        if self.model.train_state.iteration < self.start_from_epoch:
            return
        current = self.get_monitor_value()
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.train_state.iteration
                self.model.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))

    def get_monitor_value(self):
        if self.monitor == "loss_train":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "loss_test":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


class Timer(Callback):
    """Stop training when training time reaches the threshold.
    This Timer starts after the first call of `on_train_begin`.

    Args:
        available_time (float): Total time (in minutes) available for the training.
    """

    def __init__(self, available_time):
        super().__init__()

        self.threshold = available_time * 60  # convert to seconds
        self.t_start = None

    def on_train_begin(self):
        if self.t_start is None:
            self.t_start = time.time()

    def on_epoch_end(self):
        if time.time() - self.t_start > self.threshold:
            self.model.stop_training = True
            print(
                "\nStop training as time used up. time used: {:.1f} mins, epoch trained: {}".format(
                    (time.time() - self.t_start) / 60, self.model.train_state.iteration
                )
            )


class DropoutUncertainty(Callback):
    """Uncertainty estimation via MC dropout.

    References:
        `Y. Gal, & Z. Ghahramani. Dropout as a Bayesian approximation: Representing
        model uncertainty in deep learning. International Conference on Machine
        Learning, 2016 <https://arxiv.org/abs/1506.02142>`_.

    Warning:
        This cannot be used together with other techniques that have different behaviors
        during training and testing, such as batch normalization.
    """

    def __init__(self, period=1000):
        super().__init__()
        self.period = period
        self.epochs_since_last = 0

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            y_preds = []
            for _ in range(1000):
                y_pred_test_one = self.model._outputs(
                    True, self.model.train_state.X_test
                )
                y_preds.append(y_pred_test_one)
            self.model.train_state.y_std_test = np.std(y_preds, axis=0)

    def on_train_end(self):
        self.on_epoch_end()


class VariableValue(Callback):
    """Get the variable values.

    Args:
        var_list: A `TensorFlow Variable <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
            or a list of TensorFlow Variable.
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
    """

    def __init__(self, var_list, period=1, filename=None, precision=2):
        super().__init__()
        self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.period = period
        self.precision = precision

        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0

    def on_train_begin(self):
        if backend_name == "tensorflow.compat.v1":
            self.value = self.model.sess.run(self.var_list)
        elif backend_name == "tensorflow":
            self.value = [var.numpy() for var in self.var_list]
        elif backend_name in ["pytorch", "paddle"]:
            self.value = [var.detach().item() for var in self.var_list]
        elif backend_name == "jax":
            self.value = [var.value for var in self.var_list]

        print(
            self.model.train_state.iteration,
            utils.list_to_str(self.value, precision=self.precision),
            file=self.file,
        )
        self.file.flush()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def on_train_end(self):
        if not self.epochs_since_last == 0:
            self.on_train_begin()

    def get_value(self):
        """Return the variable values."""
        return self.value


class OperatorPredictor(Callback):
    """Generates operator values for the input samples.

    Args:
        x: The input data.
        op: The operator with inputs (x, y).
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
    """

    def __init__(self, x, op, period=1, filename=None, precision=2):
        super().__init__()
        self.x = x
        self.op = op
        self.period = period
        self.precision = precision

        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0

    def init(self):
        if backend_name == "tensorflow.compat.v1":
            self.tf_op = self.op(self.model.net.inputs, self.model.net.outputs)
        elif backend_name == "tensorflow":

            @tf.function
            def op(inputs):
                y = self.model.net(inputs)
                return self.op(inputs, y)

            self.tf_op = op
        elif backend_name == "pytorch":
            self.x = torch.as_tensor(self.x)
            self.x.requires_grad_()
        elif backend_name == "jax":

            @jax.jit
            def op(inputs, params):
                y_fn = lambda _x: self.model.net.apply(params, _x)
                return self.op(inputs, (y_fn(inputs), y_fn))

            self.jax_op = op
        elif backend_name == "paddle":
            self.x = paddle.to_tensor(self.x, stop_gradient=False)

    def on_train_begin(self):
        self.on_predict_end()
        print(
            self.model.train_state.iteration,
            utils.list_to_str(self.value.flatten().tolist(), precision=self.precision),
            file=self.file,
        )
        self.file.flush()

    def on_train_end(self):
        if not self.epochs_since_last == 0:
            self.on_train_begin()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def on_predict_end(self):
        if backend_name == "tensorflow.compat.v1":
            self.value = self.model.sess.run(
                self.tf_op, feed_dict=self.model.net.feed_dict(False, self.x)
            )
        elif backend_name == "tensorflow":
            self.value = utils.to_numpy(self.tf_op(self.x))
        elif backend_name == "pytorch":
            self.model.net.eval()
            outputs = self.model.net(self.x)
            self.value = utils.to_numpy(self.op(self.x, outputs))
        elif backend_name == "jax":
            self.value = utils.to_numpy(self.jax_op(self.x, self.model.net.params))
        elif backend_name == "paddle":
            self.model.net.eval()
            outputs = self.model.net(self.x)
            self.value = utils.to_numpy(self.op(self.x, outputs))

    def get_value(self):
        return self.value


class FirstDerivative(OperatorPredictor):
    """Generates the first order derivative of the outputs with respect to the inputs.

    Args:
        x: The input data.
        component_x (int): Input component for the derivative (default: 0).
        component_y (int): Output component for the derivative (default: 0).
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
    """

    def __init__(
        self, x, component_x=0, component_y=0, period=1, filename=None, precision=2
    ):
        def first_derivative(x, y):
            return grad.jacobian(y, x, i=component_y, j=component_x)

        super().__init__(
            x, first_derivative, period=period, filename=filename, precision=precision
        )


class MovieDumper(Callback):
    """Dump a movie to show the training progress of the function along a line.

    Args:
        spectrum: If True, dump the spectrum of the Fourier transform.
    """

    def __init__(
        self,
        filename,
        x1,
        x2,
        num_points=100,
        period=1,
        component=0,
        save_spectrum=False,
        y_reference=None,
    ):
        super().__init__()
        self.filename = filename
        x1 = np.array(x1)
        x2 = np.array(x2)
        self.x = (
            x1 + (x2 - x1) / (num_points - 1) * np.arange(num_points)[:, None]
        ).astype(dtype=config.real(np))
        self.period = period
        self.component = component
        self.save_spectrum = save_spectrum
        self.y_reference = y_reference

        self.y = []
        self.spectrum = []
        self.epochs_since_last_save = 0

    def on_train_begin(self):
        self.y.append(self.model._outputs(False, self.x)[:, self.component])
        if self.save_spectrum:
            A = np.fft.rfft(self.y[-1])
            self.spectrum.append(np.abs(A))

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.on_train_begin()

    def on_train_end(self):
        fname_x = self.filename + "_x.txt"
        fname_y = self.filename + "_y.txt"
        fname_movie = self.filename + "_y.gif"
        print(
            "\nSaving the movie of function to {}, {}, {}...".format(
                fname_x, fname_y, fname_movie
            )
        )
        np.savetxt(fname_x, self.x)
        np.savetxt(fname_y, np.array(self.y))
        if self.y_reference is None:
            utils.save_animation(fname_movie, np.ravel(self.x), self.y)
        else:
            y_reference = np.ravel(self.y_reference(self.x))
            utils.save_animation(
                fname_movie, np.ravel(self.x), self.y, y_reference=y_reference
            )

        if self.save_spectrum:
            fname_spec = self.filename + "_spectrum.txt"
            fname_movie = self.filename + "_spectrum.gif"
            print(
                "Saving the movie of spectrum to {}, {}...".format(
                    fname_spec, fname_movie
                )
            )
            np.savetxt(fname_spec, np.array(self.spectrum))
            xdata = np.arange(len(self.spectrum[0]))
            if self.y_reference is None:
                utils.save_animation(fname_movie, xdata, self.spectrum, logy=True)
            else:
                A = np.fft.rfft(y_reference)
                utils.save_animation(
                    fname_movie, xdata, self.spectrum, logy=True, y_reference=np.abs(A)
                )


class TrainingMonitor(Callback):
    """Live-plot the predicted solution and the loss history during training.

    Every `period` epochs, this callback redraws the current network
    prediction over `x_plot` (optionally against a reference solution), and
    the train/test loss history on a log scale. Unlike ``MovieDumper``,
    which only writes an animation to disk in ``on_train_end``, or
    ``dde.saveplot``, which produces a single static plot after training
    finishes, this callback gives feedback while the model is still
    training.

    Both ODEs (`x_plot` of shape (N, 1), e.g. `y = f(t)`) and 2D
    space-time PDEs (`x_plot` of shape (N, 2), e.g. `y = f(x, t)` as in
    ``diffusion_1d.py``) are supported. For the 2D case, the solution is
    shown as a scatter plot colored by `y`, with a matching panel for
    `y_reference` (if given) sharing the same color scale for an
    at-a-glance comparison.

    Args:
        period (int): Interval (number of epochs) between plot updates.
        component (int): Which component of the solution to plot.
        x_plot: Points at which the solution is evaluated and plotted, of
            shape (N, 1) for `y = f(t)`-like problems, or (N, 2) for
            `y = f(x, t)`-like problems.
        y_reference: A function `y_reference(x_plot)` returning the
            reference (e.g., exact) solution for comparison. If ``None``,
            only the predicted solution is shown.
        show_loss (bool): If True, also plot the training/testing loss
            history in an additional subplot.

    Warning:
        Live plotting requires an interactive Matplotlib backend. In
        headless environments (e.g., CI, servers without a display) this
        callback automatically disables live plotting instead of raising
        an error, so training is never interrupted.
    """

    def __init__(
        self,
        period=100,
        component=0,
        x_plot=None,
        y_reference=None,
        show_loss=True,
    ):
        super().__init__()
        if x_plot is None:
            raise ValueError("`x_plot` must be provided.")
        self.period = period
        self.component = component
        self.x_plot = np.asarray(x_plot, dtype=config.real(np))
        if self.x_plot.ndim == 1:
            self.x_plot = self.x_plot[:, None]
        if self.x_plot.shape[1] not in (1, 2):
            raise ValueError(
                "TrainingMonitor only supports 1D (e.g. y = f(t)) or 2D "
                "(e.g. y = f(x, t)) `x_plot`, got {} columns.".format(
                    self.x_plot.shape[1]
                )
            )
        self.dim = self.x_plot.shape[1]
        self.y_reference = y_reference
        self.show_loss = show_loss

        self.epochs_since_last = 0
        self.enabled = True
        self.plt = None
        self.fig = None
        self.ax_sol = None
        self.ax_ref = None
        self.ax_loss = None
        self.cbar_sol = None
        self.cbar_ref = None

    def on_train_begin(self):
        self.epochs_since_last = 0

        if not self._has_display(matplotlib):
            self.enabled = False
            print(
                "TrainingMonitor: no interactive display detected; "
                "live plotting is disabled for this run."
            )
            return

        self.plt = plt
        plt.ion()
        self.ax_ref = None
        self.cbar_sol = None
        self.cbar_ref = None
        n_sol_axes = 1 if (self.dim == 1 or self.y_reference is None) else 2
        n_axes = n_sol_axes + (1 if self.show_loss else 0)
        self.fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 4))
        axes = np.atleast_1d(axes)
        i = 0
        self.ax_sol = axes[i]
        i += 1
        if n_sol_axes == 2:
            self.ax_ref = axes[i]
            i += 1
        self.ax_loss = axes[i] if self.show_loss else None
        self._redraw()

    @staticmethod
    def _has_display(matplotlib_module):
        """Best-effort check for an interactive display/backend."""
        if matplotlib_module.get_backend().lower() == "agg":
            return False
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            return False
        return True

    def on_epoch_end(self):
        if not self.enabled:
            return
        self.epochs_since_last += 1
        if self.epochs_since_last < self.period:
            return
        self.epochs_since_last = 0
        self._update_plot()

    def on_train_end(self):
        if self.enabled and self.fig is not None:
            self._update_plot()

    def _update_plot(self):
        try:
            y_pred = self.model.predict(self.x_plot)[:, self.component]

            if self.dim == 1:
                self._plot_1d(y_pred)
            else:
                self._plot_2d(y_pred)

            if self.show_loss:
                self._plot_loss()

            self._redraw()
        except Exception as e:  # pylint: disable=broad-except
            # Never let a plotting error interrupt training.
            # Eventual improvment, here we can manage multipl Exception types
            self.enabled = False
            print("TrainingMonitor: disabling live plot due to error: {}".format(e))

    def _plot_1d(self, y_pred):
        x = np.ravel(self.x_plot)
        self.ax_sol.cla()
        self.ax_sol.plot(x, y_pred, "--r", label="Predicted")
        if self.y_reference is not None:
            y_ref = np.ravel(self.y_reference(self.x_plot))
            self.ax_sol.plot(x, y_ref, "-k", label="Reference")
        self.ax_sol.set_xlabel("x")
        self.ax_sol.set_ylabel("y")
        self.ax_sol.set_title("Epoch {}".format(self.model.train_state.iteration))
        self.ax_sol.legend()

    def _plot_2d(self, y_pred):
        x0, x1 = self.x_plot[:, 0], self.x_plot[:, 1]
        y_ref = None
        if self.y_reference is not None:
            y_ref = np.ravel(self.y_reference(self.x_plot))
            vmin = min(y_pred.min(), y_ref.min())
            vmax = max(y_pred.max(), y_ref.max())
        else:
            vmin, vmax = y_pred.min(), y_pred.max()

        if self.cbar_sol is not None:
            self.cbar_sol.remove()
        self.ax_sol.cla()
        sc = self.ax_sol.scatter(x0, x1, c=y_pred, cmap="jet", vmin=vmin, vmax=vmax, s=10)
        self.ax_sol.set_xlabel("x")
        self.ax_sol.set_ylabel("t")
        self.ax_sol.set_title(
            "Predicted, epoch {}".format(self.model.train_state.iteration)
        )
        self.cbar_sol = self.fig.colorbar(sc, ax=self.ax_sol)

        if self.ax_ref is not None:
            if self.cbar_ref is not None:
                self.cbar_ref.remove()
            self.ax_ref.cla()
            sc_ref = self.ax_ref.scatter(
                x0, x1, c=y_ref, cmap="jet", vmin=vmin, vmax=vmax, s=10
            )
            self.ax_ref.set_xlabel("x")
            self.ax_ref.set_ylabel("t")
            self.ax_ref.set_title("Reference")
            self.cbar_ref = self.fig.colorbar(sc_ref, ax=self.ax_ref)

    def _plot_loss(self):
        loss_history = self.model.losshistory
        loss_train = [np.sum(loss) for loss in loss_history.loss_train]
        loss_test = [np.sum(loss) for loss in loss_history.loss_test]
        self.ax_loss.cla()
        self.ax_loss.semilogy(loss_history.steps, loss_train, label="Train loss")
        self.ax_loss.semilogy(loss_history.steps, loss_test, label="Test loss")
        self.ax_loss.set_xlabel("# Steps")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()

    def _redraw(self):
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)


class PDEPointResampler(Callback):
    """Resample the training points for PDE and/or BC losses every given period.

    Args:
        period: How often to resample the training points (default is 100 iterations).
        pde_points: If True, resample the training points for PDE losses (default is
            True).
        bc_points: If True, resample the training points for BC losses (default is
            False; only supported by PyTorch and PaddlePaddle backend currently).
    """

    def __init__(self, period=100, pde_points=True, bc_points=False):
        super().__init__()
        self.period = period
        self.pde_points = pde_points
        self.bc_points = bc_points

        self.num_bcs_initial = None
        self.epochs_since_last_resample = 0

    def on_train_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0
        self.model.data.resample_train_points(self.pde_points, self.bc_points)

        if not np.array_equal(self.num_bcs_initial, self.model.data.num_bcs):
            print("Initial value of self.num_bcs:", self.num_bcs_initial)
            print("self.model.data.num_bcs:", self.model.data.num_bcs)
            raise ValueError(
                "`num_bcs` changed! Please update the loss function by `model.compile`."
            )
