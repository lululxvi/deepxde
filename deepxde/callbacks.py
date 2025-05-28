import sys
import time

import numpy as np

from . import config
from . import gradients as grad
from . import utils
from .backend import backend_name, jax, paddle, tf, torch


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


class PDEPointResampler(Callback):
    """
    Resample the training points for PDE and/or BC losses every given period.

    Args:
        period (int): How often to resample the training points (default is 100 iterations).

        name (str): Choose one of the following options:
            - 'static': Resample using the same method that was used initially.
            - 'RAR-G': Residual-based adaptive refinement with greed (only supported by PyTorch).
            - 'RAD': Residual-based adaptive distribution (only supported by PyTorch).
            - 'RAR-D': Residual-based adaptive refinement distribution (only supported by PyTorch).

        sampler (dict): Specify the sampler to resample the training points (default is {'pde_points': True, 
                                                                                         'bc_points': False, 
                                                                                         'k': 1.0, 
                                                                                         'c': 0.0, 
                                                                                         'number_of_points': 100}). 

            Only for non-residual-based methods:
            
            - 'pde_points' (bool): Whether to resample the PDE points (default is True).
            - 'bc_points' (bool): Whether to resample the BC points (default is False; always False for residual-based methods).

            Only for distribution-based adaptive refinement methods:
            
            - 'k' (float): The exponent for the residuals (default is 1.0).
            - 'c' (float): A constant determining the 'strength' of the PDF compared to randomness (default is 0.0).
            
            Only for residual-based adaptive refinement methods:
            
            - 'number_of_points' (int): The number of points sampled from the PDF and added to the original PDE training points.

        save (bool): Whether to save the new training points in the model's data (default is False).

        For more information on the residual-based adaptive refinement methods, see the paper: https://www.sciencedirect.com/science/article/pii/S0045782522006260
    """


    def __init__(self, period:int=100, name:str='static', sampler_config:dict={}):
        super().__init__()
        
        default_config = {'pde_points': True, 'bc_points': False, 'k': 1.0, 'c': 0.0, 'number_of_points': 100}
        default_config.update(sampler_config)
        
        self.period = period
        self.name = name
        self.num_bcs_initial = None
        self.epochs_since_last_resample = 0
        self.sampler_config = default_config

    def on_train_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs
            
    def generate_dense_training_set(self, num_domain:int=None):
        """Generating a training set"""
        X = np.empty((0, self.model.data.geom.dim), dtype=config.real(np))

        if self.model.data.train_distribution == "uniform":
            X = self.model.data.geom.uniform_points(num_domain, boundary=False)
        else:
            X = self.model.data.geom.random_points(num_domain, random=self.model.data.train_distribution)

        return X
    
    def generate_pdf(self, residual:np.array=None, k:float=1.0, c:float=0.0):
        """Generating the probability density function (PDF) based on the PDE residuals."""
        
        eps = np.nan_to_num(residual, nan=0.0)
        eps_k = np.abs(np.pow(eps, k))
        pdf = (eps_k / np.sum(eps_k)) + c
        if c != 0: pdf /= np.sum(pdf) # need to renormalize the PDF if c != 0 in order for np.random.choice(..., p=pdf) to work
        
        return pdf
        
    def get_residual(self, X:np.array=None):
        """Calculating the residual of the PDE for the given inputs."""
        
        if backend_name == 'pytorch':
            inputs = torch.as_tensor(X)
            inputs.requires_grad_()
            outputs = self.model.net(inputs)
            residual = self.model.data.pde(inputs, outputs)
            return residual.detach().cpu().numpy().flatten()
        else:
            raise ValueError("Unsupported backend.")

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0
        
        if self.name == 'static':
            
            self.model.data.resample_train_points(self.sampler_config['pde_points'], self.sampler_config['bc_points'])

            if not np.array_equal(self.num_bcs_initial, self.model.data.num_bcs):
                print("Initial value of self.num_bcs:", self.num_bcs_initial)
                print("self.model.data.num_bcs:", self.model.data.num_bcs)
                raise ValueError(
                    "`num_bcs` changed! Please update the loss function by `model.compile`."
                )
                
        elif self.name == 'RAR-G':

            inputs = self.generate_dense_training_set(self.model.data.num_domain)
            residual = self.get_residual(inputs)
            
            indices = np.argpartition(residual,-self.sampler_config['number_of_points'])[-self.sampler_config['number_of_points']:]
            self.model.data.add_anchors(inputs[indices, :])

        elif self.name == 'RAD':

            inputs = self.generate_dense_training_set(2*self.model.data.num_domain)
            residual = self.get_residual(inputs)

            pdf = self.generate_pdf(residual, self.sampler_config['k'], self.sampler_config['c'])
            
            indices = np.random.choice(pdf.size, size=self.model.data.num_domain, p=pdf)
            self.model.data.replace_with_anchors(inputs[indices, :])
                
        elif self.name == 'RAR-D':

            inputs = self.generate_dense_training_set(self.model.data.num_domain)
            residual = self.get_residual(inputs)

            pdf = self.generate_pdf(residual, self.sampler_config['k'], self.sampler_config['c'])
                
            indices = np.random.choice(pdf.size, size=self.sampler_config['number_of_points'], p=pdf)
            self.model.data.add_anchors(inputs[indices, :])

        else:
            raise ValueError("Unsupported sampling strategy.")