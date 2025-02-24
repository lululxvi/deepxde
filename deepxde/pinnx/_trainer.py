# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


import time
from typing import Union, Sequence, Callable, Optional

import brainstate as bst
import brainunit as u
import jax.numpy as jnp
import jax.tree
import numpy as np

from deepxde.model import LossHistory, TrainState as TrainStateBase
from deepxde.utils.internal import timing
from . import metrics as metrics_module
from .callbacks import CallbackList, Callback
from .problem.base import Problem
from .utils.display import training_display
from .utils.external import saveplot

__all__ = [
    "Trainer",
    "TrainState",
    "LossHistory",
]


class Trainer:
    """
    A ``Trainer`` trains a neural network on a ``Problem``.

    Args:
        problem: ``pinnx.problem.Problem`` instance.
        external_trainable_variables: A trainable ``brainstate.ParamState`` object or a list
                of trainable ``brainstate.ParamState`` objects. The unknown parameters in the
                physics systems that need to be recovered.
    """
    __module__ = 'deepxde.pinnx'
    optimizer: bst.optim.Optimizer  # optimizer
    problem: Problem  # problem
    params: bst.util.FlattedDict  # trainable variables

    def __init__(
        self,
        problem: Problem,
        external_trainable_variables: Union[bst.ParamState, Sequence[bst.ParamState]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the Trainer.

        Args:
            problem (Problem): The problem instance to be solved.
            external_trainable_variables (Union[bst.ParamState, Sequence[bst.ParamState]], optional): 
                External trainable variables to be included in the optimization process. 
                Can be a single ParamState or a sequence of ParamStates. Defaults to None.
            batch_size (Optional[int], optional): The batch size to be used during training. 
                If None, the entire dataset will be used. Defaults to None.

        Raises:
            ValueError: If the problem does not define an approximator.
            AssertionError: If the problem is not a Problem instance or if external_trainable_variables
                are not ParamState instances.

        Returns:
            None
        """
        # the problem
        self.problem = problem
        assert isinstance(self.problem, Problem), "problem must be a Problem instance."

        # the approximator
        if self.problem.approximator is None:
            raise ValueError("Problem must define an approximator before training.")

        # parameters and external trainable variables
        params = bst.graph.states(self.problem.approximator, bst.ParamState)
        if external_trainable_variables is None:
            external_trainable_variables = []
        else:
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
        for i, var in enumerate(external_trainable_variables):
            assert isinstance(var, bst.ParamState), ("external_trainable_variables must be a "
                                                     "list of ParamState instance.")
            params[('external_trainable_variable', i)] = var
        self.params = params

        # other useful parameters
        self.metrics = None
        self.batch_size = batch_size

        # training state
        self.train_state = TrainState()
        self.loss_history = LossHistory()
        self.stop_training = False

    @timing
    def compile(
        self,
        optimizer: bst.optim.Optimizer,
        metrics: Union[str, Sequence[str]] = None,
        measture_train_step_compile_time: bool = False,
    ):
        """
        Configures the trainer for training.

        Args:
            optimizer: String name of an optimizer, or an optimizer class instance.
            metrics: List of metrics to be evaluated by the trainer during training.
        """
        print("Compiling trainer...")

        # optimizer
        assert isinstance(optimizer, bst.optim.Optimizer), "optimizer must be an Optimizer instance."
        self.optimizer = optimizer
        self.optimizer.register_trainable_weights(self.params)

        # metrics may use trainer variables such as self.net,
        # and thus are instantiated after compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

        def fn_outputs(training: bool, inputs):
            with bst.environ.context(fit=training):
                inputs = jax.tree.map(lambda x: u.math.asarray(x), inputs, is_leaf=u.math.is_quantity)
                return self.problem.approximator(inputs)

        def fn_outputs_losses(training, inputs, targets, **kwargs):
            with bst.environ.context(fit=training):
                # inputs
                inputs = jax.tree.map(lambda x: u.math.asarray(x), inputs, is_leaf=u.math.is_quantity)

                # outputs
                outputs = self.problem.approximator(inputs)

                # targets
                if targets is not None:
                    targets = jax.tree.map(lambda x: u.math.asarray(x), targets, is_leaf=u.math.is_quantity)

                # compute losses
                if training:
                    losses = self.problem.losses_train(inputs, outputs, targets, **kwargs)
                else:
                    losses = self.problem.losses_test(inputs, outputs, targets, **kwargs)
                return outputs, losses

        def fn_outputs_losses_train(inputs, targets, **aux):
            return fn_outputs_losses(True, inputs, targets, **aux)

        def fn_outputs_losses_test(inputs, targets, **aux):
            return fn_outputs_losses(False, inputs, targets, **aux)

        def fn_train_step(inputs, targets, **aux):
            def _loss_fun():
                losses = fn_outputs_losses_train(inputs, targets, **aux)[1]
                return u.math.sum(u.math.asarray([loss.sum() for loss in jax.tree.leaves(losses)]))

            grads = bst.augment.grad(_loss_fun, grad_states=self.params)()
            self.optimizer.update(grads)

        # Callables
        self.fn_outputs = bst.compile.jit(fn_outputs, static_argnums=0)
        self.fn_outputs_losses_train = bst.compile.jit(fn_outputs_losses_train)
        self.fn_outputs_losses_test = bst.compile.jit(fn_outputs_losses_test)
        self.fn_train_step = bst.compile.jit(fn_train_step)

        if measture_train_step_compile_time:
            t0 = time.time()
            self._compile_training_step(self.batch_size)
            t1 = time.time()
            return self, t1 - t0

        return self

    @timing
    def train(
        self,
        iterations: int,
        batch_size: int = None,
        display_every: int = 1000,
        disregard_previous_best: bool = False,
        callbacks: Union[Callback, Sequence[Callback]] = None,
        model_restore_path: str = None,
        model_save_path: str = None,
        measture_train_step_time: bool = False,
    ):
        """
        Trains the trainer.

        Args:
            iterations (Integer): Number of iterations to train the trainer, i.e., number
                of times the network weights are updated.
            batch_size: Integer, tuple, or ``None``.

                - If you solve PDEs via ``pinnx.problem.PDE`` or ``pinnx.problem.TimePDE``, do not use `batch_size`,
                  and instead use `pinnx.callbacks.PDEPointResampler
                  <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEPointResampler>`_,
                  see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_resample.py>`_.
                - For DeepONet in the format of Cartesian product, if `batch_size` is an Integer,
                  then it is the batch size for the branch input;
                  if you want to also use mini-batch for the trunk net input,
                  set `batch_size` as a tuple, where the fist number is the batch size for the branch net input
                  and the second number is the batch size for the trunk net input.
            display_every (Integer): Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                trainer.
            callbacks: List of ``pinnx.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path (String): Path where parameters were previously saved.
            model_save_path (String): Prefix of filenames created for the checkpoint.
        """

        if measture_train_step_time:
            t0 = time.time()

        if self.metrics is None:
            raise ValueError("Compile the trainer before training.")

        # callbacks
        callbacks = CallbackList(callbacks=[callbacks] if isinstance(callbacks, Callback) else callbacks)
        callbacks.set_model(self)

        # disregard previous best
        if disregard_previous_best:
            self.train_state.disregard_best()

        # restore
        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training trainer...\n")
        self.stop_training = False

        # testing
        self.train_state.set_data_train(*self.problem.train_next_batch(batch_size))
        self.train_state.set_data_test(*self.problem.test())
        self._test()

        # training
        callbacks.on_train_begin()
        self._train(iterations, display_every, batch_size, callbacks)
        callbacks.on_train_end()

        # summary
        print("")
        training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)

        if measture_train_step_time:
            t1 = time.time()
            return self, t1 - t0
        return self

    def _compile_training_step(self, batch_size=None):
        # get data
        self.train_state.set_data_train(*self.problem.train_next_batch(batch_size))

        # train one batch
        self.fn_train_step.compile(self.train_state.X_train,
                                   self.train_state.y_train,
                                   **self.train_state.Aux_train)

    def _train(self, iterations, display_every, batch_size, callbacks):
        for i in range(iterations):
            callbacks.on_epoch_begin()
            callbacks.on_batch_begin()

            # get data
            self.train_state.set_data_train(*self.problem.train_next_batch(batch_size))

            # train one batch
            self.fn_train_step(self.train_state.X_train, self.train_state.y_train, **self.train_state.Aux_train)

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            callbacks.on_batch_end()
            callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _test(self):
        # evaluate the training data
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self.fn_outputs_losses_train(
            self.train_state.X_train,
            self.train_state.y_train,
            **self.train_state.Aux_train,
        )

        # evaluate the test data
        (
            self.train_state.y_pred_test,
            self.train_state.loss_test
        ) = self.fn_outputs_losses_test(
            self.train_state.X_test,
            self.train_state.y_test,
            **self.train_state.Aux_test,
        )

        # metrics
        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i],
                  self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test,
                  self.train_state.y_pred_test)
                for m in self.metrics
            ]

        # history
        self.train_state.update_best()
        self.loss_history.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        # check NaN
        if (
            jnp.isnan(jnp.asarray(jax.tree.leaves(self.train_state.loss_train))).any()
            or jnp.isnan(jnp.asarray(jax.tree.leaves(self.train_state.loss_test))).any()
        ):
            self.stop_training = True

        # display
        training_display(self.train_state)

    def predict(
        self,
        xs,
        operator: Optional[Callable] = None,
        callbacks: Union[Callback, Sequence[Callback]] = None,
    ):
        """Generates predictions for the input samples. If `operator` is ``None``,
        returns the network output, otherwise returns the output of the `operator`.

        Args:
            xs: The network inputs. A Numpy array or a tuple of Numpy arrays.
            operator: A function takes arguments (`neural_net`, `inputs`) and outputs a tensor. `inputs` and
                `outputs` are the network input and output tensors, respectively. `operator` is typically
                chosen as the PDE (used to define `pinnx.problem.PDE`) to predict the PDE residual.
            callbacks: List of ``pinnx.callbacks.Callback`` instances. List of callbacks
                to apply during prediction.
        """
        xs = jax.tree.map(
            lambda x: u.math.asarray(x, dtype=bst.environ.dftype()),
            xs,
            is_leaf=u.math.is_quantity
        )
        callbacks = CallbackList(callbacks=[callbacks] if isinstance(callbacks, Callback) else callbacks)
        callbacks.set_model(self)
        callbacks.on_predict_begin()
        ys = self.fn_outputs(False, xs)
        if operator is not None:
            ys = operator(xs, ys)
        callbacks.on_predict_end()
        return ys

    def save(self, save_path, verbose: int = 0):
        """Saves all variables to a disk file.

        Args:
            save_path (string): Prefix of filenames to save the trainer file.
            verbose (int): Verbosity mode, 0 or 1.

        Returns:
            string: Path where trainer is saved.
        """
        import braintools

        # save path
        save_path = f"{save_path}-{self.train_state.epoch}.msgpack"

        # avoid the duplicate ParamState save
        model = bst.graph.Dict(params=self.params, optimizer=self.optimizer)

        checkpoint = bst.graph.states(model).to_nest()
        braintools.file.msgpack_save(save_path, checkpoint)

        if verbose > 0:
            print(
                "Epoch {}: saving trainer to {} ...\n".format(
                    self.train_state.epoch, save_path
                )
            )
        return save_path

    def restore(self, save_path, verbose: int = 0):
        """Restore all variables from a disk file.

        Args:
            save_path (string): Path where trainer was previously saved.
            verbose (int): Verbosity mode, 0 or 1.
        """
        import braintools
        if verbose > 0:
            print("Restoring trainer from {} ...\n".format(save_path))

        data = bst.graph.Dict(params=self.params, optimizer=self.optimizer)

        checkpoint = bst.graph.states(data).to_nest()
        braintools.file.msgpack_load(save_path, target=checkpoint)

    def saveplot(
        self,
        issave: bool = True,
        isplot: bool = True,
        loss_fname: str = "loss.dat",
        train_fname: str = "train.dat",
        test_fname: str = "test.dat",
        output_dir: str = None,
    ):
        """
        Saves and plots the loss and metrics.

        Args:
            issave: If ``True``, save the loss and metrics to files.
            isplot: If ``True``, plot the loss and metrics.
            loss_fname: Filename to save the loss.
            train_fname: Filename to save the training metrics.
            test_fname: Filename to save the test metrics.
            output_dir: Directory to save the files.
        """
        saveplot(
            self.loss_history,
            self.train_state,
            issave=issave,
            isplot=isplot,
            loss_fname=loss_fname,
            train_fname=train_fname,
            test_fname=test_fname,
            output_dir=output_dir,
        )


class TrainState(TrainStateBase):
    __module__ = 'deepxde.pinnx'

    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.Aux_train = dict()
        self.X_test = None
        self.y_test = None
        self.Aux_test = dict()

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

    def set_data_train(self, X_train, y_train, *args):
        self.X_train = X_train
        self.y_train = y_train
        if len(args) > 0:
            assert len(args) == 1, "Auxiliary training data must be a single argument."
            assert isinstance(args[0], dict), "Auxiliary training data must be a dictionary."
            self.Aux_train = args[0]

    def set_data_test(self, X_test, y_test, *args):
        self.X_test = X_test
        self.y_test = y_test
        if len(args) > 0:
            assert len(args) == 1, "Auxiliary test data must be a single argument."
            assert isinstance(args[0], dict), "Auxiliary test data must be a dictionary."
            self.Aux_test = args[0]

    def update_best(self):
        current_loss_train = jnp.sum(jnp.asarray(jax.tree.leaves(self.loss_train)))
        if self.best_loss_train > current_loss_train:
            self.best_step = self.step
            self.best_loss_train = current_loss_train
            self.best_loss_test = jnp.sum(jnp.asarray(jax.tree.leaves(self.loss_test)))
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test
