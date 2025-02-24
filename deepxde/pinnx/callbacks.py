# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

import sys

import brainstate as bst
import brainunit as u
import jax.tree
import numpy as np

from deepxde.callbacks import (
    Callback,
    CallbackList,
    ModelCheckpoint,
    Timer,
    MovieDumper,
    PDEPointResampler,
    EarlyStopping as EarlyStoppingCallback,
    DropoutUncertainty as DropoutUncertaintyCallback,
    VariableValue as VariableValueCallback,
    OperatorPredictor as OperatorPredictorCallback,
)

__all__ = [
    'Callback',
    'CallbackList',
    'ModelCheckpoint',
    'EarlyStopping',
    'Timer',
    'DropoutUncertainty',
    'VariableValue',
    'OperatorPredictor',
    'MovieDumper',
    'PDEPointResampler',
]


class EarlyStopping(EarlyStoppingCallback):
    """Stop training when a monitored quantity (training or testing loss) has stopped improving.
    Only checked at validation step according to ``display_every`` in ``Trainer.train``.

    Args:
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the trainer doesn't show improvement
            over the baseline.
        monitor: The loss function that is monitored. Either 'loss_train' or 'loss_test'
        start_from_epoch: Number of epochs to wait before starting
            to monitor improvement. This allows for a warm-up period in which
            no improvement is expected and thus training will not be stopped.
    """

    def get_monitor_value(self):
        if self.monitor == "loss_train":
            result = np.sum(jax.tree.leaves(self.model.train_state.loss_train))
        elif self.monitor == "loss_test":
            result = np.sum(jax.tree.leaves(self.model.train_state.loss_test))
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


class DropoutUncertainty(DropoutUncertaintyCallback):
    """Uncertainty estimation via MC dropout.

    References:
        `Y. Gal, & Z. Ghahramani. Dropout as a Bayesian approximation: Representing
        trainer uncertainty in deep learning. International Conference on Machine
        Learning, 2016 <https://arxiv.org/abs/1506.02142>`_.

    Warning:
        This cannot be used together with other techniques that have different behaviors
        during training and testing, such as batch normalization.
    """

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            y_preds = []
            for _ in range(1000):
                y_pred_test_one = self.model.fn_outputs(True, self.model.train_state.X_test)
                y_preds.append(y_pred_test_one)
            y_preds = jax.tree.map(lambda *x: u.math.stack(x, axis=0), *y_preds, is_leaf=u.math.is_quantity)
            self.model.train_state.y_std_test = jax.tree.map(lambda x: u.math.std(x, axis=0), y_preds,
                                                             is_leaf=u.math.is_quantity)


class VariableValue(VariableValueCallback):
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
        self.var_list = var_list if isinstance(var_list, (tuple, list)) else [var_list]
        for v in self.var_list:
            if not isinstance(v, bst.State):
                raise ValueError("The variable must be a brainstate.State object.")

        self.period = period
        self.precision = precision

        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0


class OperatorPredictor(OperatorPredictorCallback):
    """
    Generates operator values for the input samples.

    Args:
        x: The input data.
        op: The operator with inputs (x, y).
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
    """

    def on_predict_end(self):
        self.value = self._eval()
        # self.value = jax.tree.map(np.asarray, self._eval())

    @bst.compile.jit(static_argnums=0)
    def _eval(self):
        with bst.environ.context(fit=False):
            outputs = self.model.problem.approximator(self.x)
            return self.op(self.x, outputs)
