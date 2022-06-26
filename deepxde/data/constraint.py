from .data import Data
from .. import config
from ..backend import tf


class Constraint(Data):
    """General constraints."""

    def __init__(self, constraint, train_x, test_x):
        self.constraint = constraint
        self.train_x = train_x
        self.test_x = test_x

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        f = tf.cond(
            model.net.training,
            lambda: self.constraint(inputs, outputs, self.train_x),
            lambda: self.constraint(inputs, outputs, self.test_x),
        )
        return loss_fn(tf.zeros(tf.shape(f), dtype=config.real(tf)), f)

    def train_next_batch(self, batch_size=None):
        return self.train_x, None

    def test(self):
        return self.test_x, None
