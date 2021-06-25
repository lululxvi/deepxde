from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Data(object):
    """Data base class."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def losses(self, targets, outputs, loss, model):
        """Return a list of losses, i.e., constraints."""

    @abc.abstractmethod
    def train_next_batch(self, batch_size=None):
        """Return a training dataset of the size `batch_size`."""

    @abc.abstractmethod
    def test(self):
        """Return a test dataset."""


class Tuple(Data):
    """Dataset with each data point as a tuple.

    Each data tuple is split into two parts: input tuple (x) and output tuple (y).
    """

    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def losses(self, targets, outputs, loss, model):
        """Return a list of losses, i.e., constraints."""
        return [loss(targets, outputs)]

    def train_next_batch(self, batch_size=None):
        """Return a training dataset of the size `batch_size`."""
        return self.train_x, self.train_y

    def test(self):
        """Return a test dataset."""
        return self.test_x, self.test_y
