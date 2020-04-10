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
