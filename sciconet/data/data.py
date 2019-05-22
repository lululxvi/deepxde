from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Data(object):
    """ Data base class. """

    def __init__(self):
        pass

    @abc.abstractmethod
    def losses(self, y_true, y_pred, loss, model):
        """Return a list of losses, i.e., constraints."""
        return None

    @abc.abstractmethod
    def train_next_batch(self, batch_size=None):
        return None

    @abc.abstractmethod
    def test(self):
        return None
