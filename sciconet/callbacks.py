from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Callback(object):
    """ Callback base class. """
    def __init__(self):
        pass

    def on_epoch_begin(self, training_state):
        """Called at the beginning of every epoch."""
        pass

    def on_epoch_end(self, training_state):
        """Called at the end of every epoch."""
        pass

    def on_batch_begin(self, training_state):
        """Called at the beginning of every batch."""
        pass

    def on_batch_end(self, training_state):
        """Called at the end of every batch."""
        pass

    def on_train_begin(self, training_state):
        """Called at the beginning of model training."""
        pass

    def on_train_end(self, training_state):
        """Called at the end of model training."""
        pass
