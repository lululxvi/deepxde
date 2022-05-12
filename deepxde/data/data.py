import abc


class Data(abc.ABC):
    """Data base class."""

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Return a list of losses, i.e., constraints."""
        raise NotImplementedError("Data.losses is not implemented.")

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Return a list of losses for training dataset, i.e., constraints."""
        return self.losses(targets, outputs, loss_fn, inputs, model, aux=aux)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Return a list of losses for test dataset, i.e., constraints."""
        return self.losses(targets, outputs, loss_fn, inputs, model, aux=aux)

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

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y

    def test(self):
        return self.test_x, self.test_y
