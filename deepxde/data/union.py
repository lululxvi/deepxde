import numpy as np

from .data import Data
from .mf import MfDataSet, MfFunc
from .pde import PDE
from .pde_operator import PDEOperator, PDEOperatorCartesianProd


class Union(Data):
    """Combines multiple Data objects for joint training.

    Use this class to train a single model on several independent datasets.
    The datasets remain separate: each contributes a portion of the training
    batch and produces its own loss value.

    Args:
        data_objects: A list of Data instances. PDE-based data classes (PDE,
            PDEOperator, PDEOperatorCartesianProd) are not supported.
        loss: Loss function to use. Pass a single callable to use the same
            loss for all datasets, or a list of callables corresponding to
            data_objects in the same order. If not specified, the losses
            provided to model.compile() are used.

    Example::

        data1 = dde.data.Triple(X_train=..., y_train=..., X_test=..., y_test=...)
        data2 = dde.data.Triple(X_train=..., y_train=..., X_test=..., y_test=...)
        data = dde.data.Union([data1, data2])
        model = dde.Model(data, net)
        model.compile("adam", lr=0.001)
        model.train(iterations=5000)

    Notes:
        All data objects must have the same input and output shapes, as their
        batches are concatenated along the sample axis before being passed to
        the model.

        Each training batch is split as evenly as possible across datasets.
        If the batch size is not evenly divisible by the number of datasets,
        earlier datasets in data_objects receive one extra sample.

        Training returns one loss value per dataset, so the loss history
        contains one value per dataset at each step.
    """

    def __init__(self, data_objects, loss=None):
        _incompatible = (MfDataSet, MfFunc, PDE, PDEOperator, PDEOperatorCartesianProd)
        for obj in data_objects:
            if isinstance(obj, _incompatible):
                raise TypeError(
                    f"{type(obj).__name__} is incompatible with Union."
                )

        self.data_objects = list(data_objects)
        self.loss = loss

        if isinstance(self.loss, (list, tuple)) and len(self.loss) != len(
            self.data_objects
        ):
            raise ValueError("loss list must match number of data objects")

        self._train_slices = None
        self._test_slices = None

    def _get_loss(self, i, default_loss):
        if self.loss is None:
            return default_loss
        if isinstance(self.loss, (list, tuple)):
            return self.loss[i]
        return self.loss

    def train_next_batch(self, batch_size=None):
        batch_sizes = self._split_batch_size(batch_size)
        batches = [
            data.train_next_batch(bs)
            for data, bs in zip(self.data_objects, batch_sizes)
        ]
        # slices are tied to this exact batch; losses_train() must be called before the next batch
        *merged, self._train_slices = self._merge_batches(batches)
        return tuple(merged)

    def test(self):
        batches = [data.test() for data in self.data_objects]
        *merged, self._test_slices = self._merge_batches(batches)
        return tuple(merged)

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self._train_slices is None:
            raise RuntimeError(
                "train_next_batch() must be called before losses_train()."
            )
        return self._losses(
            targets, outputs, loss_fn, inputs, model, aux, self._train_slices, train=True
        )

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self._test_slices is None:
            raise RuntimeError("test() must be called before losses_test().")
        return self._losses(
            targets, outputs, loss_fn, inputs, model, aux, self._test_slices, train=False
        )

    def _losses(self, targets, outputs, loss_fn, inputs, model, aux, slices, train):
        losses = []
        for i, (data, sl) in enumerate(zip(self.data_objects, slices)):
            x_i = tuple(xi[sl] for xi in inputs) if isinstance(inputs, tuple) else inputs[sl]
            y_i = None if targets is None else targets[sl]
            out_i = outputs[sl]
            loss_i_fn = self._get_loss(i, loss_fn)
            if train:
                loss_i = data.losses_train(y_i, out_i, loss_i_fn, x_i, model, aux=aux)
            else:
                loss_i = data.losses_test(y_i, out_i, loss_i_fn, x_i, model, aux=aux)
            if not isinstance(loss_i, list):
                loss_i = [loss_i]
            losses.extend(loss_i)
        return losses

    def _split_batch_size(self, batch_size):
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) != len(self.data_objects):
                raise ValueError("batch_size list must match number of data objects")
            return list(batch_size)
        if batch_size is None:
            return [None] * len(self.data_objects)
        q, r = divmod(batch_size, len(self.data_objects))
        return [q + int(i < r) for i in range(len(self.data_objects))]

    @staticmethod
    def _merge_batches(batches):
        # transpose batches: [(x1, y1), (x2, y2)] -> [(x1, x2), (y1, y2)]
        parts = list(zip(*batches))
        xs = parts[0]
        sizes = [x[0].shape[0] if isinstance(x, tuple) else x.shape[0] for x in xs]
        slices, start = [], 0
        for n in sizes:
            slices.append(slice(start, start + n))
            start += n
        if isinstance(xs[0], tuple):
            merged_x = tuple(
                np.concatenate([x[i] for x in xs], axis=0) for i in range(len(xs[0]))
            )
        else:
            merged_x = np.concatenate(xs, axis=0)
        merged = [merged_x]
        for group in parts[1:]:
            val = (
                None if all(g is None for g in group) else np.concatenate(group, axis=0)
            )
            merged.append(val)
        return (*merged, slices)
