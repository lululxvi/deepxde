import numpy as np

from .data import Data
from .pde import PDE
from .pde_operator import PDEOperator, PDEOperatorCartesianProd


class Union(Data):
    """Base class for combining multiple ``Data`` objects.

    Args:
        data_objects: A list of ``Data`` instances.
        loss: A shared loss function, or a list of per-data loss functions.
    """

    def __init__(self, data_objects, loss=None):
        self.data_objects = list(data_objects)
        self.loss = loss

        if isinstance(self.loss, (list, tuple)) and len(self.loss) != len(
            self.data_objects
        ):
            raise ValueError("loss list must match number of data objects")

    def _get_loss(self, i, default_loss):
        if self.loss is None:
            return default_loss
        if isinstance(self.loss, (list, tuple)):
            return self.loss[i]
        return self.loss


class UnionRoundRobin(Union):
    """Combines multiple ``Data`` objects by cycling through them in round-robin order."""

    def __init__(self, data_objects, loss=None):
        super().__init__(data_objects, loss)
        self._train_idx = 0
        self._test_idx = 0

    def train_next_batch(self, batch_size=None):
        return self.data_objects[self._train_idx].train_next_batch(batch_size)

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        i = self._train_idx
        self._train_idx = (self._train_idx + 1) % len(self.data_objects)
        return self.data_objects[i].losses_train(
            targets, outputs, self._get_loss(i, loss_fn), inputs, model, aux=aux
        )

    def test(self):
        return self.data_objects[self._test_idx].test()

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        i = self._test_idx
        self._test_idx = (self._test_idx + 1) % len(self.data_objects)
        return self.data_objects[i].losses_test(
            targets, outputs, self._get_loss(i, loss_fn), inputs, model, aux=aux
        )


class UnionMerge(Union):
    """Combines multiple ``Data`` objects into one merged forward pass per step.

    Incompatible with PDE-based ``Data`` objects (``PDE``, ``PDEOperator``, etc.);
    use ``UnionRoundRobin`` for those.
    """

    def __init__(self, data_objects, loss=None):
        _incompatible = (PDE, PDEOperator, PDEOperatorCartesianProd)
        for obj in data_objects:
            if isinstance(obj, _incompatible):
                raise TypeError(
                    f"{type(obj).__name__} is incompatible with UnionMerge. "
                    "Use UnionRoundRobin instead."
                )

        super().__init__(data_objects, loss)
        self._train_slices = None
        self._test_slices = None

    def train_next_batch(self, batch_size=None):
        batch_sizes = self._split_batch_size(batch_size)
        batches = [
            data.train_next_batch(bs)
            for data, bs in zip(self.data_objects, batch_sizes)
        ]
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
            targets,
            outputs,
            loss_fn,
            inputs,
            model,
            aux,
            self._train_slices,
            train=True,
        )

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self._test_slices is None:
            raise RuntimeError("test() must be called before losses_test().")
        return self._losses(
            targets,
            outputs,
            loss_fn,
            inputs,
            model,
            aux,
            self._test_slices,
            train=False,
        )

    def _losses(self, targets, outputs, loss_fn, inputs, model, aux, slices, train):
        losses = []
        for i, (data, sl) in enumerate(zip(self.data_objects, slices)):
            x_i = self._slice_inputs(inputs, sl)
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
        parts = list(zip(*batches))
        xs = parts[0]
        sizes = [UnionMerge._batch_size(x) for x in xs]
        slices, start = [], 0
        for n in sizes:
            slices.append(slice(start, start + n))
            start += n
        merged = [UnionMerge._concat_inputs(xs)]
        for group in parts[1:]:
            val = (
                None if all(g is None for g in group) else np.concatenate(group, axis=0)
            )
            merged.append(val)
        return (*merged, slices)

    @staticmethod
    def _batch_size(x):
        return x[0].shape[0] if isinstance(x, tuple) else x.shape[0]

    @staticmethod
    def _concat_inputs(xs):
        if isinstance(xs[0], tuple):
            return tuple(
                np.concatenate([x[i] for x in xs], axis=0) for i in range(len(xs[0]))
            )
        return np.concatenate(xs, axis=0)

    @staticmethod
    def _slice_inputs(x, sl):
        if isinstance(x, tuple):
            return tuple(xi[sl] for xi in x)
        return x[sl]
