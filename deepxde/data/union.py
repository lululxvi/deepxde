import numpy as np

from .data import Data
from .mf import MfDataSet, MfFunc
from .pde_operator import PDEOperatorCartesianProd
from .triple import TripleCartesianProd
from .quadruple import Quadruple, QuadrupleCartesianProd
from .. import backend as bkd
from ..backend import backend_name

_UNSUPPORTED_DATA = (MfDataSet, MfFunc, PDEOperatorCartesianProd, TripleCartesianProd, Quadruple, QuadrupleCartesianProd)


class Union(Data):
    """Combines multiple Data objects for joint training.

    Use this class to train a single model on several independent datasets.
    The datasets remain separate: each contributes a portion of the training
    batch and produces its own loss value.

    Args:
        data_objects: A list of Data instances. Some classes (MfDataSet,
            MfFunc, PDEOperatorCartesianProd, TripleCartesianProd, Quadruple,
            QuadrupleCartesianProd) are not supported.
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

        The requested batch size is split as evenly as possible across datasets
        and passed to their train_next_batch() methods. The actual batch size
        returned by each dataset depends on its own batching semantics; some Data
        classes may ignore this argument.

        Training returns one loss value per dataset, so the loss history
        contains one value per dataset at each step.
    """

    def __init__(self, data_objects, loss=None):
        if backend_name != "pytorch":
            raise RuntimeError(
                f"Union is only supported on the pytorch backend "
                f"(current backend: {backend_name})."
            )
        for obj in data_objects:
            if isinstance(obj, _UNSUPPORTED_DATA):
                raise TypeError(
                    f"{type(obj).__name__} is incompatible with Union."
                )
        self.data_objects = list(data_objects)
        if len(self.data_objects) < 2:
            raise ValueError("Union requires at least 2 data objects.")
        self.loss = loss
        if isinstance(self.loss, (list, tuple)) and len(self.loss) != len(self.data_objects):
            raise ValueError("loss list must match number of data objects")
        self._train_slices = None
        self._test_slices = None
        self._train_batches = None
        self._test_batches = None

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
        self._train_batches = batches
        *merged, self._train_slices = self._merge_batches(batches)
        return tuple(merged)

    def test(self):
        batches = [data.test() for data in self.data_objects]
        self._test_batches = batches
        *merged, self._test_slices = self._merge_batches(batches)
        return tuple(merged)

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self._train_slices is None:
            raise RuntimeError(
                "train_next_batch() must be called before losses_train()."
            )
        return self._losses(loss_fn, model, train=True, batches=self._train_batches, aux=aux)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self._test_slices is None:
            raise RuntimeError("test() must be called before losses_test().")
        return self._losses(loss_fn, model, train=False, batches=self._test_batches, aux=aux)

    def _losses(self, loss_fn, model, train, batches, aux=None):
        losses = []
        for i, data in enumerate(self.data_objects):
            batch = batches[i]
            batch_x = batch[0]
            batch_y = batch[1] if len(batch) > 1 else None
            batch_aux = batch[2] if len(batch) > 2 else None
            if isinstance(batch_x, tuple):
                x_i = tuple(bkd.as_tensor(xi).requires_grad_() for xi in batch_x)
            else:
                x_i = bkd.as_tensor(batch_x).requires_grad_()
            y_i = bkd.as_tensor(batch_y) if batch_y is not None else None
            aux_i = bkd.as_tensor(batch_aux) if batch_aux is not None else None
            if hasattr(model.net, "auxiliary_vars"):
                model.net.auxiliary_vars = aux_i
            out_i = model.net(x_i)
            loss_fn_i = self._get_loss(i, loss_fn)
            if train:
                loss_i = data.losses_train(y_i, out_i, loss_fn_i, x_i, model, aux=aux)
            else:
                loss_i = data.losses_test(y_i, out_i, loss_fn_i, x_i, model, aux=aux)
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
        n = len(self.data_objects)
        if batch_size < n:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= number of data objects ({n})"
            )
        q, r = divmod(batch_size, n)
        return [q + int(i < r) for i in range(n)]

    @staticmethod
    def _merge_batches(batches):
        # transpose batches: [(x1, y1), (x2, y2)] -> [(x1, x2), (y1, y2)]
        n_parts = len(batches[0])
        if not all(len(b) == n_parts for b in batches):
            raise ValueError(
                "All data objects must return the same number of components from "
                f"train_next_batch()/test(). Got lengths: {[len(b) for b in batches]}"
            )
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
