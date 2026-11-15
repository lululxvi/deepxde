import numpy as np

import deepxde as dde


def _make_dataset(n_train=10):
    x = np.arange(n_train, dtype=float).reshape(-1, 1)
    return dde.data.DataSet(X_train=x, y_train=2 * x, X_test=x[:3], y_test=2 * x[:3])


def test_train_next_batch_without_batch_size_returns_all():
    data = _make_dataset()
    x, y = data.train_next_batch()
    assert x.shape == (10, 1)
    assert y.shape == (10, 1)


def test_train_next_batch_respects_batch_size():
    data = _make_dataset()
    x, y = data.train_next_batch(4)
    assert x.shape == (4, 1)
    assert y.shape == (4, 1)
    # Inputs and outputs of a batch must stay paired.
    np.testing.assert_array_equal(y, 2 * x)


def test_train_next_batch_covers_dataset_each_epoch():
    data = _make_dataset()
    seen = np.concatenate([data.train_next_batch(5)[0] for _ in range(2)])
    np.testing.assert_array_equal(np.sort(seen.ravel()), np.arange(10))
