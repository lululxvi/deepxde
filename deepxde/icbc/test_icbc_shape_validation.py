"""Shape-validation tests for deepxde.icbc.boundary_conditions.

Covers PR #2102: the shared `_check_target_values` / `_check_func_output`
helpers and their application across the BC classes exposed to the same
silent-broadcast risk.

IMPORTANT - every `pytest.raises` here uses `match=`. A bare
`pytest.raises(RuntimeError)` will happily swallow an unrelated
RuntimeError (e.g. an autodiff "tensor not in graph" error raised before
the shape check is ever reached) and report a green test over unpatched
code. Matching on the message text is what makes these assertions mean
what they claim.

Tiers:
  1. helper unit tests   - pure Python/NumPy, backend-independent
  2. constructor-time    - `_check_target_values` call sites
  3. runtime             - `_check_func_output` call sites via bc.error()

Run under each backend with DDE_BACKEND=<name>.
"""
import numpy as np
import pytest

import deepxde as dde
from deepxde import backend as bkd
from deepxde import config
from deepxde.backend import backend_name
from deepxde.icbc.boundary_conditions import _check_func_output, _check_target_values


# Message fragments the helpers are contracted to emit.
TARGET_MSG = r"must have shape"
FUNC_MSG = r"shape N by 1"

requires_pytorch = pytest.mark.skipif(
    backend_name != "pytorch",
    reason="multi-component BCs only implemented for pytorch upstream",
)


# ---------------------------------------------------------------------------
# Tier 1 - helper unit tests (backend-independent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, n_components, should_raise",
    [
        (1.0, 1, False),                 # scalar, intentional broadcast
        (0, 1, False),                   # int scalar
        (np.ones((10, 1)), 1, False),    # canonical column
        (np.ones(10), 1, True),          # 1-D: the (N,) -> (N,N) bug
        (np.ones((10, 2)), 1, True),     # too wide
        (np.ones((8, 1)), 1, True),      # row-count mismatch
        (np.ones((10, 2)), 2, False),    # valid multi-component
        (np.ones((10, 1)), 2, True),     # too narrow for 2 components
    ],
)
def test_check_target_values_shape_contract(values, n_components, should_raise):
    if should_raise:
        with pytest.raises(ValueError, match=TARGET_MSG):
            _check_target_values(values, 10, n_components, "TestBC")
    else:
        _check_target_values(values, 10, n_components, "TestBC")


def test_check_target_values_message_reports_both_shapes():
    with pytest.raises(ValueError) as exc:
        _check_target_values(np.ones(10), 10, 1, "TestBC")
    msg = str(exc.value)
    assert "TestBC" in msg
    assert "(10, 1)" in msg
    assert "(10,)" in msg


@pytest.mark.parametrize(
    "values, should_raise",
    [
        (1.0, False),                    # 0-d scalar, harmless broadcast
        (np.array(1.0), False),          # 0-d array
        (np.ones((10, 1)), False),       # canonical column
        (np.ones(10), True),             # 1-D
        (np.ones((10, 2)), True),        # wide 2-D
        (np.ones((2, 10, 1)), True),     # 3-D
    ],
)
def test_check_func_output_numpy_path(values, should_raise):
    """Exercises the NumPy fallback branch."""
    if should_raise:
        with pytest.raises(RuntimeError, match=FUNC_MSG):
            _check_func_output(values, "TestBC")
    else:
        _check_func_output(values, "TestBC")


@pytest.mark.parametrize(
    "array, should_raise",
    [
        (np.ones((10, 1)), False),
        (np.ones(10), True),
        (np.ones((10, 2)), True),
    ],
)
def test_check_func_output_backend_tensor_path(array, should_raise):
    """Exercises the bkd.ndim/bkd.shape branch with a real backend tensor.

    This is the test that catches a graph-mode backend where bkd.shape()
    returns a symbolic tensor and `shape[1] != 1` cannot be evaluated
    eagerly. If it fails on a backend, that is a finding about the helper,
    not about the test.
    """
    tensor = bkd.as_tensor(array.astype(config.real(np)))
    if should_raise:
        with pytest.raises(RuntimeError, match=FUNC_MSG):
            _check_func_output(tensor, "TestBC")
    else:
        _check_func_output(tensor, "TestBC")


# ---------------------------------------------------------------------------
# Tier 2 - constructor-time validation
# ---------------------------------------------------------------------------


def test_pointsetbc_accepts_column_target():
    bc = dde.icbc.PointSetBC(np.ones((10, 1)), np.ones((10, 1)), component=0)
    assert bc.values is not None


@pytest.mark.parametrize("bad_values", [np.ones(10), np.ones((10, 2)), np.ones((8, 1))])
def test_pointsetbc_rejects_bad_target(bad_values):
    with pytest.raises(ValueError, match=TARGET_MSG):
        dde.icbc.PointSetBC(np.ones((10, 1)), bad_values, component=0)


@requires_pytorch
def test_pointsetbc_multi_component_accepts_matching_width():
    bc = dde.icbc.PointSetBC(np.ones((10, 1)), np.ones((10, 2)), component=[0, 1])
    assert bc.values is not None


@requires_pytorch
def test_pointsetbc_multi_component_rejects_narrow_target():
    with pytest.raises(ValueError, match=TARGET_MSG):
        dde.icbc.PointSetBC(np.ones((10, 1)), np.ones((10, 1)), component=[0, 1])


def test_pointsetoperatorbc_accepts_column_target():
    bc = dde.icbc.PointSetOperatorBC(
        np.ones((10, 1)), np.ones((10, 1)), lambda i, o, x: o
    )
    assert bc.values is not None


@pytest.mark.parametrize("bad_values", [np.ones(10), np.ones((10, 2)), np.ones((8, 1))])
def test_pointsetoperatorbc_rejects_bad_target(bad_values):
    """np.ones(10) is the exact IndexError reported in #2102."""
    with pytest.raises(ValueError, match=TARGET_MSG):
        dde.icbc.PointSetOperatorBC(np.ones((10, 1)), bad_values, lambda i, o, x: o)


def test_pointsetoperatorbc_scalar_target_allowed():
    bc = dde.icbc.PointSetOperatorBC(np.ones((10, 1)), 1.0, lambda i, o, x: o)
    assert bc.values is not None


# ---------------------------------------------------------------------------
# Tier 3 - runtime validation via bc.error(...)
# ---------------------------------------------------------------------------
#
# NOTE ON THE FIXTURE: `outputs` here is a standalone constant tensor with
# no autodiff link to `inputs`. For BCs that compute a normal derivative
# (NeumannBC, RobinBC), this means grad.jacobian will fail if it is reached.
# That is deliberate: the patch evaluates and validates the user func
# *before* computing the gradient, so a correctly patched class raises the
# shape RuntimeError first. If you see a backend-specific autodiff error
# instead, the check is running too late (or not at all) in that class.


def _interval_fixture(n=5):
    X = np.linspace(0, 1, n).reshape(-1, 1).astype(config.real(np))
    inputs = bkd.as_tensor(X)
    outputs = bkd.as_tensor(np.zeros((n, 1), dtype=config.real(np)))
    return X, inputs, outputs


def test_dirichletbc_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.DirichletBC(geom, lambda x: np.ravel(x[:, 0]), lambda x, on: on)
    X, inputs, outputs = _interval_fixture()
    with pytest.raises(RuntimeError, match=FUNC_MSG):
        bc.error(X, inputs, outputs, 0, len(X))


def test_dirichletbc_accepts_column_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.DirichletBC(geom, lambda x: x[:, 0:1], lambda x, on: on)
    X, inputs, outputs = _interval_fixture()
    bc.error(X, inputs, outputs, 0, len(X))


def test_dirichletbc_accepts_scalar_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda x, on: on)
    X, inputs, outputs = _interval_fixture()
    bc.error(X, inputs, outputs, 0, len(X))


def test_neumannbc_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.NeumannBC(geom, lambda x: np.ravel(x[:, 0]), lambda x, on: on)
    X, inputs, outputs = _interval_fixture()
    with pytest.raises(RuntimeError, match=FUNC_MSG):
        bc.error(X, inputs, outputs, 0, len(X))


def test_robinbc_rejects_1d_func_output():
    """Requires RobinBC.error to evaluate+validate func before the jacobian."""
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.RobinBC(geom, lambda x, y: np.ravel(x[:, 0]), lambda x, on: on)
    X, inputs, outputs = _interval_fixture()
    with pytest.raises(RuntimeError, match=FUNC_MSG):
        bc.error(X, inputs, outputs, 0, len(X))


def test_operatorbc_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.OperatorBC(
        geom, lambda inputs, outputs, x: outputs[:, 0], lambda x, on: on
    )
    X, inputs, outputs = _interval_fixture()
    with pytest.raises(RuntimeError, match=FUNC_MSG):
        bc.error(X, inputs, outputs, 0, len(X))


def test_operatorbc_accepts_column_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.OperatorBC(
        geom, lambda inputs, outputs, x: outputs[:, 0:1], lambda x, on: on
    )
    X, inputs, outputs = _interval_fixture()
    bc.error(X, inputs, outputs, 0, len(X))


def test_pointsetoperatorbc_rejects_1d_func_output():
    """outputs[:, 0] uses an integer index, dropping a dim -> shape (N,)."""
    points = np.ones((10, 1), dtype=config.real(np))
    values = np.ones((10, 1), dtype=config.real(np))
    bc = dde.icbc.PointSetOperatorBC(
        points, values, lambda inputs, outputs, x: outputs[:, 0]
    )
    outputs = bkd.as_tensor(np.ones((10, 1), dtype=config.real(np)))
    inputs = bkd.as_tensor(points)
    with pytest.raises(RuntimeError, match=FUNC_MSG):
        bc.error(points, inputs, outputs, 0, 10)


def test_pointsetoperatorbc_accepts_column_func_output():
    points = np.ones((10, 1), dtype=config.real(np))
    values = np.ones((10, 1), dtype=config.real(np))
    bc = dde.icbc.PointSetOperatorBC(
        points, values, lambda inputs, outputs, x: outputs[:, 0:1]
    )
    outputs = bkd.as_tensor(np.ones((10, 1), dtype=config.real(np)))
    inputs = bkd.as_tensor(points)
    bc.error(points, inputs, outputs, 0, 10)


def test_interface2dbc_rejects_1d_func_output():
    """The case flagged 'out of scope' in the original PR description."""
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def on_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0)

    def on_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1)

    bc = dde.icbc.Interface2DBC(
        geom, lambda x: np.ravel(x[:, 0]), on_left, on_right, direction="normal"
    )

    n = 4
    y = np.linspace(0, 1, n).reshape(-1, 1)
    left = np.hstack([np.zeros((n, 1)), y]).astype(config.real(np))
    right = np.hstack([np.ones((n, 1)), y]).astype(config.real(np))
    X = np.vstack([left, right])
    inputs = bkd.as_tensor(X)
    # Interface2DBC expects a 2-component network output.
    outputs = bkd.as_tensor(np.ones((2 * n, 2), dtype=config.real(np)))

    with pytest.raises(RuntimeError, match=FUNC_MSG):
        bc.error(X, inputs, outputs, 0, len(X))