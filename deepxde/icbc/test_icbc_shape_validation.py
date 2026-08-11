"""Shape-validation tests for deepxde.icbc.boundary_conditions.

Covers PR #2102 (standardized `_check_target_values` / `_check_func_output`
helpers) and extends coverage to every BC class that shares the same
silent-broadcast risk: DirichletBC, NeumannBC, RobinBC, OperatorBC,
PointSetBC (incl. multi-component), PointSetOperatorBC, and Interface2DBC.
"""
import numpy as np
import pytest

import deepxde as dde
from deepxde import backend as bkd
from deepxde import config
from deepxde.backend import backend_name
from deepxde.icbc.boundary_conditions import _check_func_output, _check_target_values


# ---------------------------------------------------------------------------
# Unit tests for the helpers themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected_shape, n_components, should_raise",
    [
        (1.0, (10, 1), 1, False),
        (np.ones((10, 1)), (10, 1), 1, False),
        (np.ones(10), (10, 1), 1, True),
        (np.ones((10, 2)), (10, 1), 1, True),
        (np.ones((8, 1)), (10, 1), 1, True),
    ],
)
def test_check_target_values_shape_contract(
    values, expected_shape, n_components, should_raise
):
    points = np.ones((10, 1))
    if should_raise:
        with pytest.raises(ValueError):
            _check_target_values(values, len(points), n_components, "TestBC")
    else:
        _check_target_values(values, len(points), n_components, "TestBC")


@pytest.mark.parametrize(
    "values, should_raise",
    [
        (1.0, False),
        (np.ones((10, 1)), False),
        (np.ones(10), True),
        (np.ones((10, 2)), True),
    ],
)
def test_check_func_output_runtime_contract(values, should_raise):
    if should_raise:
        with pytest.raises(RuntimeError):
            _check_func_output(values, "TestBC")
    else:
        _check_func_output(values, "TestBC")


# ---------------------------------------------------------------------------
# PointSetBC (construction-time, incl. multi-component)
# ---------------------------------------------------------------------------


def test_pointsetbc_valid_shape_accepts_column_target():
    points = np.ones((10, 1))
    values = np.ones((10, 1))
    bc = dde.icbc.PointSetBC(points, values, component=0)
    assert bc.values is not None


@pytest.mark.parametrize("bad_values", [np.ones(10), np.ones((10, 2)), np.ones((8, 1))])
def test_pointsetbc_invalid_shape_rejected(bad_values):
    points = np.ones((10, 1))
    with pytest.raises(ValueError):
        dde.icbc.PointSetBC(points, bad_values, component=0)


@pytest.mark.skipif(
    backend_name != "pytorch", reason="multi-component only implemented for pytorch"
)
def test_pointsetbc_multi_component_valid_shape():
    points = np.ones((10, 1))
    values = np.ones((10, 2))
    bc = dde.icbc.PointSetBC(points, values, component=[0, 1])
    assert bc.values is not None


@pytest.mark.skipif(
    backend_name != "pytorch", reason="multi-component only implemented for pytorch"
)
def test_pointsetbc_multi_component_invalid_shape_rejected():
    points = np.ones((10, 1))
    # Only one column of targets for two requested components.
    values = np.ones((10, 1))
    with pytest.raises(ValueError):
        dde.icbc.PointSetBC(points, values, component=[0, 1])


# ---------------------------------------------------------------------------
# PointSetOperatorBC (construction-time + runtime)
# ---------------------------------------------------------------------------


def test_pointsetoperatorbc_valid_constructor_shape_accepts_column_target():
    points = np.ones((10, 1))
    values = np.ones((10, 1))
    bc = dde.icbc.PointSetOperatorBC(points, values, lambda i, o, x: o)
    assert bc.values is not None


@pytest.mark.parametrize("bad_values", [np.ones(10), np.ones((10, 2)), np.ones((8, 1))])
def test_pointsetoperatorbc_invalid_constructor_shape_rejected(bad_values):
    points = np.ones((10, 1))
    with pytest.raises(ValueError):
        dde.icbc.PointSetOperatorBC(points, bad_values, lambda i, o, x: o)


def test_pointsetoperatorbc_error_rejects_1d_func_output():
    points = np.ones((10, 1), dtype=config.real(np))
    values = np.ones((10, 1), dtype=config.real(np))
    # outputs[:, 0] (integer index, not slice) drops a dimension -> shape (10,)
    bc = dde.icbc.PointSetOperatorBC(
        points, values, lambda inputs, outputs, x: outputs[:, 0]
    )

    outputs = bkd.as_tensor(np.ones((10, 1), dtype=config.real(np)))
    inputs = bkd.as_tensor(points)

    with pytest.raises(RuntimeError):
        bc.error(points, inputs, outputs, 0, 10)


def test_pointsetoperatorbc_error_accepts_column_func_output():
    points = np.ones((10, 1), dtype=config.real(np))
    values = np.ones((10, 1), dtype=config.real(np))
    bc = dde.icbc.PointSetOperatorBC(
        points, values, lambda inputs, outputs, x: outputs[:, 0:1]
    )

    outputs = bkd.as_tensor(np.ones((10, 1), dtype=config.real(np)))
    inputs = bkd.as_tensor(points)

    bc.error(points, inputs, outputs, 0, 10)  # should not raise


# ---------------------------------------------------------------------------
# DirichletBC / NeumannBC / RobinBC / OperatorBC (runtime, func-based)
# ---------------------------------------------------------------------------


def _linspace_inputs_outputs(n=5):
    X = np.linspace(0, 1, n).reshape(-1, 1).astype(config.real(np))
    inputs = bkd.as_tensor(X)
    outputs = bkd.as_tensor(np.zeros((n, 1), dtype=config.real(np)))
    return X, inputs, outputs


def test_dirichletbc_error_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    # func deliberately returns a 1D array — this is the exact bug pattern
    bc = dde.icbc.DirichletBC(geom, lambda x: np.ravel(x[:, 0]), lambda x, on: on)

    X, inputs, outputs = _linspace_inputs_outputs()
    with pytest.raises(RuntimeError):
        bc.error(X, inputs, outputs, 0, len(X))


def test_neumannbc_error_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.NeumannBC(geom, lambda x: np.ravel(x[:, 0]), lambda x, on: on)

    X, inputs, outputs = _linspace_inputs_outputs()
    with pytest.raises(RuntimeError):
        bc.error(X, inputs, outputs, 0, len(X))


def test_robinbc_error_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.RobinBC(
        geom, lambda x, y: np.ravel(x[:, 0]), lambda x, on: on
    )

    X, inputs, outputs = _linspace_inputs_outputs()
    with pytest.raises(RuntimeError):
        bc.error(X, inputs, outputs, 0, len(X))


def test_operatorbc_error_rejects_1d_func_output():
    geom = dde.geometry.Interval(0, 1)
    # outputs[:, 0] drops a dimension -> shape (N,), the same bug pattern
    bc = dde.icbc.OperatorBC(
        geom, lambda inputs, outputs, x: outputs[:, 0], lambda x, on: on
    )

    X, inputs, outputs = _linspace_inputs_outputs()
    with pytest.raises(RuntimeError):
        bc.error(X, inputs, outputs, 0, len(X))


def test_operatorbc_error_accepts_column_func_output():
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.OperatorBC(
        geom, lambda inputs, outputs, x: outputs[:, 0:1], lambda x, on: on
    )

    X, inputs, outputs = _linspace_inputs_outputs()
    bc.error(X, inputs, outputs, 0, len(X))  # should not raise


# ---------------------------------------------------------------------------
# Interface2DBC (runtime) — the case originally flagged as "out of scope"
# ---------------------------------------------------------------------------


def test_interface2dbc_error_rejects_1d_func_output():
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def on_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0)

    def on_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1)

    # func deliberately returns a 1D array for the jump condition
    bc = dde.icbc.Interface2DBC(
        geom, lambda x: np.ravel(x[:, 0]), on_left, on_right, direction="normal"
    )

    n = 4
    y = np.linspace(0, 1, n).reshape(-1, 1)
    left = np.hstack([np.zeros((n, 1)), y]).astype(config.real(np))
    right = np.hstack([np.ones((n, 1)), y]).astype(config.real(np))
    X = np.vstack([left, right])

    inputs = bkd.as_tensor(X)
    # Network must output exactly 2 components for Interface2DBC.
    outputs = bkd.as_tensor(np.ones((2 * n, 2), dtype=config.real(np)))

    with pytest.raises(RuntimeError):
        bc.error(X, inputs, outputs, 0, len(X))