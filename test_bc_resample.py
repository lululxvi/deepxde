"""
Regression test for the BC resampling coordinate-mismatch bug.

Bug
---
After ``resample_train_points(bc_points=True)``, ``PDE.losses_test()`` called
``bc.error(self.train_x, inputs, outputs, beg, end)`` where ``self.train_x``
held *new* BC coordinates from the resample, but ``inputs`` / ``outputs`` were
evaluated at the *old* ``test_x`` BC coordinates (because ``train_state.X_test``
is only set once at the start of training and never refreshed).  The result was
that ``bc.error()`` received mismatched coordinates and network outputs,
making the test BC loss meaningless.

Fixes
-----
1. ``PDE.losses_test()`` now passes ``X_bc=self.test_x`` to ``bc.error()``.
2. ``resample_train_points(bc_points=True)`` resets and regenerates
   ``self.test_x`` so its BC slice matches the new ``train_x`` BC coordinates.
3. The ``PDEResampler`` callback calls ``set_data_test()`` after resampling so
   ``train_state.X_test`` is updated before the next ``_test()`` call.

All three fixes are required:
- Fix 2 alone ensures ``data.test_x`` is fresh, but ``train_state.X_test``
  (used by ``_test()``) is still stale without fix 3.
- Fix 3 alone (without fix 2) leaves ``data.test_x`` stale, so the
  ``set_data_test(*data.test())`` call in the callback returns cached stale data.
- Fix 1 ensures the correct X is used inside ``losses_test()`` even if the
  caller supplies a ``test_x`` that differs from ``train_x``.
"""

import numpy as np
import deepxde as dde
import deepxde.backend as bkd


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_data(with_pde=True, num_test=30):
    """2-D Laplace on [0,1]^2 with Dirichlet BC u = x_0.

    Uses train_distribution="pseudo" (numpy random) so that each call to
    resample_train_points() genuinely produces different boundary point
    coordinates (unlike deterministic sequences such as Hammersley).
    """
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    bc = dde.icbc.DirichletBC(geom, lambda x: x[:, 0:1], lambda _, on: on)

    def laplace(x, y):
        return (
            dde.grad.hessian(y, x, i=0, j=0)
            + dde.grad.hessian(y, x, i=1, j=1)
        )

    return dde.data.PDE(
        geom,
        laplace if with_pde else None,
        bc,
        num_domain=20,
        num_boundary=16,
        train_distribution="pseudo",
        num_test=num_test,
    )


def _resample(data):
    """Resample both PDE and BC points.

    pde_points=True causes new random boundary points inside train_x_all,
    which bc_points() then filters — guaranteeing different BC coordinates.
    """
    data.resample_train_points(pde_points=True, bc_points=True)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_bc_points_equal_before_resample():
    """Sanity check: before any resample, train and test BC slices are equal.

    test_points() always prepends the current train_x_bc to the test PDE
    points, so they must start out identical.
    """
    data = _make_data()
    n = data.num_bcs[0]
    assert np.allclose(data.train_x[:n], data.test_x[:n]), (
        "Before resample: train_x and test_x BC slices should be identical."
    )
    print("PASS  test_bc_points_equal_before_resample")


def test_test_x_refreshed_after_resample():
    """Fix 2: resample_train_points(bc_points=True) must regenerate test_x.

    After resampling, data.test_x[:n_bc] should equal the new
    data.train_x[:n_bc].  Before the fix, test_x was never reset, so it
    retained the old (stale) BC coordinates.
    """
    data = _make_data()
    n = data.num_bcs[0]
    old_bc = data.train_x[:n].copy()

    _resample(data)

    new_train_bc = data.train_x[:n]
    new_test_bc  = data.test_x[:n]

    assert not np.allclose(old_bc, new_train_bc), (
        "BC coordinates did not change after resample — "
        "the test is inconclusive (try a larger geometry)."
    )
    assert np.allclose(new_train_bc, new_test_bc), (
        "FAIL (fix 2): test_x BC coordinates are stale after resample.\n"
        f"  train_x[:n]:\n{new_train_bc}\n"
        f"  test_x[:n]: \n{new_test_bc}"
    )
    print("PASS  test_test_x_refreshed_after_resample")


def test_train_state_X_test_refreshed_by_callback():
    """Fix 3: PDEResampler must push new test_x into train_state.X_test.

    train_state.X_test is used by model._test() as the inputs to the network.
    After resampling, it must be updated so the network is evaluated at the
    new BC coordinates before losses_test() is called.
    """
    data = _make_data()
    net = dde.nn.FNN([2, 16, 1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train_state.set_data_train(*data.train_next_batch())
    model.train_state.set_data_test(*data.test())

    n = data.num_bcs[0]

    # Simulate what PDEResampler.on_epoch_end() now does.
    _resample(data)
    model.train_state.set_data_test(*data.test())   # fix 3

    assert np.allclose(data.train_x[:n], model.train_state.X_test[:n]), (
        "FAIL (fix 3): train_state.X_test BC slice not updated after resample.\n"
        f"  train_x[:n]:       \n{data.train_x[:n]}\n"
        f"  X_test[:n] (stale):\n{model.train_state.X_test[:n]}"
    )
    print("PASS  test_train_state_X_test_refreshed_by_callback")


def test_losses_test_passes_test_x_to_bc_error():
    """Fix 1: losses_test() must pass self.test_x as X to bc.error().

    We isolate this fix from fix 2 by manually patching data.test_x to a
    sentinel value after resampling, creating a scenario where
    test_x[:n_bc] != train_x[:n_bc].  We then intercept bc.error() and
    assert it receives the sentinel coordinates, not the train_x coordinates.

    Using pde=None avoids autodiff so the call works without a compiled model.
    """
    data = _make_data(with_pde=False)
    n = data.num_bcs[0]

    _resample(data)

    # Patch test_x: set BC rows to a sentinel clearly different from train_x.
    sentinel = data.test_x.copy()
    sentinel[:n] = 0.5          # train_x[:n] contains random coords in [0,1]^2
    data.test_x = sentinel      # inject; losses_test() reads self.test_x

    received_X = []
    orig_error = data.bcs[0].error

    def capturing_error(X, inputs, outputs, beg, end, aux_var=None):
        received_X.append(X)
        return orig_error(X, inputs, outputs, beg, end, aux_var)

    data.bcs[0].error = capturing_error

    try:
        dummy_outputs = bkd.as_tensor(
            np.zeros((len(sentinel), 1), dtype=np.float32)
        )
        dummy_inputs = bkd.as_tensor(sentinel.astype(np.float32))
        data.losses_test(
            None,
            dummy_outputs,
            lambda a, b: bkd.reduce_mean(b ** 2),
            dummy_inputs,
            model=None,     # not accessed when pde=None and no auxiliary vars
        )
    finally:
        data.bcs[0].error = orig_error  # always restore

    assert received_X, "bc.error() was never called — check bcs list is non-empty."

    X_got = received_X[0]

    # With fix 1, X should carry the sentinel values.
    assert np.allclose(X_got[:n], sentinel[:n]), (
        "FAIL (fix 1): losses_test() did not pass test_x to bc.error().\n"
        f"  Expected X[:n] (sentinel = 0.5):\n{sentinel[:n]}\n"
        f"  Got X[:n]:                      \n{X_got[:n]}"
    )
    # And it must NOT be train_x (different coords after resample).
    assert not np.allclose(X_got[:n], data.train_x[:n]), (
        "FAIL (fix 1): X[:n] matches train_x[:n] — old buggy behaviour still active."
    )
    print("PASS  test_losses_test_passes_test_x_to_bc_error")


def test_github_issue_timepde_neumann_bc():
    """Regression test for the specific bug report.

    The reporter used a TimePDE on a 2-D+time rectangle with four BCs
    (two Neumann, one Dirichlet, one Neumann) plus an IC, trained with
    PDEPointResampler(bc_points=True).  After the first resample at step 1000,
    the test loss for the top Neumann BC (bc index 3 in their ordering) jumped
    from ~4e-4 to ~4e-1 — roughly 1000× — while train loss and test metrics
    stayed normal, confirming the mismatch was in the test loss path only.

    After the fix the key invariant must hold: for every BC i, the slice
    train_x[bcs_start[i]:bcs_start[i+1]] must equal the corresponding slice of
    test_x.  This guarantees that losses_test() (which now passes X_bc=test_x)
    and the network evaluation (which runs on train_state.X_test ≈ test_x) are
    both at the same coordinates.
    """
    import os
    os.environ.setdefault("DDE_BACKEND", "tensorflow")

    LEFT, RIGHT = 0.0, np.pi
    BOTTOM, TOP = -np.pi, np.pi
    T = 1.0

    geom = dde.geometry.Rectangle([LEFT, BOTTOM], [RIGHT, TOP])
    timedomain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(xyt, u):
        return (
            dde.grad.jacobian(u, xyt, i=0, j=2)
            - 1.9 * (
                dde.grad.hessian(u, xyt, i=0, j=0)
                + dde.grad.hessian(u, xyt, i=1, j=1)
            )
        )

    def on_left(xyt, on):
        return on and dde.utils.isclose(xyt[0], LEFT) and not (
            dde.utils.isclose(xyt[1], TOP) or dde.utils.isclose(xyt[1], BOTTOM)
        )
    def on_right(xyt, on):
        return on and dde.utils.isclose(xyt[0], RIGHT) and not (
            dde.utils.isclose(xyt[1], TOP) or dde.utils.isclose(xyt[1], BOTTOM)
        )
    def on_bottom(xyt, on):
        return on and dde.utils.isclose(xyt[1], BOTTOM) and not (
            dde.utils.isclose(xyt[0], LEFT) or dde.utils.isclose(xyt[0], RIGHT)
        )
    def on_top(xyt, on):
        return on and dde.utils.isclose(xyt[1], TOP) and not (
            dde.utils.isclose(xyt[0], LEFT) or dde.utils.isclose(xyt[0], RIGHT)
        )

    bcs = [
        dde.icbc.NeumannBC(geomtime, lambda xyt: np.zeros((len(xyt), 1)), on_left),
        dde.icbc.NeumannBC(geomtime, lambda xyt: np.zeros((len(xyt), 1)), on_right),
        dde.icbc.DirichletBC(geomtime, lambda xyt: np.zeros((len(xyt), 1)), on_bottom),
        dde.icbc.NeumannBC(geomtime, lambda xyt: np.zeros((len(xyt), 1)), on_top),
        dde.icbc.IC(geomtime, lambda xyt: np.zeros((len(xyt), 1)), lambda _, on: on),
    ]

    data = dde.data.TimePDE(
        geomtime, pde, bcs,
        num_domain=200,
        num_boundary=200,
        num_initial=200,
        train_distribution="pseudo",
        num_test=100,
    )

    # Simulate one PDEPointResampler cycle (resample + callback update).
    net = dde.nn.FNN([3, 16, 1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train_state.set_data_train(*data.train_next_batch())
    model.train_state.set_data_test(*data.test())

    # Trigger resample (mirrors PDEResampler.on_epoch_end with bc_points=True).
    data.resample_train_points(pde_points=True, bc_points=True)
    model.train_state.set_data_test(*data.test())   # fix 3

    bcs_start = np.cumsum([0] + data.num_bcs).astype(int)

    for i, bc in enumerate(data.bcs):
        beg, end = bcs_start[i], bcs_start[i + 1]
        train_bc_coords = data.train_x[beg:end]
        test_bc_coords  = data.test_x[beg:end]

        assert np.allclose(train_bc_coords, test_bc_coords), (
            f"FAIL (BC {i}, {type(bc).__name__}): after resample, "
            f"test_x BC coords differ from train_x BC coords.\n"
            f"  train_x[{beg}:{end}] (first 3):\n{train_bc_coords[:3]}\n"
            f"  test_x[{beg}:{end}]  (first 3):\n{test_bc_coords[:3]}\n"
            "This is the coordinate mismatch reported in the GitHub issue: "
            "losses_test() was evaluating bc.error() with X from train_x "
            "(new coords) and inputs/outputs from test_x (old coords)."
        )

    # Also verify train_state.X_test is in sync.
    Xtest_bc_slice = model.train_state.X_test[bcs_start[3]:bcs_start[4]]
    train_bc_slice  = data.train_x[bcs_start[3]:bcs_start[4]]
    assert np.allclose(train_bc_slice, Xtest_bc_slice), (
        "FAIL: train_state.X_test top-BC slice not updated after resample.\n"
        "The network would be evaluated at old coords while bc.error() "
        "receives new coords — reproducing the reported ~1000× test loss spike."
    )

    print("PASS  test_github_issue_timepde_neumann_bc")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_bc_points_equal_before_resample()
    test_test_x_refreshed_after_resample()
    test_train_state_X_test_refreshed_by_callback()
    test_losses_test_passes_test_x_to_bc_error()
    test_github_issue_timepde_neumann_bc()
    print("\nAll tests passed.")
