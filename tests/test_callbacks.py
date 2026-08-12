"""Tests for deepxde.callbacks.TrainingMonitor.

These tests use a lightweight fake model instead of a full PINN training
run, so they stay fast and backend-agnostic, and force the "Agg" (headless)
Matplotlib backend so they can run in CI without a display.
"""
import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

import deepxde as dde
from deepxde.callbacks import TrainingMonitor


class _FakeTrainState:
    def __init__(self):
        self.iteration = 0


class _FakeLossHistory:
    def __init__(self):
        self.steps = [0]
        self.loss_train = [np.array([1.0])]
        self.loss_test = [np.array([1.0])]


class _FakeModel:
    """Minimal stand-in for ``dde.Model`` exposing what the callback needs."""

    def __init__(self):
        self.train_state = _FakeTrainState()
        self.losshistory = _FakeLossHistory()

    def predict(self, x):
        return np.zeros((len(x), 1))


def _make_monitor(period=3, show_loss=True):
    x_plot = np.linspace(0, 1, 10)[:, None]
    return TrainingMonitor(
        period=period, x_plot=x_plot, y_reference=lambda x: x, show_loss=show_loss
    )


def test_training_monitor_requires_x_plot():
    try:
        TrainingMonitor()
    except ValueError:
        pass
    else:
        raise AssertionError("TrainingMonitor() without x_plot should raise ValueError")


def test_training_monitor_disables_gracefully_without_display():
    # With the "Agg" backend there is no interactive display, so the
    # callback must disable itself instead of raising.
    monitor = _make_monitor(period=2)
    model = _FakeModel()
    monitor.set_model(model)

    monitor.on_train_begin()
    assert monitor.enabled is False

    for _ in range(10):
        model.train_state.iteration += 1
        monitor.on_epoch_end()
    monitor.on_train_end()


def test_training_monitor_triggers_at_period():
    # Force the "interactive" code path so the periodic-plotting logic and
    # the actual Matplotlib drawing calls are exercised, even though the
    # test still runs under the non-interactive "Agg" backend.
    period = 3
    monitor = _make_monitor(period=period)
    monitor._has_display = lambda matplotlib_module: True

    model = _FakeModel()
    monitor.set_model(model)
    monitor.on_train_begin()
    assert monitor.enabled is True

    calls = []
    original_update_plot = monitor._update_plot

    def spy():
        calls.append(model.train_state.iteration)
        original_update_plot()

    monitor._update_plot = spy

    num_epochs = 10
    for _ in range(num_epochs):
        model.train_state.iteration += 1
        monitor.on_epoch_end()

    assert calls == [period, 2 * period, 3 * period]
    # No exception should have disabled the callback along the way.
    assert monitor.enabled is True


def test_training_monitor_accessible_from_dde_callbacks():
    assert dde.callbacks.TrainingMonitor is TrainingMonitor
