"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Damped harmonic oscillator, solved with a PINN while using
``dde.callbacks.TrainingMonitor`` to watch the predicted solution and the
loss history update live, every few hundred epochs, instead of only seeing
a plot after training has finished.
"""
import deepxde as dde
import numpy as np

# Damped harmonic oscillator: y'' + 2*zeta*omega0*y' + omega0^2*y = 0
# with y(0) = 1, y'(0) = 0 (underdamped case, zeta < 1).
omega0 = 4
zeta = 0.3
omega_d = omega0 * np.sqrt(1 - zeta ** 2)


def ode(t, y):
    dy_dt = dde.grad.jacobian(y, t)
    d2y_dt2 = dde.grad.hessian(y, t)
    return d2y_dt2 + 2 * zeta * omega0 * dy_dt + omega0 ** 2 * y


def func(t):
    return np.exp(-zeta * omega0 * t) * (
        np.cos(omega_d * t) + (zeta * omega0 / omega_d) * np.sin(omega_d * t)
    )


geom = dde.geometry.TimeDomain(0, 5)


def boundary_l(t, on_initial):
    return on_initial and dde.utils.isclose(t[0], 0)


def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None)


ic1 = dde.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial)
ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

data = dde.data.TimePDE(geom, ode, [ic1, ic2], 32, 2, solution=func, num_test=500)
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# Points at which the live plot evaluates and shows the predicted solution.
x_plot = np.linspace(0, 5, 200)[:, None]
monitor = dde.callbacks.TrainingMonitor(
    period=200,
    x_plot=x_plot,
    y_reference=func,
    show_loss=True,
)

losshistory, train_state = model.train(iterations=10000, callbacks=[monitor])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
