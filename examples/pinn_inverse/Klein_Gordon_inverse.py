"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Inverse problem for the Klein-Gordon equation: recover the mass parameter m^2
from sparse observations of the solution.

The Klein-Gordon equation is:
    u_tt - u_xx + m^2 u = 0

with exact solution u(x,t) = sin(pi*x) cos(omega*t), omega^2 = pi^2 + m^2.
The true mass parameter is m^2 = 4. We initialize at m^2 = 1 and recover it
from 50 randomly sampled observation points.
"""
import deepxde as dde
import numpy as np

# True mass parameter: m^2 = 4 (to be recovered)
m_sq_true = 4.0
omega = np.sqrt(np.pi**2 + m_sq_true)

# Learnable parameter, initialized far from true value
m_sq = dde.Variable(1.0)


def pde(x, y):
    """Klein-Gordon residual: u_tt - u_xx + m^2 u = 0."""
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - dy_xx + m_sq * y


def func(x):
    """Exact solution: u(x,t) = sin(pi*x) cos(omega*t)."""
    return np.sin(np.pi * x[:, 0:1]) * np.cos(omega * x[:, 1:2])


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
# Velocity IC: du/dt(x,0) = 0
ic_2 = dde.icbc.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda _, on_initial: on_initial,
)

# Sparse observation data at 50 random interior (x, t) pairs
rng = np.random.default_rng(42)
observe_x = np.column_stack(
    [rng.uniform(-1, 1, 50), rng.uniform(0, 1, 50)]  # columns are (x, t)
)
observe_y = func(observe_x)
# component=0 means the first (and only) network output must match observations
ptset = dde.icbc.PointSetBC(observe_x, observe_y, component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic_1, ic_2, ptset],
    num_domain=2000,
    num_boundary=200,
    num_initial=200,
    anchors=observe_x,
    solution=func,
    num_test=5000,
)

net = dde.nn.FNN([2] + [40] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Phase 1: Adam optimization
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    external_trainable_variables=m_sq,
)
# VariableValue callback logs m_sq every 1000 iterations to track convergence
variable = dde.callbacks.VariableValue(m_sq, period=1000, filename="variables.dat")
losshistory, train_state = model.train(iterations=30000, callbacks=[variable])

# Phase 2: L-BFGS refinement
model.compile(
    "L-BFGS",
    metrics=["l2 relative error"],
    external_trainable_variables=m_sq,
)
losshistory, train_state = model.train(callbacks=[variable])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
