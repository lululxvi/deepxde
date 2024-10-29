"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import deepxde as dde
import numpy as np


def gen_testdata():
    data = np.load("../dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Run Adam+L-BFGS
model.compile("adam", lr=1e-3)
model.train(iterations=15000)

model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Get test data
X, y_true = gen_testdata()

# Get the results after running Adam+L-BFGS
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual after Adam+L-BFGS:", np.mean(np.absolute(f)))
print(
    "L2 relative error after Adam+L-BFGS:",
    dde.metrics.l2_relative_error(y_true, y_pred),
)
np.savetxt("test_adam_lbfgs.dat", np.hstack((X, y_true, y_pred)))

# Run NNCG after Adam+L-BFGS
dde.optimizers.set_NNCG_options(rank=50, mu=1e-1)
model.compile("NNCG")
losshistory_nncg, train_state_nncg = model.train(iterations=1000, display_every=100)
dde.saveplot(losshistory_nncg, train_state_nncg, issave=True, isplot=True)

# Get the final results after running Adam+L-BFGS+NNCG
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual after Adam+L-BFGS+NNCG:", np.mean(np.absolute(f)))
print(
    "L2 relative error after Adam+L-BFGS+NNCG:",
    dde.metrics.l2_relative_error(y_true, y_pred),
)
np.savetxt("test_adam_lbfgs_nncg.dat", np.hstack((X, y_true, y_pred)))
