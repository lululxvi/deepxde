"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

EI = 1.0
GA_s = 5.0
q = -1.0

def pde(x, y):
    # y: [w, phi]
    dw_x = dde.grad.jacobian(y, x, i=0, j=0)
    dphi_x = dde.grad.jacobian(y, x, i=1, j=0)

    dw_xx = dde.grad.hessian(y, x, component=0)
    dphi_xx = dde.grad.hessian(y, x, component=1)

    res_w = GA_s * (dw_xx - dphi_x) + q
    res_phi = EI * dphi_xx + GA_s * (dw_x - y[:, 1:2])

    return [res_w, res_phi]

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

# Boundary conditions: Free end M=0 and V=0
def bc_moment_free(x, y, _):
    return dde.grad.jacobian(y, x, i=1, j=0)

def bc_shear_free(x, y, _):
    dw_x = dde.grad.jacobian(y, x, i=0, j=0)
    return dw_x - y[:, 1:2]

# Analytical solution
def func(x):
    w_b = -(x**4) / 24.0 + (x**3) / 6.0 - (x**2) / 4.0
    w_s = (1.0 / GA_s) * ((x**2) / 2.0 - x)
    w_total = w_b + w_s
    phi = -(x**3) / 6.0 + (x**2) / 2.0 - x / 2.0
    return np.hstack((w_total, phi))


geom = dde.geometry.Interval(0, 1)

bc_w0 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l, component=0)
bc_phi0 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l, component=1)
bc_ML = dde.icbc.OperatorBC(geom, bc_moment_free, boundary_r)
bc_VL = dde.icbc.OperatorBC(geom, bc_shear_free, boundary_r)

data = dde.data.PDE(
    geom,
    pde,
    [bc_w0, bc_phi0, bc_ML, bc_VL],
    num_domain=60,
    num_boundary=2,
    solution=func,
    num_test=100,
)

net = dde.nn.FNN([1] + [30] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Weights ranking: [res_w, res_phi, bc_w0, bc_phi0, bc_ML, bc_VL]
loss_weights = [1.0, 1.0, 10.0, 10.0, 5.0, 5.0]

# 1. stage: Adam convergence
#model.compile("adam", lr=0.001, loss_weights=loss_weights ,metrics=["l2 relative error"])
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
model.train(iterations=6000)

# 2. stage: L-BFGS fine tuning
#model.compile("L-BFGS", loss_weights=loss_weights, metrics=["l2 relative error"])
model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state = model.train()

# Visualization
x_test = np.linspace(0, 1, 200).reshape(-1, 1)
y_pred = model.predict(x_test)
y_true = func(x_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(x_test, y_true[:, 0], "k--", label="Analytical (Exact)")
ax1.plot(x_test, y_pred[:, 0], "r-", label="PINN prediction")
ax1.set_title("Transverse displacement $w(x)$")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

ax2.plot(x_test, y_true[:, 1], "k--", label="Analytical (Exact)")
ax2.plot(x_test, y_pred[:, 1], "b-", label="PINN prediction")
ax2.set_title(r"Section rotation $\varphi(x)$")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()
plt.tight_layout()
plt.show()
