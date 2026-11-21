"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Euler-Bernoulli beam PDE
# ---------------------------------------------------------

def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + 1


# ---------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


# ---------------------------------------------------------
# Closed-form solution
# ---------------------------------------------------------

def func(x):
    return -(x**4) / 24 + x**3 / 6 - x**2 / 4


# ---------------------------------------------------------
# Synthetic FEM / experimental measurements
# ---------------------------------------------------------

np.random.seed(42)

num_measurements = 30

# Random measurement locations
X_measurement = np.random.uniform(
    0, 1, (num_measurements, 1)
)

# Exact displacement at measurement locations
Y_exact = func(X_measurement)

# Simulated numerical / measurement error
noise_level = 0.08  # 8% noise

noise = np.random.normal(
    loc=0.0,
    scale=noise_level,
    size=Y_exact.shape
)

# FEM-like noisy measurements
Y_measurement = Y_exact * (1.0 + noise)


# ---------------------------------------------------------
# PINN measurement data
# ---------------------------------------------------------

measurement_bc = dde.icbc.PointSetBC(
    X_measurement,
    Y_measurement,
    component=0,
)


# ---------------------------------------------------------
# Geometry and PINN boundary conditions
# ---------------------------------------------------------

geom = dde.geometry.Interval(0, 1)

bc1 = dde.icbc.DirichletBC(
    geom,
    lambda x: 0,
    boundary_l
)

bc2 = dde.icbc.NeumannBC(
    geom,
    lambda x: 0,
    boundary_l
)

bc3 = dde.icbc.OperatorBC(
    geom,
    lambda x, y, _: ddy(x, y),
    boundary_r
)

bc4 = dde.icbc.OperatorBC(
    geom,
    lambda x, y, _: dddy(x, y),
    boundary_r
)


# =========================================================
# PINN
# =========================================================

data = dde.data.PDE(
    geom,
    pde,
    [
        bc1,
        bc2,
        bc3,
        bc4,
        measurement_bc,
    ],
    num_domain=10,
    num_boundary=2,
    solution=func,
    num_test=100,
)


# ---------------------------------------------------------
# PINN neural network
# ---------------------------------------------------------

pinn_layer_size = [1] + [20] * 4 + [1]
mlp_layer_size = [1] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"

pinn_net = dde.nn.FNN(
    pinn_layer_size ,
    activation,
    initializer
)


# ---------------------------------------------------------
# PINN training
# ---------------------------------------------------------

pinn_model = dde.Model(
    data,
    pinn_net
)

pinn_model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"]
)

pinn_losshistory, pinn_train_state = pinn_model.train(
    iterations=10000
)


# =========================================================
# STANDARD MLP
# =========================================================

# Independent test set
X_test = np.linspace(
    0, 1, 200
).reshape(-1, 1)

Y_test = func(X_test)


# ---------------------------------------------------------
# MLP data
# ---------------------------------------------------------

mlp_data = dde.data.DataSet(
    X_train=X_measurement,
    y_train=Y_measurement,
    X_test=X_test,
    y_test=Y_test,
)


# ---------------------------------------------------------
# MLP neural network
# ---------------------------------------------------------

mlp_net = dde.nn.FNN(
    mlp_layer_size,
    activation,
    initializer
)


# ---------------------------------------------------------
# MLP training
# ---------------------------------------------------------

mlp_model = dde.Model(
    mlp_data,
    mlp_net
)

mlp_model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"]
)

mlp_losshistory, mlp_train_state = mlp_model.train(
    iterations=50000
)


# =========================================================
# Evaluation
# =========================================================

# Predictions on independent test points
Y_pinn = pinn_model.predict(X_test)
Y_mlp = mlp_model.predict(X_test)


# Calculate relative L2 errors
pinn_error = np.linalg.norm(
    Y_pinn - Y_test
) / np.linalg.norm(Y_test)

mlp_error = np.linalg.norm(
    Y_mlp - Y_test
) / np.linalg.norm(Y_test)

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(f"PINN relative L2 error: {pinn_error:.6e}")
print(f"MLP  relative L2 error: {mlp_error:.6e}")

print("=" * 50)


# =========================================================
# Plot comparison
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    X_test,
    Y_test,
    label="Exact solution",
    linewidth=2
)

plt.plot(
    X_test,
    Y_pinn,
    label="PINN",
    linewidth=2
)

plt.plot(
    X_test,
    Y_mlp,
    label="MLP",
    linewidth=2
)

plt.scatter(
    X_measurement,
    Y_measurement,
    label="Noisy measurements",
    marker="o",
    zorder=5
)

plt.xlabel("x")
plt.ylabel("Displacement")
plt.title("PINN vs MLP with Noisy Measurements")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
