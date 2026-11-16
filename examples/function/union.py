"""
Backend supported: pytorch

Demonstrates dde.data.Union: training one network on two separate Data objects.

Target function: f(x) = x * sin(pi * x)

  data_1 — clean observations on [-1, 0] via dde.data.Function
  data_2 — observations on [0, 1] via dde.data.DataSet
           with multiplicative frequency noise:
           y = x * sin(noise * pi * x), where noise ~ N(1, 0.1)

Union merges batches from both datasets into one forward pass
and collects each dataset's losses independently.
"""

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)


def func(x):
    return x * np.sin(np.pi * x)


# Data 1: clean observations on [-1, 0] via dde.data.Function
# 10 points without noise
geom_1 = dde.geometry.Interval(-1, 0)
data_1 = dde.data.Function(geom_1, func, num_train=10, num_test=10)

# Data 2: noisy observations on [0, 1] via dde.data.DataSet
# 100 points with noise
x2_train = rng.uniform(0, 1, (100, 1)).astype(np.float32)
noise = rng.normal(1, 0.1, x2_train.shape)
y2_train = (x2_train * np.sin(noise * np.pi * x2_train)).astype(np.float32)
x2_test = np.linspace(0, 1, 100)[:, None].astype(np.float32)
y2_test = func(x2_test).astype(np.float32)
data_2 = dde.data.DataSet(
    X_train=x2_train, y_train=y2_train, X_test=x2_test, y_test=y2_test
)

# Union merges data_1 and data_2
data = dde.data.Union([data_1, data_2])

net = dde.nn.FNN([1] + [32] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss_weights=[1, 1e-1])
model.train(iterations=5000)

x_plot = np.linspace(-1, 1, 500)[:, None]
y_pred = model.predict(x_plot)
y_true = func(x_plot)

plt.figure()
plt.plot(
    x_plot[:, 0],
    y_true[:, 0],
    label="ground true: x * sin(pi * x)",
    c="black",
    linewidth=2,
)
plt.plot(x_plot[:, 0], y_pred[:, 0], "--", label="prediction", c="red", linewidth=2)
plt.scatter(
    data_1.train_x[:, 0],
    data_1.train_y[:, 0],
    s=20,
    marker="x",
    c="blue",
    zorder=5,
    label="data_1",
)
plt.scatter(
    x2_train[:, 0],
    y2_train[:, 0],
    s=20,
    marker="x",
    c="green",
    zorder=5,
    label="data_2",
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
