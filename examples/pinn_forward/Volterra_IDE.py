"""Backend supported: tensorflow.compat.v1, paddle
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import deepxde.backend as bkd


def ide(x, y, int_mat):
    int_mat = bkd.as_tensor(int_mat)
    rhs = bkd.matmul(int_mat, y)
    lhs1 = bkd.gradients(y, x)[0]
    return (lhs1 + y)[: bkd.size(rhs)] - rhs


def kernel(x, s):
    return np.exp(s - x)


def func(x):
    return np.exp(-x) * np.cosh(x)


geom = dde.geometry.TimeDomain(0, 5)
ic = dde.icbc.IC(geom, func, lambda _, on_initial: on_initial)

quad_deg = 20
data = dde.data.IDE(
    geom,
    ide,
    ic,
    quad_deg,
    kernel=kernel,
    num_domain=10,
    num_boundary=2,
    train_distribution="uniform",
)
print("*********************")
layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("L-BFGS")
model.train()

# an temporary alternative with adam optimizer
# model.compile("adam", lr=1e-3)
# model.train(iterations=10000)

X = geom.uniform_points(100)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X, y_true, "-")
plt.plot(X, y_pred, "o")
plt.show()
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
