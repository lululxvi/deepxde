"""Backend supported: tensorflow.compat.v1, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
# Import tf if using backend tensorflow.compat.v1
from deepxde.backend import tf
# Import paddle if using backend paddle
# import paddle


# Backend tensorflow.compat.v1
def ide(x, y, int_mat):
    """int_0^x y(t)dt"""
    lhs1 = tf.matmul(int_mat, y)
    lhs2 = tf.gradients(y, x)[0]
    rhs = 2 * np.pi * tf.cos(2 * np.pi * x) + tf.sin(np.pi * x) ** 2 / np.pi
    return lhs1 + (lhs2 - rhs)[: tf.size(lhs1)]
# Backend paddle
# def ide(x, y, int_mat):
#     """int_0^x y(t)dt"""
#     lhs1 = paddle.mm(int_mat, y)
#     lhs2 = paddle.grad(y, x, create_graph=True)[0]
#     rhs = 2 * np.pi * paddle.cos(2 * np.pi * x) + paddle.sin(np.pi * x) ** 2 / np.pi
#     return lhs1 + (lhs2 - rhs)[: paddle.numel(lhs1)]


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return np.sin(2 * np.pi * x)


geom = dde.geometry.TimeDomain(0, 1)
ic = dde.icbc.IC(geom, func, lambda _, on_initial: on_initial)

quad_deg = 16
data = dde.data.IDE(geom, ide, ic, quad_deg, num_domain=16, num_boundary=2)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
model.train(iterations=10000)

X = geom.uniform_points(100, True)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X, y_true, "-")
plt.plot(X, y_pred, "o")
plt.show()
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
