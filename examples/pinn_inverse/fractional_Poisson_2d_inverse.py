"""Backend supported: tensorflow.compat.v1, paddle"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1
from deepxde.backend import tf
# Import paddle if using backend paddle
# import paddle
from scipy.special import gamma


alpha0 = 1.8
alpha = tf.Variable(1.5)


# Backend tensorflow.compat.v1
def fpde(x, y, int_mat):
    r"""\int_theta D_theta^alpha u(x)"""
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        int_mat = tf.SparseTensor(*int_mat)
        lhs = tf.sparse_tensor_dense_matmul(int_mat, y)
    else:
        lhs = tf.matmul(int_mat, y)
    lhs = lhs[:, 0]
    lhs *= -tf.exp(tf.lgamma((1 - alpha) / 2) + tf.lgamma((2 + alpha) / 2)) / (
        2 * np.pi ** 1.5
    )
    x = x[: tf.size(lhs)]
    rhs = (
        2 ** alpha0
        * gamma(2 + alpha0 / 2)
        * gamma(1 + alpha0 / 2)
        * (1 - (1 + alpha0 / 2) * tf.reduce_sum(x ** 2, axis=1))
    )
    return lhs - rhs
# Backend paddle
# def fpde(x, y, int_mat):
#     r"""\int_theta D_theta^alpha u(x)"""
#     if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
#         indices, values, shape = int_mat
#         int_mat = paddle.sparse.sparse_coo_tensor(list(zip(*indices)), values, shape, stop_gradient=False)
#         lhs = paddle.sparse.matmul(int_mat, y)
#     else:
#         lhs = paddle.mm(int_mat, y)
#     lhs = lhs[:, 0]
#     lhs *= -paddle.exp(paddle.lgamma((1 - alpha) / 2) + paddle.lgamma((2 + alpha) / 2)) / (
#         2 * np.pi ** 1.5
#     )
#     x = x[: paddle.size(lhs)]
#     rhs = (
#         2 ** alpha0
#         * gamma(2 + alpha0 / 2)
#         * gamma(1 + alpha0 / 2)
#         * (1 - (1 + alpha0 / 2) * paddle.sum(x ** 2, axis=1))
#     )
#     return lhs - rhs


def func(x):
    return (1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2) ** (1 + alpha0 / 2)


geom = dde.geometry.Disk([0, 0], 1)

observe_x = geom.random_points(30)
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x))

data = dde.data.FPDE(
    geom,
    fpde,
    alpha,
    observe_y,
    [8, 100],
    num_domain=64,
    anchors=observe_x,
    solution=func,
)

net = dde.nn.FNN([2] + [20] * 4 + [1], "tanh", "Glorot normal")
net.apply_output_transform(
    lambda x, y: (1 - tf.reduce_sum(x ** 2, axis=1, keepdims=True)) * y
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss_weights=[1, 100], external_trainable_variables=[alpha])
variable = dde.callbacks.VariableValue(alpha, period=1000)
losshistory, train_state = model.train(iterations=10000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
