"""Backend supported: tensorflow.compat.v1, paddle"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1
from deepxde.backend import tf
# Import paddle if using backend paddle
# import paddle
from scipy.special import gamma


alpha = 1.5

# Backend tensorflow.compat.v1
def fpde(x, y, int_mat):
    """(D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)"""
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        int_mat = tf.SparseTensor(*int_mat)
        lhs = tf.sparse_tensor_dense_matmul(int_mat, y)
    else:
        lhs = tf.matmul(int_mat, y)
    rhs = (
        gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
        - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
        + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
        - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    )
    # lhs /= 2 * np.cos(alpha * np.pi / 2)
    # rhs = gamma(alpha + 2) * x
    return lhs - rhs[: tf.size(lhs)]
# Backend paddle
# def fpde(x, y, int_mat):
#     """(D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)"""
#     if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
#         indices, values, shape = int_mat
#         int_mat = paddle.sparse.sparse_coo_tensor(list(zip(*indices)), values, shape, stop_gradient=False)
#         lhs = paddle.sparse.matmul(int_mat, y)
#     else:
#         lhs = paddle.mm(int_mat, y)
#     rhs = (
#         gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
#         - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
#         + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
#         - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
#     )
#     # lhs /= 2 * np.cos(alpha * np.pi / 2)
#     # rhs = gamma(alpha + 2) * x
#     return lhs - rhs[: paddle.numel(lhs)]


def func(x):
    # return x * (np.abs(1 - x**2)) ** (alpha / 2)
    return x ** 3 * (1 - x) ** 3


geom = dde.geometry.Interval(0, 1)
bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

# Static auxiliary points
data = dde.data.FPDE(geom, fpde, alpha, bc, [101], meshtype="static", solution=func)
# Dynamic auxiliary points
# data = dde.data.FPDE(
#     geom, fpde, alpha, bc, [100], meshtype="dynamic", num_domain=20, num_boundary=2, solution=func, num_test=100
# )

net = dde.nn.FNN([1] + [20] * 4 + [1], "tanh", "Glorot normal")
net.apply_output_transform(lambda x, y: x * (1 - x) * y)

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
