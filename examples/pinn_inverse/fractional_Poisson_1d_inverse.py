"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import numpy as np
import deepxde.backend as bkd
# from deepxde.backend import tf
from scipy.special import gamma
import paddle


alpha0 = 1.8
alpha = bkd.Variable(1.5)


def fpde(x, y, int_mat):
    """(D_{0+}^alpha + D_{1-}^alpha) u(x)"""
    int_mat_ = bkd.as_tensor(int_mat)
    if isinstance(int_mat, (list, tuple)) and len(int_mat_) == 3:
        int_mat_ = bkd.SparseTensor(*int_mat_)
        lhs = bkd.sparse_tensor_dense_matmul(int_mat_, y)
    else:
        lhs = bkd.matmul(int_mat_, y)
    lhs /= 2 * bkd.cos(alpha * np.pi / 2)
    rhs = gamma(alpha0 + 2) * x
    return lhs - rhs[: bkd.size(lhs)]


def func(x):
    return x * (np.abs(1 - x ** 2)) ** (alpha0 / 2)


geom = dde.geometry.Interval(-1, 1)

observe_x = np.linspace(-1, 1, num=20)[:, None]
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x))

# Static auxiliary points
# data = dde.data.FPDE(
#     geom,
#     fpde,
#     alpha,
#     observe_y,
#     [101],
#     meshtype="static",
#     anchors=observe_x,
#     solution=func,
# )
# Dynamic auxiliary points
data = dde.data.FPDE(
    geom,
    fpde,
    alpha,
    observe_y,
    [100],
    meshtype="dynamic",
    num_domain=20,
    anchors=observe_x,
    solution=func,
    num_test=100,
)

net = dde.nn.FNN([1] + [20] * 4 + [1], "tanh", "Glorot normal")
net.apply_output_transform(lambda x, y: (1 - x ** 2) * y)

model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss_weights=[1, 100])
variable = dde.callbacks.VariableValue(alpha, period=1000)
losshistory, train_state = model.train(iterations=10000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
