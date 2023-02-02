"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Import torch if using backend pytorch
# import torch
# Import paddle if using backend paddle
# import paddle


def pde(x, y):
    dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    return x[:, 0:1] * dy_r + x[:, 0:1] ** 2 * dy_rr + dy_thetatheta


def solution(x):
    r, theta = x[:, 0:1], x[:, 1:]
    return r * np.cos(theta)


geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
bc_rad = dde.icbc.DirichletBC(
    geom,
    lambda x: np.cos(x[:, 1:2]),
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
)
data = dde.data.PDE(
    geom, pde, bc_rad, num_domain=2540, num_boundary=80, solution=solution
)

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
# Backend tensorflow.compat.v1 or tensorflow
def feature_transform(x):
    return tf.concat(
        [x[:, 0:1] * tf.sin(x[:, 1:2]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
    )
# Backend pytorch
# def feature_transform(x):
#     return torch.cat(
#         [x[:, 0:1] * torch.sin(x[:, 1:2]), x[:, 0:1] * torch.cos(x[:, 1:2])], dim=1
#     )
# Backend paddle
# def feature_transform(x):
#     return paddle.cat(
#         [x[:, 0:1] * paddle.sin(x[:, 1:2]), x[:, 0:1] * paddle.cos(x[:, 1:2])], axis=1
#     )

net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=15000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
