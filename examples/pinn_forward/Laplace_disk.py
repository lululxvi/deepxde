"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np

# Define function
if dde.backend.backend_name == "paddle":
    # Backend paddle
    import paddle

    sin = paddle.sin
    cos = paddle.cos
    concat = paddle.concat
elif dde.backend.backend_name == "pytorch":
    # Backend pytorch
    import torch

    sin = torch.sin
    cos = torch.cos
    concat = torch.cat
elif dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    # Backend tensorflow.compat.v1 or tensorflow
    from deepxde.backend import tf

    sin = tf.sin
    cos = tf.cos
    concat = tf.concat
elif dde.backend.backend_name == "jax":
    # Backend jax
    import jax.numpy as jnp

    sin = jnp.sin
    cos = jnp.cos
    concat = jnp.concatenate


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
# Backend tensorflow.compat.v1 or tensorflow or paddle or jax
def feature_transform(x):
    return concat([x[:, 0:1] * sin(x[:, 1:2]), x[:, 0:1] * cos(x[:, 1:2])], axis=1)


# Backend pytorch
# def feature_transform(x):
#     return cat(
#         [x[:, 0:1] * sin(x[:, 1:2]), x[:, 0:1] * cos(x[:, 1:2])], dim=1
#     )

net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=15000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
