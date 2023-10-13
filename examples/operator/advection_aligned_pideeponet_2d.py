"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

if dde.backend.backend_name == "paddle":
    import paddle

    dim_x = 5
    sin = paddle.sin
    cos = paddle.cos
    concat = paddle.concat
elif dde.backend.backend_name == "pytorch":
    import torch

    dim_x = 5
    sin = torch.sin
    cos = torch.cos
    concat = torch.cat
else:
    from deepxde.backend import tf

    dim_x = 2
    sin = tf.sin
    cos = tf.cos
    concat = tf.concat


# PDE
def pde(x, y, v):
    dy_x = dde.grad.jacobian(y, x, j=0)
    dy_t = dde.grad.jacobian(y, x, j=1)
    return dy_t + dy_x


# The same problem as advection_aligned_pideeponet.py
# But consider time as the 2nd space coordinate
# to demonstrate the implementation of 2D problems
geom = dde.geometry.Rectangle([0, 0], [1, 1])


def func_ic(x, v):
    return v


def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)


ic = dde.icbc.DirichletBC(geom, func_ic, boundary)

pde = dde.data.PDE(geom, pde, ic, num_domain=200, num_boundary=200)

# Function space
func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=32
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [dim_x, 128, 128, 128],
    "tanh",
    "Glorot normal",
)


def periodic(x):
    x, t = x[:, :1], x[:, 1:]
    x = x * 2 * np.pi
    return concat([cos(x), sin(x), cos(2 * x), sin(2 * x), t], 1)


net.apply_feature_transform(periodic)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(iterations=30000)
dde.utils.plot_loss_history(losshistory)

x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
u_true = np.sin(2 * np.pi * (x - t[:, None]))
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = np.sin(2 * np.pi * eval_pts).T
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((v_branch, x_trunk))
u_pred = u_pred.reshape((100, 100))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
print(dde.metrics.l2_relative_error(u_true, u_pred))
