"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde import backend as bkd
# Import tf if using backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import tf
# Import torch if using backend pytorch
# import torch
import random
# Import paddle i`f using backend paddle
import paddle
import os
# paddle.set_default_dtype("float64")
# dde.config.set_default_float('float64')

import argparse
import paddle
import random
paddle.seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--static', default=False, action="store_true")
parser.add_argument(
    '--prim', default=False, action="store_true")
args = parser.parse_args()

if args.static is True:
    print("============= 静态图静态图静态图静态图静态图 =============")
    paddle.enable_static()
    if args.prim:
        paddle.incubate.autograd.enable_prim()
        print("============= prim prim prim prim prim  =============")
else:
    print("============= 动态图动态图动态图动态图动态图 =============")


task_name = os.path.basename(__file__).split(".")[0]

# 创建任务日志文件夹
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - np.pi ** 2 * bkd.sin(np.pi * x)


def boundary(x, on_boundary):
    return on_boundary


def func(x):
    return np.sin(np.pi * x)

geom = dde.geometry.Interval(-1, 1)
bc = dde.icbc.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
# set the same initializer param #
w_array = []
for i in range(1, len(layer_size)):
    shape = (layer_size[i-1], layer_size[i])
    w = np.random.normal(size=shape).astype('float32')
    w_array.append(w)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer, task_name)

# new_save = False
# for name, param in net.named_parameters():
#     if os.path.exists(f"/home/wangruting/science/deepxde_wrt_44_orig/deepxde_wrt_44/Poisson_Dirichlet_1d/{name}.npy"):
#         continue
#     new_save = True
#     np.save(f"/workspace/hesensen/paddlescience_project/deepxde_wrt_new/Poisson_Dirichlet_1d/{name}.npy", param.numpy())
#     print(f"successfully save param {name} at [/workspace/hesensen/paddlescience_project/deepxde_wrt_new/Poisson_Dirichlet_1d/{name}.npy]")

# if new_save:
#     print("第一次保存模型完毕，自动退出，请再次运行")
#     exit(0)
# else:
#     print("所有模型参数均存在，开始训练...............")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000, display_every=1)
# Optional: Save the model during training.
# checkpointer = dde.callbacks.ModelCheckpoint(
#     "model/model", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = dde.callbacks.MovieDumper(
#     "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# losshistory, train_state = model.train(iterations=10000, callbacks=[checkpointer, movie])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Optional: Restore the saved model with the smallest training loss
# model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
# Plot PDE residual
# x = geom.uniform_points(1000, True)
# y_ = func(x)
# file_name_y_ = 'standard_y'
# with open(file_name_y_,'w') as f:
#     np.savetxt(f,y_,delimiter=",")

# y = model.predict(x, operator=pde)
# y = model.predict(x, operator=None)

# if backend_name == 'paddle':
#     file_namex = 'paddle_x'
#     file_namey = 'paddle_y'
# elif backend_name == 'pytorch':
#     file_namex = 'pytorch_x'
#     file_namey = 'pytorch_y'
# elif backend_name == 'tensorflow':
#     file_namex = 'tensorflow_x'
#     file_namey = 'tensorflow_y'


# with open(file_namex,'ab') as f:
#     np.savetxt(f,x,delimiter=",")
# with open(file_namey,'ab') as g:
#     np.savetxt(g,y,delimiter=",")

# plt.figure()
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("PDE residual")
# plt.show()
