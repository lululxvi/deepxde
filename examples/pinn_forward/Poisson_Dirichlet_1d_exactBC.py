"""Backend supported: tensorflow.compat.v1, tensorflow, paddle"""
import deepxde as dde
import numpy as np

geom = dde.geometry.Interval(0, np.pi)


# Define sine function
if dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    from deepxde.backend import tf

    sin = tf.sin
elif dde.backend.backend_name == "paddle":
    import paddle

    sin = paddle.sin


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    summation = sum([i * sin(i * x) for i in range(1, 5)])
    return -dy_xx - summation - 8 * sin(8 * x)

def func(x):
    summation = sum([np.sin(i * x) / i for i in range(1, 5)])
    return x + summation + np.sin(8 * x) / 8

data = dde.data.PDE(geom, pde, [], num_domain=64, solution=func, num_test=400)

layer_size = [1] + [50] * 3 + [1]
activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

def output_transform(x, y):
    return x * (np.pi - x) * y + x 

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4, decay=("inverse time", 1000, 0.3), metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=30000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
