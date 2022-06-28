"""Backend supported: tensorflow.compat.v1, tensorflow"""
import deepxde as dde
import numpy as np


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return x * np.sin(5 * x)


geom = dde.geometry.Interval(-1, 1)
num_train = 10
num_test = 1000
data = dde.data.Function(geom, func, num_train, num_test)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
regularization = ["l2", 1e-5]
dropout_rate = 0.01
net = dde.nn.FNN(
    layer_size,
    activation,
    initializer,
    regularization=regularization,
    dropout_rate=dropout_rate,
)

model = dde.Model(data, net)
uncertainty = dde.callbacks.DropoutUncertainty(period=1000)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=30000, callbacks=[uncertainty])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
