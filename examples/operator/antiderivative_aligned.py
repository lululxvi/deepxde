"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train, y_train = (d["X"][0], d["X"][1]), d["y"]
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test, y_test = (d["X"][0], d["X"][1]), d["y"]

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# Choose a network
m = 100
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()
