"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
d = np.load("antiderivative_unaligned_train.npz", allow_pickle=True)
X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
d = np.load("antiderivative_unaligned_test.npz", allow_pickle=True)
X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Choose a network
m = 100
dim_x = 1
net = dde.nn.DeepONet(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=10000)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()
