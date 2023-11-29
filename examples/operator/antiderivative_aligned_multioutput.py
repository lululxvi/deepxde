"""Backend supported: tensorflow.compat.v1, tensorflow"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)

# Add random noise
noisy_y_train = y_train + 0.1 * np.random.randn(*y_train.shape)
y_train = np.concatenate(
    [y_train[:, :, np.newaxis], noisy_y_train[:, :, np.newaxis]], axis=-1
)
noisy_y_test = y_test + 0.1 * np.random.randn(*y_test.shape)
y_test = np.concatenate(
    [y_test[:, :, np.newaxis], noisy_y_test[:, :, np.newaxis]], axis=-1
)

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
    num_outputs=2,
    multi_output_strategy="independent",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()
