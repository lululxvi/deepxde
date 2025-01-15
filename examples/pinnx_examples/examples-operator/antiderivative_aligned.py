import brainstate as bst
import matplotlib.pyplot as plt
import numpy as np

from deepxde import pinnx

# Load dataset
d = np.load("../dataset/antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
d = np.load("../dataset/antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)

# Choose a network
m = 100
dim_x = 1
net = pinnx.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
)

# problem
problem = pinnx.problem.TripleCartesianProd(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    approximator=net,
)

# Define a Trainer
model = pinnx.Trainer(problem)

# Compile and Train
model.compile(bst.optim.Adam(0.001), metrics=["mean l2 relative error"]).train(iterations=10000)

# Plot the loss trajectory
pinnx.utils.plot_loss_history(model.loss_history)
plt.show()
