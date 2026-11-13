# # linux
# wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz
# wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz

# # windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/deeponet/antiderivative_unaligned_train.npz -o antiderivative_unaligned_train.npz
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/deeponet/antiderivative_unaligned_test.npz -o antiderivative_unaligned_test.npz


import brainstate as bst
import numpy as np

import deepxde.experimental as deepxde

# Load dataset
d = np.load("./antiderivative_unaligned_train.npz", allow_pickle=True)
X_train = (d["X_train0"].astype(np.float32), d["X_train1"].astype(np.float32))
y_train = d["y_train"].astype(np.float32)
d = np.load("./antiderivative_unaligned_test.npz", allow_pickle=True)
X_test = (d["X_test0"].astype(np.float32), d["X_test1"].astype(np.float32))
y_test = d["y_test"].astype(np.float32)

# Choose a network
m = 100
dim_x = 1
net = deepxde.nn.DeepONet(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
)

# problem
problem = deepxde.problem.TripleDataset(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    approximator=net,
)

# Define a Trainer
trainer = deepxde.Trainer(problem)
# Compile and Train
trainer.compile(bst.optim.Adam(0.001)).train(iterations=10000)
# Plot the loss trajectory
trainer.saveplot(issave=True, isplot=True)
