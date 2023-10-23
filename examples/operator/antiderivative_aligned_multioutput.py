"""Supported backend: tensorflow.compat.v1, tensorflow"""
import os

os.environ["DDEBACKEND"] = "tensorflow"

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

"""
Loss function from: 
    DeepONet-Grid-UQ: A Trustworthy Deep Operator Framework
    for Predicting the Power Grid’s Post-Fault Trajectories. 
    Christian Moya, Shiqi Zhang, Meng Yue, and Guang Lin
"""


def log_likelihood(y_true, model_output):
    # y_pred = model_output[0]
    sigma_pred = model_output[:, :, 1]

    value = tf.math.reduce_mean(
        tf.math.log(2 * np.pi * tf.math.square(sigma_pred))
    )

    return value


def my_MSE(y_true, model_output):
    y_pred = model_output[:, :, 0]
    sigma_pred = model_output[:, :, 1]

    value = tf.math.reduce_mean(
        tf.math.square(y_pred - y_true) / (tf.math.square(sigma_pred))
    )

    return value


def L2metric(y_true, y_pred):
    return dde.metrics.mean_l2_relative_error(y_test, y_pred[:, :, 0])


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


def output_transform(inputs, outputs):
    y_pred = outputs[:, :, 0]
    sigma_pred = tf.math.exp(outputs[:, :, 1])
    return tf.stack([y_pred, sigma_pred], 2)


net.apply_output_transform(output_transform)

# Define a Model
model = dde.Model(data, net)

# Compile and Train

model.compile(
    "adam", lr=0.001, loss=[my_MSE, log_likelihood], metrics=[L2metric]
)
losshistory, train_state = model.train(iterations=10000)
print(model.predict(X_test).shape)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()
