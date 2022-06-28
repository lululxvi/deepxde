import deepxde as dde
from deepxde.nn.pytorch.mionet import MIONetCartesianProd,PODMIONet
# from deepxde.nn.tensorflow_compat_v1.deeponet import DeepONet, DeepONetCartesianProd
from deepxde.data.quadruple import Quadruple, QuadrupleCartesianProd
from deepxde.data.triple import Triple, TripleCartesianProd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf
def pod(y):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    # w_cumsum = np.cumsum(w)
    # print(w_cumsum[:16] / w_cumsum[-1])
    # plt.figure()
    # plt.plot(y_mean)
    # plt.figure()
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.plot(v[:, i])
    # plt.show()
    v = np.ascontiguousarray(v)
    return y_mean, v

def network(problem, m):
    if problem == "ODE":
        branch = [m, 200, 200]
        trunk = [1, 200, 200]
    elif problem == "DR":
        branch = [m, 200, 200]
        trunk = [2, 200, 200]
    elif problem == "ADVD":
        branch = [m, 300, 300, 32]
        trunk = [2, 300, 300, 32]
    return branch, trunk


def run(problem, lr, epochs, m, activation, initializer):
    training_data = np.load(problem + '_train.npz', allow_pickle=True)
    testing_data = np.load(problem + '_test.npz', allow_pickle=True)
    X_train = training_data['X_train']
    y_train = training_data['y_train'].astype(np.float32)
    X_test = testing_data['X_test']
    y_test = testing_data['y_test'].astype(np.float32)
    X_train = (X_train[0].astype(np.float32), X_train[1].astype(np.float32), np.array(X_train[2]).astype(np.float32))
    X_test = (X_test[0].astype(np.float32), X_test[1].astype(np.float32), np.array(X_test[2]).astype(np.float32))

    branch_net, trunk_net = network(problem, m)
    y_mean, v = pod(y_test)
    data = QuadrupleCartesianProd(X_train, y_train, X_test, y_test)
    # net = MIONetCartesianProd(branch_net, branch_net, trunk_net,
    #                           {"branch1": activation[0],
    #                            "branch2": activation[1],
    #                            "trunk": activation[2]},
    #                           initializer, regularization=None)
    net = PODMIONet(v[:, :32],branch_net,branch_net,
                    {"branch1":activation[0],
                    "branch2":activation[1],
                    "trunk":activation[2]},
                    initializer, regularization=None)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    checker = dde.callbacks.ModelCheckpoint("model/mionet_model.ckpt", save_better_only=True, period=1000)
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    # print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))


def main():
    # Problems:
    # - "ODE": Antiderivative, Nonlinear ODE
    # - "DR": Diffusion-reaction
    # - "ADVD": Advection-diffusion
    problem = "ADVD"
    T = 1
    m = 100
    lr = 0.0002 if problem in ["ADVD"] else 0.001
    epochs = 100000
    activation = ["relu", None, "relu"] if problem in ["ADVD"] else ["relu", "relu", "relu"]
    initializer = "Glorot normal"

    run(problem, lr, epochs, m, activation, initializer)


if __name__ == "__main__":
    main()
