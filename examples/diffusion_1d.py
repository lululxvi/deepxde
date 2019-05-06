from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sciconet as scn


def get_diffusion_bcs(nx, nt):
    x = np.linspace(-1, 1, num=nx)[:, None]
    bcs = np.hstack((x, np.zeros((nx, 1))))

    t = np.linspace(0, 1, num=nt)[:, None]
    ics = np.vstack(
        (np.hstack((np.full((nt, 1), -1), t)), np.hstack((np.full((nt, 1), 1), t)))
    )
    return np.vstack((bcs, ics))


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_t = dy_x[:, 0], dy_x[:, 1]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0]
        return (
            dy_t
            - dy_xx
            + tf.exp(-x[:, 1])
            * (tf.sin(np.pi * x[:, 0]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0]))
        )

    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    geom = scn.geometry.Hypercube([-1, 0], [1, 1])
    anchors = get_diffusion_bcs(10, 10)
    data = scn.data.PDE(geom, pde, func, 0, anchors=anchors)

    layer_size = [2] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    batch_size = 36
    ntest = 10000
    model.compile(optimizer, lr, batch_size, ntest, metrics=["l2 relative error"])

    epochs = 10000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
