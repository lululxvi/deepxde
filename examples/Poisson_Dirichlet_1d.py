from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_xx = tf.gradients(dy_x, x)[0]
        return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

    def boundary(x, on_boundary):
        return on_boundary

    def func(x):
        return np.sin(np.pi * x)

    geom = scn.geometry.Interval(-1, 1)
    bc = scn.DirichletBC(geom, func, boundary)
    data = scn.data.PDE(geom, 1, pde, bc, 16, 2, func=func, num_test=100)

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(
        epochs=10000, model_save_path="./model/model.ckpt"
    )

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)

    # Plot PDE residue
    x = geom.uniform_points(1000, True)
    f = scn.callbacks.OperatorPredictor(x, pde)
    model.predict(x, callbacks=[f])
    plt.figure()
    plt.plot(x, f.get_value())
    plt.xlabel("x")
    plt.ylabel("PDE residue")
    plt.show()


if __name__ == "__main__":
    main()
