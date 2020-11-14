from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

    def boundary(x, on_boundary):
        return on_boundary

    def func(x):
        return np.sin(np.pi * x)

    geom = dde.geometry.Interval(-1, 1)
    bc = dde.DirichletBC(geom, func, boundary)
    data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    checkpointer = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", verbose=1, save_better_only=True
    )
    # ImageMagick (https://imagemagick.org/) is required to generate the movie.
    movie = dde.callbacks.MovieDumper(
        "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
    )
    losshistory, train_state = model.train(
        epochs=10000, callbacks=[checkpointer, movie]
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # Plot PDE residual
    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    x = geom.uniform_points(1000, True)
    y = model.predict(x, operator=pde)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("PDE residual")
    plt.show()


if __name__ == "__main__":
    main()
