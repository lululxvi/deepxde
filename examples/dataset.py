from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepxde as dde


def main():
    fname_train = "dataset/dataset.train"
    fname_test = "dataset/dataset.test"
    data = dde.data.DataSet(
        fname_train=fname_train,
        fname_test=fname_test,
        col_x=(0,),
        col_y=(1,),
        standardize=True,
    )

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=50000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
