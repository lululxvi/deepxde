from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sciconet as scn


def main():
    fname_lo_train = "examples/mf_lo_train.dat"
    fname_hi_train = "examples/mf_hi_train.dat"
    fname_hi_test = "examples/mf_hi_test.dat"

    data = scn.data.MfDataSet(
        fname_lo_train=fname_lo_train,
        fname_hi_train=fname_hi_train,
        fname_hi_test=fname_hi_test,
        col_x=(0,),
        col_y=(1,),
    )

    x_dim, y_dim = 1, 1
    activation = "tanh"
    initializer = "Glorot uniform"
    regularization = ["l2", 0.01]
    net = scn.maps.MfNN(
        [x_dim] + [20] * 4 + [y_dim],
        [10] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
    )

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    model.compile(
        optimizer,
        lr,
        len(data.X_lo_train),
        len(data.X_hi_test),
        metrics=["l2 relative error"],
    )

    epochs = 80000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
