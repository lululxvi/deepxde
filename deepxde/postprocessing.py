from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def saveplot(losshistory, train_state, issave=True, isplot=True):
    if isplot:
        plot_loss_history(losshistory)
        plot_best_state(train_state)
        plt.show()

    if issave:
        save_loss_history(losshistory, "loss.dat")
        save_best_state(train_state, "train.dat", "test.dat")


def plot_loss_history(losshistory):
    loss_train = np.sum(
        np.array(losshistory.loss_train) * losshistory.loss_weights, axis=1
    )
    loss_test = np.sum(
        np.array(losshistory.loss_test) * losshistory.loss_weights, axis=1
    )

    plt.figure()
    plt.semilogy(losshistory.steps, loss_train, label="Train loss")
    plt.semilogy(losshistory.steps, loss_test, label="Test loss")
    for i in range(len(losshistory.metrics_test[0])):
        plt.semilogy(
            losshistory.steps,
            np.array(losshistory.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()


def save_loss_history(losshistory, fname):
    loss = np.hstack(
        (
            np.array(losshistory.steps)[:, None],
            np.array(losshistory.loss_train),
            np.array(losshistory.loss_test),
            np.array(losshistory.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")


def plot_best_state(train_state):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = y_train.shape[1]

    # Regression plot
    plt.figure()
    for i in range(y_dim):
        plt.plot(X_train[:, 0], y_train[:, i], "ok", label="Train")
        plt.plot(X_test[:, 0], y_test[:, i], "-k", label="True")
        plt.plot(X_test[:, 0], best_y[:, i], "--r", label="Prediction")
        if best_ystd is not None:
            plt.plot(
                X_test[:, 0], best_y[:, i] + 2 * best_ystd[:, i], "-b", label="95% CI"
            )
            plt.plot(X_test[:, 0], best_y[:, i] - 2 * best_ystd[:, i], "-b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Residual plot
    plt.figure()
    residual = y_test[:, 0] - best_y[:, 0]
    plt.plot(best_y[:, 0], residual, "o", zorder=1)
    plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual = Observed - Predicted")
    plt.tight_layout()

    if best_ystd is not None:
        plt.figure()
        for i in range(y_dim):
            plt.plot(X_test[:, 0], best_ystd[:, i], "-b")
            plt.plot(
                X_train[:, 0],
                np.interp(X_train[:, 0], X_test[:, 0], best_ystd[:, i]),
                "ok",
            )
        plt.xlabel("x")
        plt.ylabel("std(y)")


def save_best_state(train_state, fname_train, fname_test):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
    train = np.hstack((X_train, y_train))
    np.savetxt(fname_train, train, header="x, y")

    test = np.hstack((X_test, y_test, best_y))
    if best_ystd is not None:
        test = np.hstack((test, best_ystd))
    np.savetxt(fname_test, test, header="x, y_true, y_pred, y_std")
