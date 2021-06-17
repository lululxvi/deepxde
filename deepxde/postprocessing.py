from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=True,
    loss_fname="loss.dat",
    train_fname="train.dat",
    test_fname="test.dat",
    best_state_loss_plot_fname="loss.png",
    loss_hist_plot_fname="loss_hist.png",
    save_plot=False,
    show_plot=True,
    output_dir=os.getcwd(),
):

    if not os.path.exists(output_dir):
        print(f"Warning: Directory {output_dir} doesn't exist. Creating it.")
        os.mkdir(output_dir)

    loss_fpath = os.path.join(output_dir, loss_fname)
    train_fpath = os.path.join(output_dir, train_fname)
    test_fpath = os.path.join(output_dir, test_fname)
    best_state_plot_fpath = os.path.join(output_dir, best_state_loss_plot_fname)
    loss_hist_plot_fpath = os.path.join(output_dir, loss_hist_plot_fname)

    if issave:
        save_loss_history(losshistory, loss_fpath)
        save_best_state(train_state, train_fpath, test_fpath)

    if isplot:
        plot_loss_history(losshistory, fname=loss_hist_plot_fpath, save_plot=save_plot)
        plot_best_state(train_state, fname=best_state_plot_fpath, save_plot=save_plot)

        if show_plot:
            plt.show()


def plot_loss_history(losshistory, fname="loss_history.png", save_plot=False):
    loss_train = np.sum(losshistory.loss_train, axis=1)
    loss_test = np.sum(losshistory.loss_test, axis=1)

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

    if save_plot:
        plt.savefig(fname)


def save_loss_history(losshistory, fname):
    print("Saving loss history to {} ...".format(fname))
    loss = np.hstack(
        (
            np.array(losshistory.steps)[:, None],
            np.array(losshistory.loss_train),
            np.array(losshistory.loss_test),
            np.array(losshistory.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")


def plot_best_state(train_state, fname="best_state.png", save_plot=False):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = best_y.shape[1]

    # Regression plot
    if X_test.shape[1] == 1:
        idx = np.argsort(X_test[:, 0])
        X = X_test[idx, 0]
        plt.figure()
        for i in range(y_dim):
            if y_train is not None:
                plt.plot(X_train[:, 0], y_train[:, i], "ok", label="Train")
            if y_test is not None:
                plt.plot(X, y_test[idx, i], "-k", label="True")
            plt.plot(X, best_y[idx, i], "--r", label="Prediction")
            if best_ystd is not None:
                plt.plot(
                    X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% CI"
                )
                plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    elif X_test.shape[1] == 2:
        for i in range(y_dim):
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(X_test[:, 0], X_test[:, 1], best_y[:, i], ".")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$y_{}$".format(i + 1))

    # Residual plot
    if y_test is not None:
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

    if save_plot:
        plt.savefig(fname)


def save_best_state(train_state, fname_train, fname_test):
    print("Saving training data to {} ...".format(fname_train))
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
    if y_train is None:
        np.savetxt(fname_train, X_train, header="x")
    else:
        train = np.hstack((X_train, y_train))
        np.savetxt(fname_train, train, header="x, y")

    print("Saving test data to {} ...".format(fname_test))
    if y_test is None:
        test = np.hstack((X_test, best_y))
        if best_ystd is None:
            np.savetxt(fname_test, test, header="x, y_pred")
        else:
            test = np.hstack((test, best_ystd))
            np.savetxt(fname_test, test, header="x, y_pred, y_std")
    else:
        test = np.hstack((X_test, y_test, best_y))
        if best_ystd is None:
            np.savetxt(fname_test, test, header="x, y_true, y_pred")
        else:
            test = np.hstack((test, best_ystd))
            np.savetxt(fname_test, test, header="x, y_true, y_pred, y_std")
