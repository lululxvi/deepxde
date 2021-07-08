from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def saveplot(
    loss_history,
    train_state,
    issave=True,
    isplot=True,
    loss_fname="loss.dat",
    train_fname="train.dat",
    test_fname="test.dat",
    output_dir=None,
):
    """Save/plot the best trained result and loss history.

    This function is used to quickly check your results. To better investigate your
    result, use ``save_loss_history()`` and ``save_best_state()``.

    Args:
        output_dir (string): If ``None``, use the current working directory.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    if not os.path.exists(output_dir):
        print(f"Warning: Directory {output_dir} doesn't exist. Creating it.")
        os.mkdir(output_dir)

    if issave:
        loss_fname = os.path.join(output_dir, loss_fname)
        train_fname = os.path.join(output_dir, train_fname)
        test_fname = os.path.join(output_dir, test_fname)
        save_loss_history(loss_history, loss_fname)
        save_best_state(train_state, train_fname, test_fname)

    if isplot:
        plot_loss_history(loss_history)
        plot_best_state(train_state)
        plt.show()


def plot_loss_history(loss_history, fname=None):
    """Plot the training and testing loss history.

    Note:
        You need to call ``plt.show()`` to show the figure.

    Args:
        loss_history: ``LossHistory`` instance. The first variable returned from
            ``Model.train()``.
        fname (string): If `fname` is a string (e.g., 'loss_history.png'), then save the
            figure to the file of the file name `fname`.
    """
    loss_train = np.sum(loss_history.loss_train, axis=1)
    loss_test = np.sum(loss_history.loss_test, axis=1)

    plt.figure()
    plt.semilogy(loss_history.steps, loss_train, label="Train loss")
    plt.semilogy(loss_history.steps, loss_test, label="Test loss")
    for i in range(len(loss_history.metrics_test[0])):
        plt.semilogy(
            loss_history.steps,
            np.array(loss_history.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()

    if isinstance(fname, str):
        plt.savefig(fname)


def save_loss_history(loss_history, fname):
    """Save the training and testing loss history to a file."""
    print("Saving loss history to {} ...".format(fname))
    loss = np.hstack(
        (
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train),
            np.array(loss_history.loss_test),
            np.array(loss_history.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")


def plot_best_state(train_state):
    """Plot the best result of the smallest training loss.

    This function only works for 1D and 2D problems. For other problems and to better
    customize the figure, use ``save_best_state()``.

    Note:
        You need to call ``plt.show()`` to show the figure.

    Args:
        train_state: ``TrainState`` instance. The second variable returned from
            ``Model.train()``.
    """
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


def save_best_state(train_state, fname_train, fname_test):
    """Save the best result of the smallest training loss to a file."""
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
