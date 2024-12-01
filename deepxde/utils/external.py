"""External utilities."""

import csv
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


class PointSet:
    """A set of points.

    Args:
        points: A NumPy array of shape (`N`, `dx`). A list of `dx`-dim points.
    """

    def __init__(self, points):
        self.points = np.array(points)

    def inside(self, x):
        """Returns ``True`` if `x` is in this set of points, otherwise, returns
        ``False``.

        Args:
            x: A NumPy array. A single point, or a list of points.

        Returns:
            If `x` is a single point, returns ``True`` or ``False``. If `x` is a list of
                points, returns a list of ``True`` or ``False``.
        """
        if x.ndim == 1:
            # A single point
            return np.any(np.all(isclose(x, self.points), axis=1))
        if x.ndim == 2:
            # A list of points
            return np.any(
                np.all(isclose(x[:, np.newaxis, :], self.points), axis=-1),
                axis=-1,
            )

    def values_to_func(self, values, default_value=0):
        """Convert the pairs of points and values to a callable function.

        Args:
            values: A NumPy array of shape (`N`, `dy`). `values[i]` is the `dy`-dim
                function value of the `i`-th point in this point set.
            default_value (float): The function value of the points not in this point
                set.

        Returns:
            A callable function. The input of this function should be a NumPy array of
                shape (?, `dx`).
        """

        def func(x):
            pt_equal = np.all(isclose(x[:, np.newaxis, :], self.points), axis=-1)
            not_inside = np.logical_not(np.any(pt_equal, axis=-1, keepdims=True))
            return np.matmul(pt_equal, values) + default_value * not_inside

        return func


def apply(func, args=None, kwds=None):
    """Launch a new process to call the function.

    This can be used to clear Tensorflow GPU memory after model execution:
    https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def standardize(X_train, X_test):
    """Standardize features by removing the mean and scaling to unit variance.

    The mean and std are computed from the training data `X_train` using
    `sklearn.preprocessing.StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_,
    and then applied to the testing data `X_test`.

    Args:
        X_train: A NumPy array of shape (n_samples, n_features). The data used to
            compute the mean and standard deviation used for later scaling along the
            features axis.
        X_test: A NumPy array.

    Returns:
        scaler: Instance of ``sklearn.preprocessing.StandardScaler``.
        X_train: Transformed training data.
        X_test: Transformed testing data.
    """
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return scaler, X_train, X_test


def uniformly_continuous_delta(X, Y, eps):
    """Compute the supremum of delta in uniformly continuous.

    Args:
        X: N x d, equispaced points.
    """
    if X.shape[1] == 1:
        # 1d equispaced points
        dx = np.linalg.norm(X[1] - X[0])
        n = len(Y)
        k = 1
        while True:
            if np.any(np.linalg.norm(Y[: n - k] - Y[k:], ord=np.inf, axis=1) >= eps):
                return (k - 0.5) * dx
            k += 1
    else:
        dX = scipy.spatial.distance.pdist(X, "euclidean")
        dY = scipy.spatial.distance.pdist(Y, "chebyshev")
        delta = np.min(dX)
        dx = delta / 2
        while True:
            if np.max(dY[dX <= delta]) >= eps:
                return delta - dx / 2
            delta += dx


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
    """Save/plot the loss history and best trained result.

    This function is used to quickly check your results. To better investigate your
    result, use ``save_loss_history()`` and ``save_best_state()``.

    Args:
        loss_history: ``LossHistory`` instance. The first variable returned from
            ``Model.train()``.
        train_state: ``TrainState`` instance. The second variable returned from
            ``Model.train()``.
        issave (bool): Set ``True`` (default) to save the loss, training points,
            and testing points.
        isplot (bool): Set ``True`` (default) to plot loss, metric, and the predicted
            solution.
        loss_fname (string): Name of the file to save the loss in.
        train_fname (string): Name of the file to save the training points in.
        test_fname (string): Name of the file to save the testing points in.
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
    # np.sum(loss_history.loss_train, axis=1) is error-prone for arrays of varying lengths.
    # Handle irregular array sizes.
    loss_train = np.array([np.sum(loss) for loss in loss_history.loss_train])
    loss_test = np.array([np.sum(loss) for loss in loss_history.loss_test])

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


def _pack_data(train_state):
    def merge_values(values):
        if values is None:
            return None
        return np.hstack(values) if isinstance(values, (list, tuple)) else values

    y_train = merge_values(train_state.y_train)
    y_test = merge_values(train_state.y_test)
    best_y = merge_values(train_state.best_y)
    best_ystd = merge_values(train_state.best_ystd)
    return y_train, y_test, best_y, best_ystd


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
    if isinstance(train_state.X_train, (list, tuple)):
        print(
            "Error: The network has multiple inputs, and plotting such result han't been implemented."
        )
        return

    y_train, y_test, best_y, best_ystd = _pack_data(train_state)
    y_dim = best_y.shape[1]

    # Regression plot
    # 1D
    if train_state.X_test.shape[1] == 1:
        idx = np.argsort(train_state.X_test[:, 0])
        X = train_state.X_test[idx, 0]
        plt.figure()
        for i in range(y_dim):
            if y_train is not None:
                plt.plot(train_state.X_train[:, 0], y_train[:, i], "ok", label="Train")
            if y_test is not None:
                plt.plot(X, y_test[idx, i], "-k", label="True")
            plt.plot(X, best_y[idx, i], "--r", label="Prediction")
            if best_ystd is not None:
                plt.plot(
                    X, best_y[idx, i] + 1.96 * best_ystd[idx, i], "-b", label="95% CI"
                )
                plt.plot(X, best_y[idx, i] - 1.96 * best_ystd[idx, i], "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    # 2D
    elif train_state.X_test.shape[1] == 2:
        for i in range(y_dim):
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(
                train_state.X_test[:, 0],
                train_state.X_test[:, 1],
                best_y[:, i],
                ".",
            )
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$y_{}$".format(i + 1))

    # Residual plot
    # Not necessary to plot
    # if y_test is not None:
    #     plt.figure()
    #     residual = y_test[:, 0] - best_y[:, 0]
    #     plt.plot(best_y[:, 0], residual, "o", zorder=1)
    #     plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Residual = Observed - Predicted")
    #     plt.tight_layout()

    # Uncertainty plot
    # Not necessary to plot
    # if best_ystd is not None:
    #     plt.figure()
    #     for i in range(y_dim):
    #         plt.plot(train_state.X_test[:, 0], best_ystd[:, i], "-b")
    #         plt.plot(
    #             train_state.X_train[:, 0],
    #             np.interp(
    #                 train_state.X_train[:, 0], train_state.X_test[:, 0], best_ystd[:, i]
    #             ),
    #             "ok",
    #         )
    #     plt.xlabel("x")
    #     plt.ylabel("std(y)")


def save_best_state(train_state, fname_train, fname_test):
    """Save the best result of the smallest training loss to a file."""
    if isinstance(train_state.X_train, (list, tuple)):
        print(
            "Error: The network has multiple inputs, and saving such result han't been implemented."
        )
        return

    print("Saving training data to {} ...".format(fname_train))
    y_train, y_test, best_y, best_ystd = _pack_data(train_state)
    if y_train is None:
        np.savetxt(fname_train, train_state.X_train, header="x")
    else:
        train = np.hstack((train_state.X_train, y_train))
        np.savetxt(fname_train, train, header="x, y")

    print("Saving test data to {} ...".format(fname_test))
    if y_test is None:
        test = np.hstack((train_state.X_test, best_y))
        if best_ystd is None:
            np.savetxt(fname_test, test, header="x, y_pred")
        else:
            test = np.hstack((test, best_ystd))
            np.savetxt(fname_test, test, header="x, y_pred, y_std")
    else:
        test = np.hstack((train_state.X_test, y_test, best_y))
        if best_ystd is None:
            np.savetxt(fname_test, test, header="x, y_true, y_pred")
        else:
            test = np.hstack((test, best_ystd))
            np.savetxt(fname_test, test, header="x, y_true, y_pred, y_std")


def dat_to_csv(dat_file_path, csv_file_path, columns):
    """Converts a dat file to CSV format and saves it.

    Args:
        dat_file_path (string): Path of the dat file.
        csv_file_path (string): Desired path of the CSV file.
        columns (list): Column names to be added in the CSV file.
    """
    with open(dat_file_path, "r", encoding="utf-8") as dat_file, open(
        csv_file_path, "w", encoding="utf-8", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)
        for line in dat_file:
            if "#" in line:
                continue
            row = [field.strip() for field in line.split(" ")]
            csv_writer.writerow(row)


def isclose(a, b):
    """A modified version of `np.isclose` for DeepXDE.

    This function changes the value of `atol` due to the dtype of `a` and `b`.
    If the dtype is float16, `atol` is `1e-4`.
    If it is float32, `atol` is `1e-6`.
    Otherwise (for float64), the default is `1e-8`.
    If you want to manually set `atol` for some reason, use `np.isclose` instead.

    Args:
        a, b (array like): Input arrays to compare.
    """
    a_dtype = np.asarray(a).dtype
    b_dtype = np.asarray(b).dtype
    atol = 1e-8
    if np.float32 in [a_dtype, b_dtype]:
        atol = 1e-6
    if np.float16 in [a_dtype, b_dtype]:
        atol = 1e-4
    return np.isclose(a, b, atol=atol)
