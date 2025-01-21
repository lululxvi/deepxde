# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


"""External utilities."""

import csv
import os
from multiprocessing import Pool

import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


def apply(func, args=None, kwds=None):
    """Launch a new process to call the function.

    This can be used to clear Tensorflow GPU memory after trainer execution:
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

    train_exp_dim = False
    if u.math.ndim(X_train) == 1:
        train_exp_dim = True
        X_train = X_train.reshape(-1, 1)
    test_exp_dim = False
    if u.math.ndim(X_test) == 1:
        test_exp_dim = True
        X_test = X_test.reshape(-1, 1)

    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if train_exp_dim:
        X_train = X_train.flatten()
    if test_exp_dim:
        X_test = X_test.flatten()
    return X_train, X_test


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
            ``Trainer.train()``.
        train_state: ``TrainState`` instance. The second variable returned from
            ``Trainer.train()``.
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
            ``Trainer.train()``.
        fname (string): If `fname` is a string (e.g., 'loss_history.png'), then save the
            figure to the file of the file name `fname`.
    """
    # np.sum(loss_history.loss_train, axis=1) is error-prone for arrays of varying lengths.
    # Handle irregular array sizes.
    loss_train = jnp.array([jnp.sum(jnp.asarray(jax.tree.leaves(loss))) for loss in loss_history.loss_train])
    loss_test = jnp.array([jnp.sum(jnp.asarray(jax.tree.leaves(loss))) for loss in loss_history.loss_test])

    plt.figure()
    plt.semilogy(loss_history.steps, loss_train, label="Train loss")
    plt.semilogy(loss_history.steps, loss_test, label="Test loss")
    metric_tests = jax.tree.map(lambda *a: u.math.asarray(a), *loss_history.metrics_test)

    for i in range(len(loss_history.metrics_test[0])):
        if isinstance(metric_tests[i], dict):
            for k, v in metric_tests[i].items():
                plt.semilogy(loss_history.steps, v, label=f"Test metric {k}")
        else:
            plt.semilogy(loss_history.steps, metric_tests[i], label=f"Test metric {i}")
    plt.xlabel("# Steps")
    plt.legend()

    if isinstance(fname, str):
        plt.savefig(fname)


def save_loss_history(loss_history, fname):
    """Save the training and testing loss history to a file."""
    print("Saving loss history to {} ...".format(fname))

    train_losses = jax.tree.map(lambda *a: u.math.asarray(a), *loss_history.loss_train)
    braintools.file.msgpack_save(fname, train_losses)


def _pack_data(train_state):
    def merge_values(values):
        if values is None:
            return None
        return jnp.hstack(values) if isinstance(values, (list, tuple)) else values

    # y_train = merge_values(train_state.y_train)
    # y_test = merge_values(train_state.y_test)
    # best_y = merge_values(train_state.best_y)
    # best_ystd = merge_values(train_state.best_ystd)
    y_train = train_state.y_train
    y_test = train_state.y_test
    best_y = train_state.best_y
    best_ystd = train_state.best_ystd
    return y_train, y_test, best_y, best_ystd


def plot_best_state(train_state):
    """Plot the best result of the smallest training loss.

    This function only works for 1D and 2D problems. For other problems and to better
    customize the figure, use ``save_best_state()``.

    Note:
        You need to call ``plt.show()`` to show the figure.

    Args:
        train_state: ``TrainState`` instance. The second variable returned from
            ``Trainer.train()``.
    """
    if isinstance(train_state.X_train, (list, tuple)):
        print("Error: The network has multiple inputs, and plotting such result hasn't been implemented.")
        return

    y_train, y_test, best_y, best_ystd = _pack_data(train_state)
    xkeys = tuple(train_state.X_test.keys())

    # Regression plot
    # 1D
    if len(train_state.X_test) == 1:
        idx = u.math.argsort(train_state.X_test[xkeys[0]])
        X = train_state.X_test[xkeys[0]][idx]
        plt.figure()
        for ykey in best_y:
            if y_train is not None:
                plt.plot(train_state.X_train[xkeys[0]], y_train[ykey], "ok", label="Train")
            if y_test is not None:
                plt.plot(X, y_test[ykey], "-k", label="True")
            y_val, y_unit = u.split_mantissa_unit(best_y[ykey])
            plt.plot(
                X, y_val, "--r",
                label=(f"{ykey} Prediction"
                       if y_unit.is_unitless else
                       f"{ykey} Prediction [{y_unit}]")
            )
            if best_ystd is not None:
                ystd_val = u.get_magnitude(best_ystd[ykey].to(y_unit))
                plt.plot(X, y_val + 1.96 * ystd_val, "-b", label="95% CI")
                plt.plot(X, y_val - 1.96 * ystd_val, "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    # 2D
    elif len(train_state.X_test) == 2:
        for ykey in best_y:
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(
                u.get_magnitude(train_state.X_test[xkeys[0]]),
                u.get_magnitude(train_state.X_test[xkeys[1]]),
                u.get_magnitude(best_y[ykey]),
                ".",
            )
            unit = u.get_unit(train_state.X_test[xkeys[0]])
            if unit.is_unitless:
                ax.set_xlabel(f'{xkeys[0]}')
            else:
                ax.set_xlabel(f'{xkeys[0]} [{unit}]')
            unit = u.get_unit(train_state.X_test[xkeys[1]])
            if unit.is_unitless:
                ax.set_ylabel(f'{xkeys[1]}')
            else:
                ax.set_ylabel(f'{xkeys[1]} [{unit}]')
            unit = u.get_unit(best_y[ykey])
            if unit.is_unitless:
                ax.set_zlabel(f'{ykey}')
            else:
                ax.set_zlabel(f'{ykey} [{unit}]')

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
        print("Error: The network has multiple inputs, and saving such result han't been implemented.")
        return

    print("Saving training data to {} ...".format(fname_train))
    y_train, y_test, best_y, best_ystd = _pack_data(train_state)
    if y_train is None:
        data = {'X_train': train_state.X_train}
    else:
        data = {'X_train': train_state.X_train, 'y_train': y_train}
    braintools.file.msgpack_save(fname_train, data)

    print("Saving test data to {} ...".format(fname_test))
    if y_test is None:
        data = {'X_test': train_state.X_test, 'best_y': best_y}
        if best_ystd is not None:
            data['best_ystd'] = best_ystd
        braintools.file.msgpack_save(fname_test, data)
    else:
        data = {'X_test': train_state.X_test, 'best_y': best_y, 'y_test': y_test}
        if best_ystd is not None:
            data['best_ystd'] = best_ystd
        braintools.file.msgpack_save(fname_test, data)


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
        a, b (array like): DictToArray arrays to compare.
    """
    pack = smart_numpy(a)
    a_dtype = a.dtype
    a_unit = u.get_unit(a)
    if a_dtype == jnp.float32:
        atol = u.maybe_decimal(u.Quantity(1e-6, unit=a_unit))
    elif a_dtype == jnp.float16:
        atol = u.maybe_decimal(u.Quantity(1e-4, unit=a_unit))
    else:
        atol = u.maybe_decimal(u.Quantity(1e-8, unit=a_unit))
    return pack.isclose(a, b, atol=atol)


def smart_numpy(x):
    if isinstance(x, jnp.ndarray):
        return jnp
    elif isinstance(x, jax.Array):
        return jax.numpy
    elif isinstance(x, u.Quantity):
        return u.math
    elif isinstance(x, np.ndarray):
        return np
    else:
        raise TypeError(f"Unknown type {type(x)}.")
