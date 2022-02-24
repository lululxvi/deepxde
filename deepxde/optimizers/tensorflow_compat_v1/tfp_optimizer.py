"""TensorFlow interface for TensorFlow Probability optimizers."""

# This is a standalone code to show how to use TFP in TensorFlow 1.x. Code below is
# inspired by https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993.
# But the DeepXDE interface is still under development...
# References:
# - https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize
# - https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/optimizer/lbfgs_test.py
# - https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
# - https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993


import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from matplotlib import pyplot

tf.disable_v2_behavior()


class LossAndFlatGradient:
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, build_loss):
        self.build_loss = build_loss

        # Shapes of all trainable parameters
        self.shapes = [v.get_shape().as_list() for v in tf.trainable_variables()]
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        Args:
           weights_1d: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """

        # Update the weights
        update_ops = self.set_flat_weights(weights_1d)
        with tf.control_dependencies([update_ops]):
            # Calculate the loss
            loss = self.build_loss()

        # Calculate gradients and convert to 1D tf.Tensor
        grads = tf.gradients(loss, tf.trainable_variables())
        grads = tf.dynamic_stitch(self.indices, grads)
        return loss, grads

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.

        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.

        Returns:
            A ``tf.Operation``.
        """

        weights = tf.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        variables = tf.trainable_variables()
        update_ops = []
        for i, (shape, param) in enumerate(zip(self.shapes, weights)):
            update_ops.append(variables[i].assign(tf.reshape(param, shape)))
        return tf.group(*update_ops)


def plot_helper(inputs, outputs, title, fname):
    """Plot helper"""
    pyplot.figure()
    pyplot.tricontourf(inputs[:, 0], inputs[:, 1], outputs.flatten(), 100)
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)


if __name__ == "__main__":
    sess = tf.keras.backend.get_session()

    # use float64 by default
    # tf.keras.backend.set_floatx("float64")

    # prepare training data
    x_1d = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
    x1, x2 = np.meshgrid(x_1d, x_1d)
    inps = np.stack((x1.flatten(), x2.flatten()), 1)
    outs = np.reshape(inps[:, 0] ** 2 + inps[:, 1] ** 2, (x_1d.size ** 2, 1))

    train_x = tf.placeholder(tf.float32, [None, 2])
    train_y = tf.placeholder(tf.float32, [None, 1])

    # prepare prediction model, loss function, and the function passed to L-BFGS solver
    loss_fun = tf.keras.losses.MeanSquaredError()

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[2]),
            tf.keras.layers.Dense(64, "tanh"),
            tf.keras.layers.Dense(64, "tanh"),
            tf.keras.layers.Dense(1, None),
        ]
    )
    func = LossAndFlatGradient(lambda: loss_fun(model(train_x), train_y))

    # W = tf.Variable(tf.random_normal([2, 1]))
    # b = tf.Variable(tf.zeros(1))
    # y = tf.matmul(train_x, W) + b
    # loss = loss_fun(y, train_y)

    # def build_loss():
    #     return loss + 0

    # func = LossAndFlatGradient(build_loss)

    # convert initial model parameters to a 1D tf.Tensor
    params = tf.dynamic_stitch(func.indices, tf.trainable_variables())
    # train the model with L-BFGS solver
    lbfgs_op = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func, initial_position=params, max_iterations=100
    )

    sess.run(tf.global_variables_initializer())
    for i in range(5):
        results = sess.run(lbfgs_op, feed_dict={train_x: inps, train_y: outs})
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        sess.run(func.set_flat_weights(results.position))
        print(
            i,
            "converged:",
            results.converged,
            "failed:",
            results.failed,
            "num_iterations:",
            results.num_iterations,
            "loss:",
            results.objective_value,
        )
        if results.converged or results.failed:
            break

    # do some prediction
    pred_outs = sess.run(model(inps))
    # pred_outs = sess.run(y, feed_dict={train_x: inps})

    err = np.abs(pred_outs - outs)
    print("\nL2-error norm: {}".format(np.linalg.norm(err) / np.sqrt(11)))

    # plot figures
    plot_helper(inps, outs, "Exact solution", "ext_soln.png")
    plot_helper(inps, pred_outs, "Predicted solution", "pred_soln.png")
    plot_helper(inps, err, "Absolute error", "abs_err.png")
    pyplot.show()
