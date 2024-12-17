"""TensorFlow interface for TensorFlow Probability optimizers."""

__all__ = ["lbfgs_minimize"]

# Possible solutions of L-BFGS
# TensorFlow (waiting...)
# - https://github.com/tensorflow/tensorflow/issues/48167
# TFP
# - https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize
# - https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/optimizer/lbfgs_test.py
# - https://github.com/tensorflow/probability/issues/565
# - https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
# - https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
# - https://github.com/tensordiffeq/TensorDiffEq/blob/main/tensordiffeq/optimizers.py
# SciPy
# - https://github.com/sciann/sciann/blob/master/sciann/utils/optimizers.py
# Manually
# - https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py
# - https://github.com/tensordiffeq/TensorDiffEq/blob/main/tensordiffeq/optimizers.py

# Code below is modified from https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..config import LBFGS_options


class LossAndFlatGradient:
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, trainable_variables, build_loss):
        self.trainable_variables = trainable_variables
        self.build_loss = build_loss

        # Shapes of all trainable parameters
        self.shapes = tf.shape_n(trainable_variables)
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.prod(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    # @tf.function(jit_compile=True) has an error.
    @tf.function
    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        Args:
           weights_1d: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        self.set_flat_weights(weights_1d)
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss = self.build_loss()
        # Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.dynamic_stitch(self.indices, grads)
        return loss, grads

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.

        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        weights = tf.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, weights)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.

        Args:
            weights: A list of tf.Tensor representing the weights.

        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)


def lbfgs_minimize(trainable_variables, build_loss, previous_optimizer_results=None):
    """TensorFlow interface for tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables: Trainable variables, also used as the initial position.
        build_loss: A function to build the loss function expression.
        previous_optimizer_results
    """
    func = LossAndFlatGradient(trainable_variables, build_loss)
    initial_position = None
    if previous_optimizer_results is None:
        initial_position = func.to_flat_weights(trainable_variables)
    results = tfp.optimizer.lbfgs_minimize(
        func,
        initial_position=initial_position,
        previous_optimizer_results=previous_optimizer_results,
        num_correction_pairs=LBFGS_options["maxcor"],
        tolerance=LBFGS_options["gtol"],
        x_tolerance=0,
        f_relative_tolerance=LBFGS_options["ftol"],
        max_iterations=LBFGS_options["maxiter"],
        parallel_iterations=1,
        max_line_search_iterations=LBFGS_options["maxls"],
    )
    # The final optimized parameters are in results.position.
    # Set them back to the variables.
    func.set_flat_weights(results.position)
    return results
