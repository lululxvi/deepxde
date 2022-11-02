"""Paddle interface for Paddle Probability optimizers."""

__all__ = ["lbfgs_minimize"]

# Possible solutions of Paddle L-BFGS
# https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/incubate/optimizer/functional/minimize_lbfgs_cn.html#minimize-lbfgs3

import numpy as np
import paddle

import deepxde.backend as bkd
from ..config import LBFGS_options


class LossAndFlatGradient:
    """A helper class to create a function required by paddle.incubate.optimizer.functional.minimize_lbfgs

    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, trainable_variables, build_loss):
        self.trainable_variables = trainable_variables
        self.build_loss = build_loss

        # Shapes of all trainable parameters
        self.shapes = []
        for train_var in trainable_variables:
            self.shapes.append(train_var.shape)
        self.n_tensors = len(self.shapes)

        # Information for dynamic_stitch and dynamic_partition
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        self.partitions.append([0])
        for i, shape in enumerate(self.shapes):
            n = np.product(shape) # number of every param
            count += n
            self.partitions.append([count])
        print("partition = ", self.partitions)


    def __call__(self, weights_1d):
        """A function that can be used by paddle.incubate.optimizer.functional.minimize_lbfgs.

        Args:
           weights_1d: a 1D Paddle.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        self.set_flat_weights(weights_1d)
        loss = self.build_loss()
        # Calculate gradients and convert to 1D tf.Tensor
        loss.backward()
        grads = []
        for param in self.trainable_variables:
            grads.append(param.grad)
        return loss, grads

    def dynamic_partition(self, weights_1d, partitions, param_num):
        weights_nd = []
        for i in range(0, param_num+1) :
            tmp = weights_1d[partitions[i]: partitions[i+1]]
            weights_nd.append(tmp)
        return weights_nd


    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D Paddle.Tensor.

        Args:
            weights_1d: a 1D Paddle.Tensor representing the trainable variables.
        """
        weights = self.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        # update the params 
        for i, (shape, param) in enumerate(zip(self.shapes, weights)):
            paddle.assign(param.reshape(shape), self.trainable_variables[i])
     
    

    def to_flat_weights(self, weights):
        """Returns a 1D Paddle.Tensor representing the `weights`.

        Args:
            weights: A list of Paddle.Tensor representing the weights.

        Returns:
            A 1D Paddle.Tensor representing the `weights`.
        """
        return paddle.concat([paddle.flatten(w) for w in weights])


def lbfgs_minimize(trainable_variables, build_loss, previous_optimizer_results=None):
    """Paddle interface for lbfgs.optimizer.lbfgs_minimize.

    Args:
        trainable_variables: Trainable variables, also used as the initial position.
        build_loss: A function to build the loss function expression.
    """
    print("orign : ", trainable_variables)
    func = LossAndFlatGradient(trainable_variables, build_loss)
    print("after : ", trainable_variables)
    initial_position = None
    if previous_optimizer_results is None:
        initial_position = func.to_flat_weights(trainable_variables)

    LBFGS_options["iter_per_step"] = min(1000, LBFGS_options["maxiter"])

    results = paddle.incubate.optimizer.functional.minimize_lbfgs(
        func,
        initial_position=initial_position,
        history_size=100, 
        max_iters=LBFGS_options["iter_per_step"], 
        tolerance_grad=LBFGS_options["gtol"], 
        tolerance_change=LBFGS_options["ftol"], 
        initial_inverse_hessian_estimate=None, 
        line_search_fn=None, 
        max_line_search_iters=LBFGS_options["maxls"], 
        initial_step_length=1.0, 
        dtype='float32', 
        name=None
    )
    # The third return result is updated weight
    func.set_flat_weights(results[2])
    return results
