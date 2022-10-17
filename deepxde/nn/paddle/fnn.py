import paddle
import numpy as np
from .nn import NN
from .. import activations
from .. import initializers
from paddle.nn.initializer import Assign, Constant
from paddle import ParamAttr
import os

class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer, task_name=None):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = paddle.nn.LayerList()
        # npz = np.load('/workspace/hesensen/paddlescience_project/deepxde_wrt_new/iter0_weights.npz')
        p = 0
        for i in range(1, len(layer_sizes)):
            # weight_attr_ = paddle.ParamAttr(initializer = paddle.nn.initializer.Assign(w_array[i-1]))
            # self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i],weight_attr=weight_attr_))
            # self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if isinstance(task_name, str) and os.path.exists(f"/workspace/hesensen/paddlescience_project/deepxde_wrt_new/{task_name}/linears.{i-1}.weight.npy") and os.path.exists(f"/workspace/hesensen/paddlescience_project/deepxde_wrt_new/{task_name}/linears.{i-1}.bias.npy"):
                print("load param from file")
                self.linears.append(
                    paddle.nn.Linear(
                        layer_sizes[i - 1],
                        layer_sizes[i],
                        # weight_attr=ParamAttr(initializer=Assign(np.load(f"/workspace/hesensen/paddlescience_project/deepxde_wrt_new/{task_name}/linears.{i-1}.weight.npy"))),
                        # bias_attr=ParamAttr(initializer=Assign(np.load(f"/workspace/hesensen/paddlescience_project/deepxde_wrt_new/{task_name}/linears.{i-1}.bias.npy")))
                        # weight_attr=ParamAttr(initializer=Constant(0.5)),
                        # bias_attr=ParamAttr(initializer=Constant(0.5))
                    )
                )
                # initializer(self.linears[-1].weight)
                # initializer_zero(self.linears[-1].bias)
                # self.linears[-1].weight.set_value(npz[f'arr_{p}'])
                # self.linears[-1].bias.set_value(npz[f'arr_{p+1}'])
                p += 2
                # print(f"{i} {self.linears[-1].weight.mean().item():.10f} {self.linears[-1].weight.std().item():.10f}")
            else:
                print("init param from random")
                self.linears.append(
                    paddle.nn.Linear(
                        layer_sizes[i - 1],
                        layer_sizes[i],
                    )
                )
                initializer(self.linears[-1].weight)
                initializer_zero(self.linears[-1].bias)

        # debug info
        if paddle.in_dynamic_mode():
            f = open('paddle_param_dygraph.log','ab')
            for linear in self.linears:
                np.savetxt(f,linear.weight.numpy().reshape(1,-1),delimiter=",")
                np.savetxt(f,linear.bias.numpy().reshape(1,-1),delimiter=",")
            f.close()
        # debug info end

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    def __init__(self, layer_sizes, activation, kernel_initializer, task_name=None):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        if len(layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(layer_sizes[0], int):
            raise ValueError("input size must be integer")
        if not isinstance(layer_sizes[-1], int):
            raise ValueError("output size must be integer")

        n_output = layer_sizes[-1]

        def make_linear(n_input, n_output):
            linear = paddle.nn.Linear(n_input, n_output)
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear

        self.layers = paddle.nn.LayerList()
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError(
                        "number of sub-layers should equal number of network outputs"
                    )
                if isinstance(prev_layer_size, (list, tuple)):
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    self.layers.append(
                        paddle.nn.LayerList(
                            [
                                make_linear(prev_layer_size[j], curr_layer_size[j])
                                for j in range(n_output)
                            ]
                        )
                    )
                else:  # e.g. 64 -> [8, 8, 8]
                    self.layers.append(
                        paddle.nn.LayerList(
                            [
                                make_linear(prev_layer_size, curr_layer_size[j])
                                for j in range(n_output)
                            ]
                        )
                    )
            else:  # e.g. 64 -> 64
                if not isinstance(prev_layer_size, int):
                    raise ValueError(
                        "cannot rejoin parallel subnetworks after splitting"
                    )
                self.layers.append(make_linear(prev_layer_size, curr_layer_size))

        # output layers
        if isinstance(layer_sizes[-2], (list, tuple)):  # e.g. [3, 3, 3] -> 3
            self.layers.append(
                paddle.nn.LayerList(
                    [make_linear(layer_sizes[-2][j], 1) for j in range(n_output)]
                )
            )
        else:
            self.layers.append(make_linear(layer_sizes[-2], n_output))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.layers[:-1]:
            if isinstance(layer, paddle.nn.LayerList):
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    x = [self.activation(f(x)) for f in layer]
            else:
                x = self.activation(layer(x))

        # output layers
        if isinstance(x, list):
            x = paddle.concat([f(x_) for f, x_ in zip(self.layers[-1], x)], axis=1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
