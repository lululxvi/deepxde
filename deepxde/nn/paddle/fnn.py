import paddle
import numpy as np
from .nn import NN
from .. import activations
from .. import initializers
from paddle.nn.initializer import Assign
from paddle import ParamAttr
import os

class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer, task_name=None):
        super().__init__()
        self.activation = activations.get(activation)
        self.layer_size = layer_sizes
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = paddle.nn.LayerList()
        p = 0
        for i in range(1, len(layer_sizes)):
            if isinstance(task_name, str) and os.path.exists(f"./{task_name}/linears.{i-1}.weight.npy") and os.path.exists(f"./{task_name}/linears.{i-1}.bias.npy"):
                print("load param from file")
                self.linears.append(
                    paddle.nn.Linear(
                        layer_sizes[i - 1],
                        layer_sizes[i],
                        weight_attr=ParamAttr(initializer=Assign(np.load(f"./{task_name}/linears.{i-1}.weight.npy").astype("float32"))),
                        bias_attr=ParamAttr(initializer=Assign(np.load(f"./{task_name}/linears.{i-1}.bias.npy").astype("float32")))
                    )
                )
                p += 2
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
        self.p = 0
        self.new_save = False

        def make_linear(n_input, n_output):
            if isinstance(task_name, str) and os.path.exists(f"./{task_name}/weight_{self.p}.npy") and os.path.exists(f"./{task_name}/bias_{self.p}.npy"):
                print("load param from file")
                linear = paddle.nn.Linear(
                    n_input,
                    n_output,
                    weight_attr=ParamAttr(initializer=Assign(np.load(f"./{task_name}/weight_{self.p}.npy").astype("float32"))),
                    bias_attr=ParamAttr(initializer=Assign(np.load(f"./{task_name}/bias_{self.p}.npy").astype("float32")))
                )
                self.p += 1
            else:
                print("init param from random")
                linear = paddle.nn.Linear(n_input, n_output)
                initializer(linear.weight)
                initializer_zero(linear.bias)
                # np.save(f"./{task_name}/weight_{self.p}.npy", linear.weight.numpy())
                # np.save(f"./{task_name}/bias_{self.p}.npy", linear.bias.numpy())
                self.p += 1
                self.new_save = True
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

        # if self.new_save:
        #     print("第一次保存模型完毕，自动退出，请再次运行")
        #     exit(0)

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
