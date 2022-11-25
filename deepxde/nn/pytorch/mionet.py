import torch

from .fnn import FNN
from .nn import NN
from .. import activations


class MIONetCartesianProd(NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=None,
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation["merger"])
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation["output merger"])
            if callable(layer_sizes_output_merger[1]):
                # User-defined network
                self.output_merger = layer_sizes_output_merger[1]
            else:
                # Fully connected network
                self.output_merger = FNN(
                    layer_sizes_output_merger,
                    self.activation_output_merger,
                    kernel_initializer,
                )
        else:
            self.output_merger = None
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2
            elif self.merge_operation == "mul":
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(
                    f"{self.merge_operation} operation to be implimented"
                )
        # Optional merger net
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            y_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        # Dot product
        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of merger net and trunk net do not match."
            )
        # output merger net
        if self.output_merger is None:
            y = torch.einsum("ip,jp->ij", y_func, y_loc)
        else:
            y_func = y_func[:, None, :]
            y_loc = y_loc[None, :]
            if self.output_merge_operation == "mul":
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == "add":
                y = y_func + y_loc
            elif self.output_merge_operation == "cat":
                y_func = y_func.repeat(1, y_loc.shape[1], 1)
                y_loc = y_loc.repeat(y_func.shape[0], 1, 1)
                y = torch.cat((y_func, y_loc), dim=2)
            shape0 = y.shape[0]
            shape1 = y.shape[1]
            y = y.reshape(shape0 * shape1, -1)
            y = self.output_merger(y)
            y = y.reshape(shape0, shape1)
        # Add bias
        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


class PODMIONet(NN):
    """MIONet with two input functions and proper orthogonal decomposition (POD)
    for Cartesian product format."""

    def __init__(
        self,
        pod_basis,
        layer_sizes_branch1,
        layer_sizes_branch2,
        activation,
        kernel_initializer,
        layer_sizes_trunk=None,
        regularization=None,
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
            self.activation_merger = activations.get(activation["merger"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if layer_sizes_merger is not None:
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        # connect two branch outputs
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2
            elif self.merge_operation == "mul":
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(
                    f"{self.merge_operation} operation to be implimented"
                )
        # Optional merger net
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        # Dot product
        if self.trunk is None:
            # POD only
            y = torch.einsum("bi,ni->bn", y_func, self.pod_basis)
        else:
            y_loc = self.trunk(x_loc)
            if self.trunk_last_activation:
                y_loc = self.activation_trunk(y_loc)
            y = torch.einsum("bi,ni->bn", y_func, torch.cat((self.pod_basis, y_loc), 1))
            y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
