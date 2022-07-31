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
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.tensor(0.0, requires_grad=True)
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            y_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        # Dot product
        if y_func1.shape[-1] != y_func2.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net1 and branch net2 do not match."
            )
        y = torch.mul(y_func1, y_func2)
        if y.shape[-1] != y_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        y = torch.einsum("ip,jp->ij", y, y_loc)
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
        connect_method="mul",
        layer_sizes_trunk=None,
        regularization=None,
        trunk_last_activation=False,
        layer_sizes_merge=None,
        merge_net_after_branch=False,
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
            self.activation_merge = activations.get(activation["merge"])
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
        self.merge_net_after_branch = merge_net_after_branch
        if self.merge_net_after_branch is True:
            if callable(layer_sizes_merge[1]):
                # User-defined network
                self.merge = layer_sizes_merge[1]
            else:
                # Fully connected network
                self.merge = FNN(
                    layer_sizes_merge, self.activation_merge, kernel_initializer
                )
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = torch.tensor(0.0, requires_grad=True)
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.connect_method = connect_method

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        # connect two branch outputs
        if self.connect_method == "cat":
            x_merge = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.connect_method == "sum":
                x_merge = y_func1 + y_func2
            elif self.connect_method == "mul":
                x_merge = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(
                    f"{self.connect_method} method to be implimented"
                )
        # Optional merge net
        if self.merge_net_after_branch is True:
            y_func = self.merge(x_merge)
        else:
            y_func = x_merge
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
