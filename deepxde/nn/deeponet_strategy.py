from abc import ABC, abstractmethod


class DeepONetStrategy(ABC):
    """DeepONet building strategy.

    See the section 3.1.6. in
    L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. Karniadakis.
    A comprehensive and fair comparison of two neural operators
    (with practical extensions) based on FAIR data.
    Computer Methods in Applied Mechanics and Engineering, 393, 114778, 2022.
    """

    def __init__(self, net):
        self.net = net

    @abstractmethod
    def build(self, layer_sizes_branch, layer_sizes_trunk):
        """Build branch and trunk nets."""

    @abstractmethod
    def call(self, x_func, x_loc):
        """Forward pass."""


class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        branch = self.net.build_branch_net(layer_sizes_branch)
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = self.net.merge_branch_trunk(x_func, x_loc, 0)
        return x


class IndependentStrategy(DeepONetStrategy):
    """Directly use n independent DeepONets,
    and each DeepONet outputs only one function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        single_output_strategy = SingleOutputStrategy(self.net)
        branch, trunk = [], []
        for _ in range(self.net.num_outputs):
            branch_, trunk_ = single_output_strategy.build(
                layer_sizes_branch, layer_sizes_trunk
            )
            branch.append(branch_)
            trunk.append(trunk_)
        return branch, trunk

    def call(self, x_func, x_loc):
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = self.net.branch[i](x_func)
            x_loc_ = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
        return self.net.concatenate_outputs(xs)


class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and the trunk net into n groups,
    and then the kth group outputs the kth solution.

    For example, if n = 2 and both the branch and trunk nets have 100 output neurons,
    then the dot product between the first 50 neurons of
    the branch and trunk nets generates the first function,
    and the remaining 50 neurons generate the second function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        single_output_strategy = SingleOutputStrategy(self.net)
        return single_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_func and x_loc into respective outputs
        shift = 0
        size = x_func.shape[1] // self.net.num_outputs
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitBranchStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_branch[-1] / self.net.num_outputs != layer_sizes_trunk[-1]:
            raise AssertionError(
                f"Output size of the trunk net does not equal to {layer_sizes_branch[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_func into respective outputs
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitTrunkStrategy(DeepONetStrategy):
    """Split the trunk net and share the branch net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_trunk[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the trunk net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_trunk[-1] / self.net.num_outputs != layer_sizes_branch[-1]:
            raise AssertionError(
                f"Output size of the branch net does not equal to {layer_sizes_trunk[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_loc into respective outputs
        shift = 0
        size = x_func.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)
