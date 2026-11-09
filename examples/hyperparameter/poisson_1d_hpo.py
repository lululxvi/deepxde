"""Backend supported: pytorch

This script runs Ray Tune hyperparameter optimization for a 1D Poisson DeepONet using HyperNOs (https://github.com/MaxGhi8/HyperNOs) and DeepXDE.
The problem setup is based on DeepXDE/examples/operator/poisson_1d_pideeponet.py.
"""

import os

# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
from hypernos.datasets import deeponet_collate_fn
from hypernos.loss_fun import LprelLoss

# Import HyperNOs components
from hypernos.tune import tune_hyperparameters
from ray import tune
from torch.utils.data import DataLoader

import deepxde as dde


# 1. Define the DeepXDE PDE problem
def solve_poisson_1d():
    # Poisson equation: -u_xx = f
    def equation(x, y, f):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - f

    geom = dde.geometry.Interval(0, 1)

    def u_boundary(_):
        return 0

    def boundary(_, on_boundary):
        return on_boundary

    bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)
    pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)

    degree = 3
    space = dde.data.PowerSeries(N=degree + 1)

    num_eval_points = 10
    evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

    pde_op = dde.data.PDEOperatorCartesianProd(
        pde,
        space,
        evaluation_points,
        num_function=500,  # Total number of functions to sample
    )
    return pde_op


# 2. Custom Dataset class for Cartesian Product DeepONet
class PDECartesianDataset(torch.utils.data.Dataset):
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return self.branch_data.shape[0]

    def __getitem__(self, idx):
        # Returns (branch_input, trunk_input, target)
        return self.branch_data[idx], self.trunk_data, self.target_data[idx]


# 3. Dataset Builder for HyperNOs
def dataset_builder(config):
    pde_op = solve_poisson_1d()

    # Generate training and test data from DeepXDE operator
    X_train, y_train, aux_train = pde_op.train_next_batch(config["training_samples"])
    X_test, y_test, aux_test = pde_op.test()

    # In PDEOperatorCartesianProd, y_train is often None (physics-informed).
    # For this HPO demo, we'll use a dummy target if y is None,
    # or you could solve the PDE to get ground truth.
    if y_train is None:
        y_train = np.ones((X_train[0].shape[0], X_train[1].shape[0], 1))
    if y_test is None:
        y_test = np.ones((X_test[0].shape[0], X_test[1].shape[0], 1))

    # Squeeze the target if it's single-output to match DeepONetCartesianProd output shape (batch, num_points)
    if y_train is not None and y_train.shape[-1] == 1:
        y_train = y_train.squeeze(-1)
    if y_test is not None and y_test.shape[-1] == 1:
        y_test = y_test.squeeze(-1)

    class PDEData:
        def __init__(self, X_train, y_train, X_test, y_test, batch_size):
            self.train_loader = DataLoader(
                PDECartesianDataset(X_train[0], X_train[1], y_train),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=deeponet_collate_fn,
            )
            self.val_loader = DataLoader(
                PDECartesianDataset(X_test[0], X_test[1], y_test),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=deeponet_collate_fn,
            )
            self.test_loader = self.val_loader

    return PDEData(X_train, y_train, X_test, y_test, config["batch_size"])


# 4. Model Builder for HyperNOs
def model_builder(config):
    num_eval_points = 10
    dim_x = 1
    p = config["network_width"]

    layer_sizes_branch = (
        [num_eval_points] + [config["network_width"]] * config["branch_depth"] + [p]
    )
    layer_sizes_trunk = (
        [dim_x] + [config["network_width"]] * config["trunk_depth"] + [p]
    )

    model = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch,
        layer_sizes_trunk,
        "tanh",
        "Glorot normal",
    )
    return model


def main():
    # Define the Search Space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "network_width": tune.choice([16, 32]),
        "branch_depth": tune.choice([1, 2]),
        "trunk_depth": tune.choice([1, 2]),
        "batch_size": 32,
        "training_samples": 200,
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 1,  # Required by some HyperNOs logic
    }

    # Initial points for HyperOpt
    default_hyper_params = {
        "learning_rate": 0.001,
        "network_width": 32,
        "branch_depth": 2,
        "trunk_depth": 2,
        "batch_size": 32,
        "training_samples": 200,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.95,
        "problem_dim": 1,
    }

    # Loss function: Relative L2 norm (data-driven part)
    # Note: DeepXDE's PDEOperatorCartesianProd also supports physics-informed loss,
    # but for this HPO example we focus on the data-driven fit of the operator.
    loss_fn = LprelLoss(p=2, size_mean=True)

    print("Starting Hyperparameter Optimization with HyperNOs...")

    best_result = tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        default_hyper_params=[default_hyper_params],
        num_samples=4,  # Low number of samples for demonstration
        max_epochs=20,  # Low number of epochs for demonstration
        grace_period=5,
        reduction_factor=2,
        runs_per_cpu=1.0,
        runs_per_gpu=0.0,  # Set to 1.0 if GPU is available
    )

    print("\nBest hyperparameters found:")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")

    print(f"\nBest relative loss: {best_result.metrics['relative_loss']:.6f}")


if __name__ == "__main__":
    main()
