"""Backend supported: pytorch

This script demonstrates a comprehensive Hyper-Parameter Optimization (HPO) for a 2D Advection PI-DeepONet.
We explore a multi-dimensional search space including architecture (depth/width),
optimizer dynamics (LR, weight decay), and non-linearities (activations).
"""

import os

# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
from hypernos.datasets import deeponet_collate_fn
from hypernos.loss_fun import LprelLoss
from hypernos.tune import tune_hyperparameters
from ray import tune
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader

import deepxde as dde


def solve_advection_2d():
    """Defines the 2D advection problem as a DeepXDE PDEOperator."""

    def pde_fn(x, y):
        dy_x = dde.grad.jacobian(y, x, j=0)
        dy_y = dde.grad.jacobian(y, x, j=1)
        return dy_y + dy_x

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    # Initial condition at y=0
    ic = dde.icbc.DirichletBC(
        geom, lambda x, v: v, lambda x, on_boundary: on_boundary and np.isclose(x[1], 0)
    )
    pde = dde.data.PDE(geom, pde_fn, ic, num_domain=200, num_boundary=200)
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)

    return dde.data.PDEOperatorCartesianProd(
        pde,
        func_space,
        np.linspace(0, 1, num=50)[:, None],
        1000,
        function_variables=[0],
        num_test=100,
        batch_size=32,
    )


class PDECartesianDataset(torch.utils.data.Dataset):
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return self.branch_data.shape[0]

    def __getitem__(self, idx):
        return self.branch_data[idx], self.trunk_data, self.target_data[idx]


def generate_advection_data(num_functions, num_eval_points=50, num_trunk_points=1000):
    """Generates supervised data for the 2D advection equation u_y + u_x = 0."""
    # 1. Generate Branch Inputs (Initial conditions at y=0)
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
    eval_pts = np.linspace(0, 1, num=num_eval_points, endpoint=False)[:, None]

    features = func_space.random(num_functions)
    # Add a constant offset to ensure the norm is never zero (avoids division by zero in relative loss)
    branch_inputs = func_space.eval_batch(features, eval_pts) + 1.0

    # 2. Generate Trunk Inputs (Random points in the domain [0,1]x[0,1])
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    trunk_inputs = geom.uniform_points(num_trunk_points)
    num_actual_trunk_points = trunk_inputs.shape[0]

    # 3. Generate Targets (Analytical solution u(x, y) = v((x - y) mod 1))
    x_coords = trunk_inputs[:, 0]
    y_coords = trunk_inputs[:, 1]
    shifted_x = (x_coords - y_coords) % 1.0

    # Interpolate branch_inputs to find values at shifted_x
    # Use num_actual_trunk_points to avoid broadcasting errors
    targets = np.zeros((num_functions, num_actual_trunk_points))
    eval_pts_flat = eval_pts.flatten()

    # Append the first point to the end at x=1.0 to close the periodic loop for the interpolator
    x_vals = np.append(eval_pts_flat, 1.0)

    for i in range(num_functions):
        v_vals = np.append(branch_inputs[i], branch_inputs[i][0])
        interpolator = interp1d(
            x_vals, v_vals, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        targets[i] = interpolator(shifted_x)

    return branch_inputs, trunk_inputs, targets


def dataset_builder(config):
    """Prepares DataLoaders for Ray Tune trials using exact analytical targets."""
    # Generate Training Data
    X_b_train, X_t_train, y_train = generate_advection_data(
        config["training_samples"], num_eval_points=50, num_trunk_points=1000
    )

    # Generate Validation Data
    X_b_test, X_t_test, y_test = generate_advection_data(
        100, num_eval_points=50, num_trunk_points=1000
    )

    class PDEData:
        def __init__(self, bs):
            device = (
                torch.get_default_device()
                if hasattr(torch, "get_default_device")
                else "cpu"
            )
            g = torch.Generator(device=device)

            self.train_loader = DataLoader(
                PDECartesianDataset(X_b_train, X_t_train, y_train),
                batch_size=bs,
                shuffle=True,
                collate_fn=deeponet_collate_fn,
                generator=g,
            )
            self.val_loader = DataLoader(
                PDECartesianDataset(X_b_test, X_t_test, y_test),
                batch_size=bs,
                shuffle=False,
                collate_fn=deeponet_collate_fn,
                generator=g,
            )
            self.test_loader = self.val_loader

    return PDEData(config["batch_size"])


def model_builder(config):
    """Constructs the DeepONet model with HPO-defined architecture and activation."""

    p = config["network_width"]

    model = dde.nn.DeepONetCartesianProd(
        [50] + [p] * config["branch_depth"] + [p],
        [5] + [p] * config["trunk_depth"] + [p],
        config["activation"],
        "Glorot normal",
    )

    def periodic(x):
        xt, yt = x[:, :1] * 2 * np.pi, x[:, 1:]
        return torch.cat(
            [torch.cos(xt), torch.sin(xt), torch.cos(2 * xt), torch.sin(2 * xt), yt], 1
        )

    model.apply_feature_transform(periodic)
    return model


def main():
    config_space = {
        # Architecture
        "branch_depth": tune.choice([2, 3, 4]),
        "trunk_depth": tune.choice([2, 3, 4]),
        "network_width": tune.choice([64, 128, 256]),
        "activation": tune.choice(["tanh", "relu", "sigmoid", "silu"]),
        # Optimizer & Training
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": 32,  # Fixed batch size
        "training_samples": 500,  # Fixed training samples
        # Scheduler
        "scheduler_step": tune.choice([5, 10, 15]),
        "scheduler_gamma": tune.uniform(0.9, 0.99),
        "problem_dim": 2,  # Metadata for HyperNOs
    }

    print("Starting Advanced HPO for 2D Advection DeepONet...")

    best_result = tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        LprelLoss(p=2, size_mean=True),
        # Total number of trials to run
        num_samples=10,
        # Maximum epochs per trial
        max_epochs=30,
        # Trials are not stopped before this many epochs (ASHA)
        grace_period=10,
        # Reduction factor for successive halving (ASHA)
        reduction_factor=4,
        # Number of CPU cores per trial
        runs_per_cpu=5.0,
        # Number of GPUs per trial (set > 0 if available)
        runs_per_gpu=0.0,
        # Initial point to guide the search (optional but powerful)
        default_hyper_params=[
            {
                "branch_depth": 3,
                "trunk_depth": 3,
                "network_width": 128,
                "activation": "tanh",
                "learning_rate": 0.001,
                "batch_size": 32,
                "training_samples": 500,
                "weight_decay": 1e-5,
                "scheduler_step": 10,
                "scheduler_gamma": 0.95,
                "problem_dim": 2,
            }
        ],
    )

    print("\n" + "=" * 30)
    print("BEST CONFIGURATION FOUND")
    print("=" * 30)
    for key, value in best_result.config.items():
        print(f"{key:20}: {value}")
    print("-" * 30)
    print(f"Best Relative L2 Loss: {best_result.metrics['relative_loss']:.6f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
