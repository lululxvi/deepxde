2D advection: Comprehensive HPO with HyperNOs
=============================================

This example demonstrates a rich hyper-parameter optimization (HPO) for a 2D advection problem using `HyperNOs <https://github.com/MaxGhi8/HyperNOs>`_ and `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_.

In scientific machine learning, finding the right combination of network architecture, optimizer settings, and non-linear activation functions is often more of an art than a science. This demo showcases how to automate this process by exploring a high-dimensional search space efficiently.

.. note::

    This code takes about 5 minutes to run on a AMD Ryzen 5 7600 with 6 physical cores (12 threads).

Problem setup
-------------

We consider the 2D advection equation:

.. math:: \frac{\partial u}{\partial y} + \frac{\partial u}{\partial x} = 0, \quad (x, y) \in [0, 1] \times [0, 1]

The goal is to learn the solution operator mapping initial conditions at :math:`y=0` to the full spatial domain. We use a PI-DeepONet architecture with a periodic feature transform on the trunk net to capture the physics of the problem.

Multi-Dimensional Search Space
------------------------------

We define a rich search space that covers:

1. **Architecture**:
   - ``branch_depth`` & ``trunk_depth``: Exploring the capacity of each sub-network (2 to 4 layers).
   - ``network_width``: Hidden layer size (64 to 256).
   - ``activation``: Comparing standard (``tanh``, ``sigmoid``, ``relu``) and modern (``silu``) functions.

2. **Optimization**:
   - ``learning_rate``: Log-uniform search between :math:`1e-4` and :math:`5e-3`.
   - ``weight_decay``: Regularization strength.

3. **Learning Rate Scheduler**:
   - ``scheduler_step`` & ``scheduler_gamma``: Parameter for the learning rate scheduler.

Implementation Details
----------------------

The HPO process requires three main components: a search space definition, a dataset builder, and a model builder.

1. Configuration Space
~~~~~~~~~~~~~~~~~~~~~~

We use ``ray.tune`` to define the distributions for our hyper-parameters. This dictionary tells the optimizer which values are allowed and how to sample them.

.. code-block:: python

    config_space = {
        # Architecture choices
        "branch_depth": tune.choice([2, 3, 4]),
        "trunk_depth": tune.choice([2, 3, 4]),
        "network_width": tune.choice([64, 128, 256]),
        "activation": tune.choice(["tanh", "relu", "sigmoid", "silu"]),
        
        # Optimizer parameters (continuous)
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        
        # Scheduler parameters
        "scheduler_step": tune.choice([5, 10, 15]),
        "scheduler_gamma": tune.uniform(0.9, 0.99),

        # Fixed parameters (not tuned)
        "batch_size": 32,
        "training_samples": 500,
        "problem_dim": 2, 
    }

2. Dataset Builder
~~~~~~~~~~~~~~~~~~

The ``dataset_builder`` is responsible for generating the training and validation data for each trial. To optimize the network correctly using a relative L2 loss, we need supervised targets. For the advection equation :math:`u_y + u_x = 0`, the analytical solution is :math:`u(x, y) = v(x - y)`, where :math:`v(x)` is the initial condition at :math:`y=0`. We use ``scipy.interpolate.interp1d`` to shift the discrete GRF branch inputs and compute the exact targets.

.. code-block:: python

    from scipy.interpolate import interp1d

    def generate_advection_data(num_functions, num_eval_points=50, num_trunk_points=1000):
        # 1. Generate Branch Inputs (Initial conditions at y=0)
        func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
        # Use endpoint=False for periodic domain to avoid duplicate 0.0/1.0
        eval_pts = np.linspace(0, 1, num=num_eval_points, endpoint=False)[:, None]
        features = func_space.random(num_functions)
        # Add a constant offset to ensure the norm is never zero (avoids division by zero in relative loss)
        branch_inputs = func_space.eval_batch(features, eval_pts) + 1.0
        
        # 2. Generate Trunk Inputs (Random points in the domain)
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        trunk_inputs = geom.uniform_points(num_trunk_points)
        num_actual_trunk_points = trunk_inputs.shape[0]
        
        # 3. Generate Targets (Analytical solution u(x, y) = v((x - y) mod 1))
        x_coords, y_coords = trunk_inputs[:, 0], trunk_inputs[:, 1]
        shifted_x = (x_coords - y_coords) % 1.0
        
        # Initialize targets based on actual points to avoid broadcasting errors
        targets = np.zeros((num_functions, num_actual_trunk_points))
        eval_pts_flat = eval_pts.flatten()
        
        # Append the first point to the end at x=1.0 to close the periodic loop
        x_vals = np.append(eval_pts_flat, 1.0)
        
        for i in range(num_functions):
            v_vals = np.append(branch_inputs[i], branch_inputs[i][0])
            interpolator = interp1d(x_vals, v_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")
            targets[i] = interpolator(shifted_x)
            
        return branch_inputs, trunk_inputs, targets

    def dataset_builder(config):
        X_b_train, X_t_train, y_train = generate_advection_data(config["training_samples"])
        X_b_test, X_t_test, y_test = generate_advection_data(100)

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

3. Model Builder
~~~~~~~~~~~~~~~~

The ``model_builder`` constructs the DeepONet model. It dynamically adjusts all the hyper-parameters based on the current configuration trial.

.. code-block:: python

    def model_builder(config):
        p = config["network_width"]
        model = dde.nn.DeepONetCartesianProd(
            [50] + [p] * config["branch_depth"] + [p],
            [5] + [p] * config["trunk_depth"] + [p],
            config["activation"], 
            "Glorot normal"
        )

        # Apply a periodic feature transform to embed domain knowledge
        def periodic(x):
            xt, yt = x[:, :1] * 2 * np.pi, x[:, 1:]
            return torch.cat([
                torch.cos(xt), torch.sin(xt), 
                torch.cos(2 * xt), torch.sin(2 * xt), yt
            ], 1)

        model.apply_feature_transform(periodic)
        return model

4. Loss Function
~~~~~~~~~~~~~~~~

For operator learning, we typically use the **Relative L2 Loss** (``LprelLoss``). This loss function is scale-invariant, making it ideal for comparing performance across different PDE scales and configurations during the HPO process.

.. code-block:: python

    loss_fn = LprelLoss(p=2, size_mean=True)

Advanced HPO Configuration
--------------------------

The ``tune_hyperparameters`` function orchestrates the search. Here is a detailed breakdown of its execution parameters:

.. list-table:: tune_hyperparameters Parameters
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``config_space``
     - The dictionary defining the search distributions.
   * - ``model_builder``
     - Function to instantiate the PyTorch model for each trial.
   * - ``dataset_builder``
     - Function to provide the data loaders.
   * - ``loss_fn``
     - The metric to be minimized (objective function).
   * - ``num_samples``
     - Total number of trials (combinations of hyper-parameters) to test.
   * - ``max_epochs``
     - The budget for training each trial.
   * - ``grace_period``
     - Number of epochs before a trial can be stopped by the ASHA scheduler.
   * - ``reduction_factor``
     - Controls the pruning aggressiveness of the ASHA scheduler.
   * - ``runs_per_cpu``
     - Fraction of CPU resources assigned to each concurrent trial.
   * - ``runs_per_gpu``
     - Fraction of GPU resources assigned to each trial (set to 0.0 for CPU-only).
   * - ``default_hyper_params``
     - A list of initial "best guess" configurations to try first.

Running the Example
-------------------

To run this HPO demo, ensure you have ``HyperNOs`` and ``ray[tune]`` installed. You can execute the script directly from the command line:

.. code-block:: bash

    python examples/hyperparameter/advection_2d_hpo.py

The script will launch the Ray Tune dashboard where you can monitor the progress of each trial in real-time. Once finished, it will print the best configuration and the corresponding relative loss.

Complete Code
-------------------

.. code-block:: python

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
