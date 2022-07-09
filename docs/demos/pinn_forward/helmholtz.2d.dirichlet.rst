Helmholtz equation over a 2D square domain
========================================================================

Problem setup
--------------

For a wavenumber :math:`k_0 = 2\pi n` with :math:`n = 2`, we will solve a Helmholtz equation:

.. math:: - u_{xx}-u_{yy} - k_0^2 u = f, \qquad  \Omega = [0,1]^2

with the Dirichlet boundary conditions

.. math:: u(x,y)=0, \qquad (x,y)\in \partial \Omega

and a source term :math:`f(x,y) = k_0^2 \sin(k_0 x)\sin(k_0 y)`.

Remark that the exact solution reads:

.. math:: u(x,y)= \sin(k_0 x)\sin(k_0 y)

This example is the Dirichlet boundary condition conterpart to `this Dolfinx tutorial <https://github.com/FEniCS/dolfinx/blob/main/python/demo/helmholtz2D/demo_helmholtz_2d.py>`_. One can refer to Ihlenburg\'s book \"Finite Element Analysis of Acoustic Scattering\" p138-139 for more details.

Implementation
--------------

This description goes through the implementation of a solver for the above described Helmholtz equation step-by-step.

First, the DeepXDE and Numpy modules are imported:

.. code-block:: python

  import deepxde as dde
  import numpy as np

We begin by defining the general parameters for the problem. We use a collocation points density of 10 (resp. 30) points per wavelength for the training (resp. testing) data along each direction.
This code allows to use both soft and hard boundary conditions. 

.. code-block:: python

  n = 2
  precision_train = 10
  precision_test = 30
  hard_constraint = True
  weights = 100  # if hard_constraint == False

The PINN will be trained over 5000 iterations. We define the learning rate, the number of dense layers and nodes, and the activation function. Moreover, we import the sine function.

.. code-block:: python

  iterations = 5000
  parameters = [1e-3, 3, 150, "sin"]

  # Define sine function
  if dde.backend.backend_name == "pytorch":
      sin = dde.backend.pytorch.sin
  else:
      from deepxde.backend import tf

      sin = tf.sin
      
  learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

Next, we express the PDE residual of the Helmholtz equation:

.. code-block:: python

  def pde(x, y):
      dy_xx = dde.grad.hessian(y, x, i=0, j=0)
      dy_yy = dde.grad.hessian(y, x, i=1, j=1)

      f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
      return -dy_xx - dy_yy - k0 ** 2 * y - f


The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate and :math:`y`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we introduce the exact solution and the Dirichlet boundary condition. 
If boundary conditions are enforced in a hard fashion, we apply the following transformation to the neural network:

.. math:: \hat{u}(x,y) = x (x-1) y (y-1) \mathcal{N}(x,y)

.. code-block:: python

  def func(x):
      return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


  def transform(x, y):
      res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
      return res * y

  def boundary(_, on_boundary):
      return on_boundary

Now, we define the geometry and evaluate the number of training and test random collocation points. The values allow to obtain collocation points density of 10 (resp. 30) points per wavelength along each direction.
For soft boundary conditions, we define the boundary and the Dirichlet boundary conditions. 

.. code-block:: python

  geom = dde.geometry.Rectangle([0, 0], [1, 1])
  k0 = 2 * np.pi * n
  wave_len = 1 / n

  hx_train = wave_len / precision_train
  nx_train = int(1 / hx_train)

  hx_test = wave_len / precision_test
  nx_test = int(1 / hx_test)

  if hard_constraint == True:
      bc = []
  else:
      bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)


Next, we generate the training and testing points.

.. code-block:: python

  data = dde.data.PDE(
      geom,
      pde,
      bc,
      num_domain=nx_train ** 2,
      num_boundary=4 * nx_train,
      solution=func,
      num_test=nx_test ** 2,
  )

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 150. Besides, we choose sin as activation function and Glorot uniform as initializer :

.. code-block:: python

  net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
  )

For the hard constraint case, we apply the transform to enforce the boundary conditions:

.. code-block:: python

  if hard_constraint == True:
      net.apply_output_transform(transform)


Now, we have the PDE problem and the network. We build a ``Model`` and define the optimizer and learning rate. When soft constraints are applied, we apply a weight to the boundary term to improve convergence for the ADAM optimizer:

.. code-block:: python

  model = dde.Model(data, net)

  if hard_constraint == True:
      model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
  else:
      loss_weights = [1, weights]
      model.compile(
          "adam",
          lr=learning_rate,
          metrics=["l2 relative error"],
          loss_weights=loss_weights,
      )

We first train the model for 5000 iterations with Adam optimizer:

.. code-block:: python

    losshistory, train_state = model.train(iterations=iterations)
    
Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Helmholtz_Dirichlet_2d.py
  :language: python
