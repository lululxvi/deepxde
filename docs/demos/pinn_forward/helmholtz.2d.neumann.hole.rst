Helmholtz equation over a 2D square domain with a hole
========================================================================

Problem setup
--------------

The purposes of this tutorial are the following: 

* Defining Neumann boundary conditions
* Working on a domain with a hole

The computational domain :math:`\Omega` is a :math:`L`-length square, :math:`L=1`, to which we remove a :math:`R = 1/4` radius circle.

For a wavenumber :math:`k_0 = 2 \pi n` with :math:`n = 1`, we solve a Helmholtz equation:

.. math:: - u_{xx}-u_{yy} - k_0^2 u = f \qquad  \text{in} \qquad \Omega 

with a source term :math:`f = k_0^2 \sin(k_0 x)\sin(k_0 y)`.

We set the exact solution as being: 

.. math:: u(x,y)= \sin(k_0 x)\sin(k_0 y)

and we remark that :math:`u(x,y)` solves the Helmholtz equation on :math:`\Omega`.

Next, we impose artificially Dirichlet boundary conditions on the outer boundary:

.. math:: u(x,y)|_{\Gamma_{outer}} = \sin(k_0 x)\sin(k_0 y) , \qquad (x,y) \in \Gamma_{outer}

In the same fashion, notice that for :math:`(x,y) \in \Gamma_{inner}` there holds that

.. math::
   :nowrap:

   \begin{align*}
   (\nabla u |_{\Gamma_{inner}}\cdot n)(x,y) &= [k_0 \cos(k_0 x)\sin(k_0 y), k_0\sin(k_0 x)\cos(k_0 y)]\cdot n\\
   &=  g(x,y)
   \end{align*}

with :math:`n` the normal exterior vector. Therefore, we set the following Neumann boundary conditions

.. math:: \nabla u| _{\Gamma_{inner}}\cdot n = g

Implementation
--------------

This description goes through the implementation of a solver for the above described Helmholtz equation step-by-step.

First, the DeepXDE, Numpy and Matplotlib modules are imported:

.. code-block:: python

  import deepxde as dde
  import matplotlib.pyplot as plt
  import numpy as np

We begin by defining the general parameters for the problem. We use a collocation points density of 15 (resp. 30) points per wavelength for the training (resp. testing) data along each direction. The PINN will be trained over 5000 epochs. We define the learning rate, the number of dense layers and nodes, and the activation function.

.. code-block:: python

  n = 1
  length = 1
  R = 1 / 4

  precision_train = 15
  precision_test = 30

  weight_inner = 10
  weight_outer = 100
  iterations = 5000
  learning_rate = 1e-3
  num_dense_layers = 3
  num_dense_nodes = 350
  activation = "sin"  

  k0 = 2 * np.pi * n
  wave_len = 1 / n

Next, we import the ``sin`` function and we express the PDE residual of the Helmholtz equation:

.. code-block:: python

  if dde.backend.backend_name == "pytorch":
      import torch
      sin = torch.sin
  elif dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
      from deepxde.backend import tf

      sin = tf.sin

  def pde(x, y):
      dy_xx = dde.grad.hessian(y, x, i=0, j=0)
      dy_yy = dde.grad.hessian(y, x, i=1, j=1)
      f = k0**2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
      return -dy_xx - dy_yy - k0**2 * y - f


The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate and :math:`y`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x,y)`, but here we use ``y`` as the name of the variable.

We introduce the exact solution and the inner (resp. outer) boundary.

.. code-block:: python

  def func(x):
      return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


  def boundary_outer(x, on_boundary):
      return on_boundary and outer.on_boundary(x)


  def boundary_inner(x, on_boundary):
      return on_boundary and inner.on_boundary(x)


We set the Neumann boundary conditions. The ``reduce_sum`` operation allows to evaluate the inner product over all collocation points. We use the ``normal = -inner.boundary_normal(x)`` in order to obtain normal vectors pointed toward the exterior of the computational domain.

.. code-block:: python

  def neumann(x):
    grad = np.array(
        [
            k0 * np.cos(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2]),
            k0 * np.sin(k0 * x[:, 0:1]) * np.cos(k0 * x[:, 1:2]),
        ]
    )

    normal = -inner.boundary_normal(x)
    normal = np.array([normal]).T
    result = np.sum(grad * normal, axis=0)
    return result
    
Now, we define the geometry and evaluate the number of training and test random collocation points. We define the boundary conditions.

.. code-block:: python

  outer = dde.geometry.Rectangle([-dim_x / 2.0, -dim_x / 2.0], [dim_x / 2.0, dim_x / 2.0])
  inner = dde.geometry.Disk([0, 0], R)

  geom = outer - inner

  hx_train = wave_len / precision_train
  nx_train = int(1 / hx_train)

  hx_test = wave_len / precision_test
  nx_test = int(1 / hx_test)

  bc_inner = dde.icbc.NeumannBC(geom, neumann, boundary_inner)
  bc_outer = dde.icbc.DirichletBC(geom, func, boundary_outer)


Next, we generate the data points, the net, the model, and the weights for the loss function:

.. code-block:: python

  data = dde.data.PDE(
      geom,
      pde,
      [bc_inner, bc_outer],
      num_domain=nx_train**2,
      num_boundary=16 * nx_train,
      solution=func,
      num_test=nx_test**2,
  )

  net = dde.nn.FNN(
      [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
  )

  model = dde.Model(data, net)

  loss_weights = [1, weight_inner, weight_outer]

We compile the model and train it for 5000 iterations with Adam optimizer:

.. code-block:: python

  model.compile(
    "adam", lr=learning_rate, metrics=["l2 relative error"], loss_weights=loss_weights
  )
  
  losshistory, train_state = model.train(iterations=iterations)
  
Now, we save the model, and plot the PINN and the solution over a square grid with 100 points per wavelength in each direction. We use masks to remove the points lying inside the :math:`R`-radius circle:

.. code-block:: python

  dde.saveplot(losshistory, train_state, issave=True, isplot=True)


  Nx = int(np.ceil(wave_len * 100))
  Ny = Nx

  # Grid points
  xmin, xmax, ymin, ymax = [-length / 2, length / 2, -length / 2, length / 2]
  plot_grid = np.mgrid[xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j]
  points = np.vstack(
      (plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))
  )

  points_2d = points[:2, :]
  u = model.predict(points[:2, :].T)
  u = u.reshape((Nx, Ny))

  ide = np.sqrt(points_2d[0, :] ** 2 + points_2d[1, :] ** 2) < R
  ide = ide.reshape((Nx, Nx))

  u_exact = func(points.T)
  u_exact = u_exact.reshape((Nx, Ny))
  diff = u_exact - u
  error = np.linalg.norm(diff) / np.linalg.norm(u_exact)
  print("Relative error = ", error)

  plt.rc("font", family="serif", size=22)

  fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(24, 12))

  matrix = np.fliplr(u).T
  matrix = np.ma.masked_where(ide, matrix)
  pcm = ax1.imshow(
      matrix,
      extent=[-length / 2, length / 2, -length / 2, length / 2],
      cmap=plt.cm.get_cmap("seismic"),
      interpolation="spline16",
      label="PINN",
  )

  fig.colorbar(pcm, ax=ax1)

  matrix = np.fliplr(u_exact).T
  matrix = np.ma.masked_where(ide, matrix)
  pcm = ax2.imshow(
      matrix,
      extent=[-length / 2, length / 2, -length / 2, length / 2],
      cmap=plt.cm.get_cmap("seismic"),
      interpolation="spline16",
      label="Exact",
  )

  ax1.set_title("PINNs")
  ax2.set_title("Exact")
  fig.colorbar(pcm, ax=ax2)

Finally, we represent the boundary normal vectors for the circle and save the plot:

.. code-block:: python

  p = inner.random_boundary_points(16 * nx_train)
  px, py = p.T
  nx, ny = inner.boundary_normal(p).T
  ax1.quiver(px, py, nx, ny)
  ax2.quiver(px, py, nx, ny)
  plt.savefig("plot_manufactured.pdf")

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Helmholtz_Neumann_2d_hole.py
  :language: python