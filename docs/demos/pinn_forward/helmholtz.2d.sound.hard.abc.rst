Helmholtz sound-hard scattering problem with absorbing boundary conditions
=====================================================================================

This example allows to solve the 2d Helmholtz sound-hard (scattering) problem by a R-radius circle. It is useful to understand how to:

* Solve PDEs with complex values, i.e. with :math:`u = u_0 + \imath u_1`
* Handle Robin boundary conditions for PDEs with complex values
* Truncate unbounded domains via absorbing boundary conditions

Problem setup
--------------

For a wavenumber :math:`k_0= 2`, we will solve a sound-hard scattering problem for :math:`u = u^{scat} =  u_0 + \imath u_1:``

.. math:: - u_{xx}-u_{yy} - k_0^2 u = 0, \qquad  \Omega = \overline{B(0,R)}^c

with the Neumann boundary conditions

.. math:: \gamma_1 u :=\nabla u \cdot n =- u^{inc}, \qquad (x,y)\in \Gamma^{in} : = \partial (B(0,R))

with :math:`n`, and suitable radiation conditions at infinity. The analytical formula for the scattered field is given by Bessel function (refer to `waves-fenicsx <https://github.com/samuelpgroth/waves-fenicsx/tree/master/frequency>`_).

We decide to truncate the domain and we approximate the radiation conditions by absorbing boundary conditions (ABCs), on a ``length`` square :math:`\Gamma^{out}`. Refer to this recent `study <https://arxiv.org/pdf/2101.02154.pdf>`_ for the wavenumber analysis error.

Projection to the real and imaginary axes for :math:`u = u_0+ \imath * u_1` leads to:

.. math:: - u_{0,xx}-u_{0,yy} - k_0^2 u_0 = 0 \qquad \text{in}\qquad \Omega \cap D^{out}

and

.. math:: - u_{1,xx}-u_{1,yy} - k_0^2 u_1 = 0 \qquad\text{in}\qquad  \Omega \cap D^{out}

The boundary conditions read:

.. math:: \gamma_1 u =  - \gamma_1 u^{inc} \qquad \text{on} \qquad\Gamma^{in}
.. math:: \gamma_1 u - \imath k_0 \gamma_0 = 0 \qquad \text{on} \qquad \Gamma^{out}.

Absorbing boundary conditions rewrite:

.. math:: \gamma_1 [u_0+ \imath  u_1] - \imath k_0 [u_0 + \imath u_1] = 0 \qquad \text{on} \qquad \Gamma^{out}

i.e.

.. math:: \gamma_1 u_0 + k_0 u_1 = 0 \qquad\text{on}\qquad \Gamma^{out}
.. math:: \gamma_1 u_1 - k_0 u_0 = 0 \qquad \text{on}\qquad\Gamma^{out}.


This example is inspired by `this Dolfinx tutorial <https://github.com/samuelpgroth/waves-fenicsx/tree/master/frequency>`_.

Implementation
--------------

This description goes through the implementation of a solver for the above scattering problem step-by-step.

First, the DeepXDE and required modules are imported:

.. code-block:: python

  import deepxde as dde
  import numpy as np
  import scipy
  from scipy.special import jv, hankel1


Then, we begin by defining the general parameters for the problem. The PINN will be trained over 5000 iterations, we also define the learning rate, the number of dense layers and nodes, and the activation function.

.. code-block:: python

  weights = 1
  epochs = 10000
  learning_rate = 1e-3
  num_dense_layers = 3
  num_dense_nodes = 350
  activation = "sin"

We set the physical parameters for the problem.

.. code-block:: python

  k0 = 2
  wave_len = 2 * np.pi / k0
  length = 2 * np.pi
  R = np.pi / 4
  n_wave = 20
  h_elem = wave_len / n_wave
  nx = int(length / h_elem)


We define the geometry (inner and outer domains):

.. code-block:: python

  outer = dde.geometry.Rectangle([-length / 2, -length / 2], [length / 2, length / 2])
  inner = dde.geometry.Disk([0, 0], R)

  geom = outer - inner

We introduce the analytic solution for the sound-hard scattering problem:

.. code-block:: python

  def sound_hard_circle_deepxde(k0, a, points):
    
    fem_xx = points[:, 0:1]
    fem_xy = points[:, 1:2]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    n_terms = np.int(30 + (k0 * a)**1.01)
    u_sc = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n-1, k0*a) - n/(k0*a) * jv(n, k0*a)
        hankel_deriv = n/(k0*a)*hankel1(n, k0*a) - hankel1(n+1, k0*a)
        u_sc += (-(1j)**(n) * (bessel_deriv/hankel_deriv) * hankel1(n, k0*r) * \
            np.exp(1j*n*theta)).ravel()
    return u_sc

Next, we express the PDE residual of the Helmholtz equation:

.. code-block:: python

  def pde(x, y):
      y0, y1 = y[:, 0:1], y[:, 1:2]
      
      y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
      y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

      y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
      y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

      return [-y0_xx - y0_yy - k0 ** 2 * y0,
              -y1_xx - y1_yy - k0 ** 2 * y1]


Then, we introduce the exact solution and both Neumann and Robin boundary conditions:

.. code-block:: python

  def sol(x):
      result = sound_hard_circle_deepxde(k0, R, x).reshape((x.shape[0],1))
      real = np.real(result)
      imag = np.imag(result)
      return np.hstack((real, imag))

  def boundary(x, on_boundary):
      return on_boundary

  def boundary_outer(x, on_boundary):
      return on_boundary and outer.on_boundary(x)

  def boundary_inner(x, on_boundary):
      return on_boundary and inner.on_boundary(x)

  def func0_inner(x):
      normal = -inner.boundary_normal(x)
      g = 1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * normal[:, 0:1]
      return np.real(-g)

  def func1_inner(x):
      normal = -inner.boundary_normal(x)
      g = 1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * normal[:, 0:1]
      return np.imag(-g)

  def func0_outer(x, y):
      normal = outer.boundary_normal(x)
      result = - k0 * y[:, 1:2]
      return result

  def func1_outer(x, y):
      normal = outer.boundary_normal(x)
      result =  k0 * y[:, 0:1]
      return result
    
  bc0_inner = dde.NeumannBC(geom, func0_inner, boundary_inner, component = 0)
  bc1_inner = dde.NeumannBC(geom, func1_inner, boundary_inner, component = 1)

  bc0_outer = dde.RobinBC(geom, func0_outer, boundary_outer, component = 0)
  bc1_outer = dde.RobinBC(geom, func1_outer, boundary_outer, component = 1)

  bcs = [bc0_inner, bc1_inner, bc0_outer, bc1_outer]

 
Next, we define the weights for the loss function and generate the training and testing points.

.. code-block:: python

  loss_weights = [1, 1, weights, weights, weights, weights]
  data = dde.data.PDE(
      geom,
      pde,
      bcs,
      num_domain=nx**2,
      num_boundary=8 * nx,
      num_test=5 * nx **2,
      solution=sol
  )


Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50. Besides, we choose sin as activation function and Glorot uniform as initializer :

.. code-block:: python

  net = dde.maps.FNN(
      [2] + [num_dense_nodes] * num_dense_layers + [2], activation, "Glorot uniform"
  )

Now, we have the PDE problem and the network. We build a ``Model`` and define the optimizer and learning rate.

.. code-block:: python

  model.compile(
      "adam", lr=learning_rate, loss_weights=loss_weights , metrics=["l2 relative error"]
  )

We first train the model for 5000 iterations with Adam optimizer:

.. code-block:: python

    losshistory, train_state = model.train(epochs=epochs)


Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py
  :language: python
