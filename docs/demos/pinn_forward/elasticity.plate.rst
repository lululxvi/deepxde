Linear elastostatic equation over a 2D square domain
==========

Problem setup
--------------

We will solve a 2D linear elasticity solid mechanic problem:

.. math:: \frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + f_x= 0, \quad 
          \frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + f_y= 0, 

.. math:: x \in [0, 1], \quad y \in [0, 1],

where the linear elastic constitutive model is defined as

.. math:: \sigma_{xx} = (\lambda + 2\mu)\epsilon_{xx} + \lambda\epsilon_{yy}, \quad
	  \sigma_{yy} = (\lambda + 2\mu)\epsilon_{yy} + \lambda\epsilon_{xx}, \quad
	  \sigma_{xy} =  2\mu\epsilon_{xy},

with kinematic relation

.. math:: \epsilon_{xx} = \frac{\partial u_{x}}{\partial x}, \quad
	  \epsilon_{yy} = \frac{\partial u_{y}}{\partial y}, \quad
          \epsilon_{xy} = \frac{1}{2}(\frac{\partial u_{x}}{\partial y} + \frac{\partial u_{y}}{\partial x}).

The 2D square domain is subjected to body forces: 

.. math:: 

   f_x & = \lambda[4\pi^2\cos(2\pi x)\sin(\pi y) - \pi\cos(\pi x)Qy^3] \\
       & + \mu[9\pi^2\cos(2\pi x)\sin(\pi y) - \pi\cos(\pi x)Qy^3], \\
   f_y & = \lambda[-3\sin(\pi x)Qy^2 + 2\pi^2\sin(2\pi x)\cos(\pi y)] \\
       & + \mu[-6 \sin(\pi x)Qy^2 + 2 \pi^2\sin(2\pi x)\cos(\pi y) + \pi^2\sin(\pi x)Qy^4/4],   

with displacement boundary conditions

.. math:: u_x(x, 0) = u_x(x, 1) = 0,
.. math:: u_y(0, y) = u_y(1, y) = u_y(x, 0) = 0, 

and traction boundary conditions

.. math:: \sigma_{xx}(0, y)=0, \quad \sigma_{xx}(1, y)=0, \quad \sigma_{yy}(x, 1)=(\lambda + 2\mu)Q\sin(\pi x). 

We set parameters :math:`\lambda = 1,` :math:`\mu = 0.5,` and :math:`Q = 4.`

The exact solution is :math:`u_x(x, y) = \cos(2\pi x)\sin(\pi y)` and :math:`u_y(x, y) = \sin(\pi x)Qy^4/4.`

Implementation
--------------

This description goes through the implementation of a solver for the above described linear elasticity problem step-by-step.

First, the DeepXDE, NumPy (``np``) and PyTorch (``torch``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    import torch


We begin by defining the general parameters for the problem.

.. code-block:: python

    lmbd = 1.0
    mu = 0.5
    Q = 4.0
    pi = torch.pi

Next, we define a computational geometry. We can use a built-in class ``Rectangle`` as follows

.. code-block:: python

    geom = dde.geometry.Rectangle([0, 0], [1, 1])


Next, we consider the left, right, top, and bottom boundary conditions. Two boundary conditions are applied on the left boundary. The location of the boundary conditions is defined by a simple Python function. The function should return ``True`` for those points satisfying :math:`x[0]=0` and ``False`` otherwise (Note that because of rounding-off errors, it is often wise to use ``dde.utils.isclose`` to test whether two floating point values are equivalent). In the functions below, the argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. Then a boolean ``on_boundary`` is used as the second argument. If the point ``x`` (the first argument) is on the boundary of the geometry, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``.

.. code-block:: python

    def boundary_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

Two boundary conditions are applied on the right boundary. Similar to ``boundary_left``, the location of these two boundary condition is defined in a similar way that the function should return ``True`` for those points satisfying :math:`x[0]=1` and ``False`` otherwise.

.. code-block:: python

    def boundary_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1.0)

Two boundary conditions are applied on the top boundary. 

.. code-block:: python

    def boundary_top(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 1.0)

Two boundary conditions are applied on the bottom boundary. 

.. code-block:: python

    def boundary_bottom(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 0.0)

Next, for better comparsion, we define a function which is the exact solution to the problem.

.. code-block:: python

    def func(x):
        ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
        uy = np.sin(pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

        E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
        E_yy = np.sin(pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
        E_xy = 0.5 * (
            np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
            + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
        )

        Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        Sxy = 2 * E_xy * mu

        return np.hstack((ux, uy, Sxx, Syy, Sxy))

The displacement boundary conditions on the boundary are defined as follows.

.. code-block:: python

    ux_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=0)
    ux_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=0)
    uy_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=1)
    uy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
    uy_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=1)

Similarly, the traction boundary conditions are defined as,

.. code-block:: python

    sxx_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=2)
    sxx_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=2)
    syy_top_bc = dde.icbc.DirichletBC(
        geom,
        lambda x: (2 * mu + lmbd) * Q * np.sin(pi * x[:, 0:1]),
        boundary_top,
        component=3,
    )

Next, we define the body forces

.. code-block:: python

    def fx(x):
        return (
            -lmbd
            * (
                4 * pi**2 * torch.cos(2 * pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
                - Q * x[:, 1:2] ** 3 * pi * torch.cos(pi * x[:, 0:1])
            )
            - mu
            * (
                pi**2 * torch.cos(2 * pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
                - Q * x[:, 1:2] ** 3 * pi * torch.cos(pi * x[:, 0:1])
            )
            - 8 * mu * pi**2 * torch.cos(2 * pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
        )


    def fy(x):
        return (
            lmbd
            * (
                3 * Q * x[:, 1:2] ** 2 * torch.sin(pi * x[:, 0:1])
                - 2 * pi**2 * torch.cos(pi * x[:, 1:2]) * torch.sin(2 * pi * x[:, 0:1])
            )
            - mu
            * (
                2 * pi**2 * torch.cos(pi * x[:, 1:2]) * torch.sin(2 * pi * x[:, 0:1])
                + (Q * x[:, 1:2] ** 4 * pi**2 * torch.sin(pi * x[:, 0:1])) / 4
            )
            + 6 * Q * mu * x[:, 1:2] ** 2 * torch.sin(pi * x[:, 0:1])
        )

Next, we express the PDE residuals of the momentum equation and the constitutive law. 

.. code-block:: python

    def pde(x, f):
        E_xx = dde.grad.jacobian(f, x, i=0, j=0)
        E_yy = dde.grad.jacobian(f, x, i=1, j=1)
        E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

        S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        S_xy = E_xy * 2 * mu

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
        Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)

        momentum_x = Sxx_x + Sxy_y - fx(x)
        momentum_y = Sxy_x + Syy_y - fy(x)

        stress_x = S_xx - f[:, 2:3]
        stress_y = S_yy - f[:, 3:4]
        stress_xy = S_xy - f[:, 4:5]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

The first argument to ``pde`` is the network input, i.e., the :math:`x` and `y` -coordinate. The second argument is the network output, i.e., the solution :math:`u_x(x, y)`, but here we use ``f`` as the name of the variable.


Now, we have specified the geometry, PDE residuals, and boundary conditions. We then define the PDE problem as

.. code-block:: python
 
    data = dde.data.PDE(
        geom,
        pde,
        [
            ux_top_bc,
            ux_bottom_bc,
            uy_left_bc,
            uy_bottom_bc,
            uy_right_bc,
            sxx_left_bc,
            sxx_right_bc,
            syy_top_bc,
        ],
        num_domain=500,
        num_boundary=500,
        solution=func,
        num_test=100,
        )

The number 500 is the number of training residual points sampled inside the domain, and the number 500 is the number of training points sampled on the boundary. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 100 residual points for testing the PDE residual.

Next, we choose the network. Here, we use 5 fully connected independent neural networks of depth 5 (i.e., 4 hidden layers) and width 40:

.. code-block:: python

    layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.PFNN(layers, activation, initializer)

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)

We then train the model for 5000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(epochs=5000)

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/elasticity_plate.py
  :language: python
