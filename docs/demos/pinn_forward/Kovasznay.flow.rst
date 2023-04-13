Kovasznay flow
================

Problem setup
--------------

We will solve the Kovasznay flow equation on :math:`\Omega = [0, 1]^2`:

.. math:: u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}= -\frac{\partial p}{\partial x} + \frac{1}{Re}(\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}), 

.. math:: u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}= -\frac{\partial p}{\partial y} + \frac{1}{Re}(\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2}),  

with the Dirichlet boundary conditions

.. math:: u(x,y)=0, \qquad (x,y)\in \partial \Omega

The reference solution is :math:`u = 1 - e^{\lambda x} \cos(2\pi y)`, :math:`v = \frac{\lambda}{2\pi}e^{\lambda x} \sin(2\pi x)`, :math:`p =\frac{1}{2}(1 - e^{2\lambda x})`, where :math:`\lambda = \frac{1}{2\nu}-\sqrt{\frac{1}{4\nu^2}+4\pi^2}`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Kovasznay flow step-by-step.

First, the DeepXDE and Numpy modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining the parameters of the equation. :math:`\lambda` is defined as l below.

.. code-block:: python

    Re = 20
    nu = 1 / Re
    l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)

Next, we express the PDE residual of the Kovasznay flow equation in terms of the x-direction, y-direction and continuity equations.

.. code-block:: python

    def pde(x, u):
        u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
        u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

        v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

        p_x = dde.grad.jacobian(u, x, i=2, j=0)
        p_y = dde.grad.jacobian(u, x, i=2, j=1)

        momentum_x = (
            u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
        )
        momentum_y = (
            u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
        )
        continuity = u_vel_x + v_vel_y

        return [momentum_x, momentum_y, continuity]

The first argument to ``pde`` is the network input, i.e. the x and y coordinates. The second argument is the network output ``u`` which is comprised of the 3 different output solutions i.e., velocity u, velocity v, and pressure p. 

Next, the exact solution of the Kovasznay flow is introduced

.. code-block:: python

    def u_func(x):
        return 1 - np.exp(l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

    def v_func(x):
        return l / (2 * np.pi) * np.exp(l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

    def p_func(x):
        return 1 / 2 * (1 - np.exp(2 * l * x[:, 0:1]))

Next, we consider the boundary condition. ``on_boundary`` is chosen here to use the whole boundary of the computational domain as the boundary condition. We include ``on_boundary`` as the BCs in the ``DirichletBC`` function of DeepXDE. 

 .. code-block:: python

    def boundary_outflow(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)
        
    spatial_domain = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

    boundary_condition_u = dde.icbc.DirichletBC(
        spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0
    )
    boundary_condition_v = dde.icbc.DirichletBC(
        spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1
    )
    boundary_condition_right_p = dde.icbc.DirichletBC(
        spatial_domain, p_func, boundary_outflow, component=2
    )

    
Now, we have specified the geometry, PDE residual, and boundary condition. We then define the ``PDE`` problem as

.. code-block:: python

    data = dde.data.PDE(
        spatial_domain,
        pde,
        [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
        num_domain=2601,
        num_boundary=400,
        num_test=100000,
    )
    
The training residual points imside the domain is 2601, and the number of training points sampled on the boundary is 400. 100000 test points were used in the ``PDE``.

Next, we choose the network. We use a fully connected neural network of 4 hidden layers, 3 outputs and width 50

.. code-block:: python

    net = dde.nn.FNN([2] + 4 * [50] + [3], "tanh", "Glorot normal")

The PDE and the network have now been defined. Next, we build a ``Model`` and choose the optimizer and learning rate.

.. code-block:: python

    model = dde.Model(data, net)
    
    model.compile("adam", lr=1e-3)
    model.train(iterations=30000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
 
We then train the model for 30000 iterations. After we train the network using ``Adam``, we continue to train the network using L-BFGS to achieve a smaller loss.

The next step is to define a spatial domain with the same number of random points 100000  and use the model created to predict the output.

.. code-block:: python

    X = spatial_domain.random_points(100000)
    output = model.predict(X)
    u_pred = output[:, 0]
    v_pred = output[:, 1]
    p_pred = output[:, 2]     

.. code-block:: python

    u_exact = u_func(X).reshape(-1)
    v_exact = v_func(X).reshape(-1)
    p_exact = p_func(X).reshape(-1)

Next, we compare the predicted output to the exact output and calculate the loss.

.. code-block:: python

   f = model.predict(X, operator=pde)

    l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
    l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
    l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
    residual = np.mean(np.absolute(f))

    print("Mean residual:", residual)
    print("L2 relative error in u:", l2_difference_u)
    print("L2 relative error in v:", l2_difference_v)
    print("L2 relative error in p:", l2_difference_p) 
    
Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Kovasznay_flow.py
  :language: python
