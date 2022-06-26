Burgers equation 
================

Problem setup
--------------

We will solve a Burgers equation:

.. math:: \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2u}{\partial x^2}, \qquad x \in [-1, 1], \quad t \in [0, 1]

with the Dirichlet boundary conditions and initial conditions  

.. math:: u(-1,t)=u(1,t)=0, \quad u(x,0) = - \sin(\pi x).

The reference solution is `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz>`_.

Implementation
--------------

This description goes through the implementation of a solver for the above described Burgers equation step-by-step.

First, the DeepXDE and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    from deepxde.backend import tf

We begin by defining a computational geometry and time domain. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

Next, we express the PDE residual of the Burgers equation:

.. code-block:: python

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

The first argument to ``pde`` is 2-dimensional vector where the first component(``x[:,0]``) is :math:`x`-coordinate and the second componenet (``x[:,1]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x,t)`, but here we use ``y`` as the name of the variable.

Next, we consider the boundary/initial condition. ``on_boundary`` is chosen here to use the whole boundary of the computational domain in considered as the boundary condition. We include the ``geomtime`` space, time geometry created above and ``on_boundary`` as the BCs in the ``DirichletBC`` function of DeepXDE. We also define ``IC`` which is the inital condition for the burgers equation and we use the computational domain, initial function, and ``on_initial`` to specify the IC. 

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
    
Now, we have specified the geometry, PDE residual, and boundary/initial condition. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(geomtime, pde, [bc, ic], 
                            num_domain=2540, num_boundary=80, num_initial=160)    

The number 2540 is the number of training residual points sampled inside the domain, and the number 80 is the number of training points sampled on the boundary. We also include 160 initial residual points for the initial conditions.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:

.. code-block:: python

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    
   
We then train the model for 15000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=15000)
    
After we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()      

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Burgers.py
  :language: python
