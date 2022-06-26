Inverse problem for the Poisson equation with unknown forcing field
===================================================================
 
Problem setup
-------------
 
We will solve
 
.. math:: \frac{d^2u}{dx^2} = q(x), \quad x \in [-1, 1]
 
with the Dirichlet boundary conditions
 
.. math:: u(-1) = 0, \quad u(1) = 0
 
Here, both :math:`u(x)` and :math:`q(x)` are unknown. Furthermore, we have the measurement of :math:`u(x)` at 100 points.
 
The reference solution is :math:`u(x) = \sin(\pi x), \quad q(x) = -\pi^2 \sin(\pi x)`.
 
Implementation
--------------
 
This description goes through the implementation of a solver for the above described Poisson equation step-by-step.
 
First, the DeepXDE, Matplotlib, and NumPy (``np``) modules are imported:
 
.. code-block:: python
 
    import deepxde as dde
    import matplotlib.pyplot as plt
    import numpy as np
       
We also define a function to generate ``num`` equally spaced points from -1 to 1 to use as training data.
 
.. code-block:: python
   
    def gen_traindata(num):
        xvals = np.linspace(-1, 1, num).reshape(num, 1)
        uvals = np.sin(np.pi * xvals)
        return xvals, uvals
 
Now we begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows:
 
.. code-block:: python
   
    geom = dde.geometry.Interval(-1, 1)
   
Next, we express the PDE residual of the Poisson equation using the ``dde.grad.hessian`` function.
 
.. code-block:: python
 
    def pde(x, y):
        u, q = y[:, 0:1], y[:, 1:2]
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        return -du_xx + q
 
The first argument to ``pde`` is the network input, i.e., the x-coordinate. The second argument is the network output, i.e., the solution :math:`u, q`.
 
Next, we consider the boundary conditions. First, let us define the function ``sol`` that will be used to compute :math:`u(-1)` and :math:`u(1)`.
 
.. code-block:: python
 
    def sol(x):
        return np.sin(np.pi * x)
 
Notice that, as required, ``sol(-1) = sol(1) = 0``. Next, we define the boundary conditions using the built-in ``dde.DirichletBC`` function.
 
.. code-block:: python
   
    bc = dde.icbc.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)
   
Here, we pass in our computational geometry, the function ``sol`` to compute the boundary values, a function which returns ``True`` if a point is on a boundary and ``False`` otherwise, and the component axis on which the boundary is satisfied.
 
Now, we generate 100 points and assign the data to ``ob_x`` and ``ob_u``. We organize and assign the train data.
 
.. code-block:: python
 
    ob_x, ob_u = gen_traindata(100)
    observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)
 
Now that the problem is fully setup, we define the PDE as:
 
.. code-block:: python  
 
    data = dde.data.PDE(
        geom,
        pde,
        [bc, observe_u],
        num_domain=200,
        num_boundary=2,
        anchors=ob_x,
        num_test=1000,
    )
 
where ``num_domain`` is the number of points inside the domain, and ``num_boundary`` is the number of points on the boundary. ``anchors`` are extra points beyond ``num_domain`` and ``num_boundary`` used for training.
 
Next, we choose the networks. We use two networks, one to train for ``u(x)`` and the other to train for ``q(x)``. Here, we use two fully connected neural networks of depth 4 (i.e., 3 hidden layers) and width 20.
 
.. code-block:: python
 
    net = dde.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
   
Now that the PDE problem and network have been created, we build a ``Model`` and choose the optimizer and learning rate.
 
.. code-block:: python
 
    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, loss_weights=[1, 100, 1000])
 
We then train the model for 20000 iterations:
 
.. code-block:: python
 
    losshistory, train_state = model.train(iterations=20000)
 
Complete code
-------------
 
.. literalinclude:: ../../../examples/pinn_inverse/elliptic_inverse_field.py
  :language: python
