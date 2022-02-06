Inverse Problem for the Poisson Equation With Unknown Forcing Field
=====================================

Problem setup
--------------

We will solve

.. math:: \frac{d^2u}{dx^2} = q(x), \quad x \in [-1, 1]

with the Dirichlet boundary conditions

.. math:: u(-1) = 0, \quad u(1) = 0

This PDE is particularly interesting because both :math:`u(x)` and :math:`q(x)` are unknown.

The reference solution is :math:`u(x) = \sin(\pi x), \quad q(x) = -\pi^2 \sin(\pi x)`.

Implementation
--------------

First, the DeepXDE, Matplotlib, and NumPy (``np``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import matplotlib.pyplot as plt
    import numpy as np
        
We also define a function to generate (``num``) equally spaced points from :math:`-1` to :math:`1` to use as training data.

.. code-block:: python
    
    def gen_traindata(num):
        # generate num equally-spaced points from -1 to 1
        xvals = np.linspace(-1, 1, num).reshape(num, 1)
        uvals = np.sin(np.pi * xvals)
        return xvals, uvals

Now we begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows

.. code-block:: python
    
    geom = dde.geometry.Interval(-1, 1)
    
Next, we express the PDE residual of the Poisson equation using the ``dde.grad.hessian`` function.

.. code-block:: python

    def pde(x, y):
        u, q = y[:, 0:1], y[:, 1:2]
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        return -du_xx + q

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u, q`.

Next, we consider the boundary conditions. First, let us define the function ``sol`` that will be used to compute :math:`u(0)` and :math:`u(1)`.

.. code-block:: python

    def sol(x):
        return np.sin(np.pi * x ** 2)

Notice that, as required, ``sol(-1) = sol(1) = 0``. Next, we define the boundary conditions using the built-in ``dde.DirichletBC`` function.

.. code-block:: python
    
    bc = dde.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)
    
Here, we pass in our computational geometry, the function ``sol`` to compute the boundary values, a function which returns ``True`` if a point is on a boundary and ``False`` otherwise, and the component axis on which the boundary is satisfied.

Now, we generate :math:``100`` points and assign the data to ``ob_x`` and ``ob_u``. We organize and assign the train data.

.. code-block:: python

    ob_x, ob_u = gen_traindata(100)
    observe_u = dde.PointSetBC(ob_x, ob_u, component=0)
  
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

Where ``num_domain`` is the number of points inside the domain, and ``num_boundary`` is the number of points on the boundary. ``anchors`` are extra points beyond ``num_domain`` and ``num_boundary`` used for training. 

Next, we choose the networks. We use two networks, one to train for :math:``u(x)`` and the other to train for ``q(x)``. Here, we use two fully connected neural networks of depth 4 (i.e., 3 hidden layers) and width 20.

.. code-block:: python

    net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
    
Now that the PDE problem and network have been created, we build a ``Model`` and choose the optimizer and learning rate.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, loss_weights=[1, 100, 1000])

We then train the model for 60000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(epochs=20000)

We can now view the results

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    x = geom.uniform_points(500)
    yhat = model.predict(x)
    uhat, qhat = yhat[:, 0:1], yhat[:, 1:2]

    utrue = np.sin(np.pi * x)
    print("l2 relative error for u: " + str(dde.metrics.l2_relative_error(utrue, uhat)))
    plt.figure()
    plt.plot(x, utrue, "-", label="u_true")
    plt.plot(x, uhat, "--", label="u_NN")
    plt.legend()

    qtrue = -np.pi ** 2 * np.sin(np.pi * x)
    print("l2 relative error for q: " + str(dde.metrics.l2_relative_error(qtrue, qhat)))
    plt.figure()
    plt.plot(x, qtrue, "-", label="q_true")
    plt.plot(x, qhat, "--", label="q_NN")
    plt.legend()

    plt.show()

Complete code
--------------

`Jupyter notebook <https://github.com/lululxvi/deepxde/blob/master/examples/elliptic_inverse_field.py>`_

.. literalinclude:: ../../examples/elliptic_inverse_field.py
  :language: python
