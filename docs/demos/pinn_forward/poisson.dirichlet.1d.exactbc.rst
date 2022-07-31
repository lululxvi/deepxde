Poisson equation in 1D with hard boundary conditions
================

Problem setup
--------------
We will solve a Poisson equation:

.. math:: -\Delta u = \sum_{i=1}^4 i\sin(ix) + 8\sin(8x), \qquad x \in [0, \pi]

with the Dirichlet boundary conditions

.. math:: u(x = 0) = 0, u(x = \pi) = \pi.

The exact solution is :math:`u(x) = x + \sum_{i=1}^4 \frac{\sin(ix)}{i} + \frac{\sin(8x)}{8}`.

Implementation
--------------
This description goes through the implementation of a solver for the above described Poisson equation step-by-step.

First, the DeepXDE, NumPy (``np``), and TensorFlow (``tf``) modules are imported.

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf
    
We begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows.

.. code-block:: python
    
    geom = dde.geometry.Interval(0, np.pi)
    
Next, we express the PDE residual of the Poisson equation.

.. code-block:: python

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        summation = sum([i * tf.sin(i * x) for i in range(1, 5)])
        return -dy_xx - summation - 8 * tf.sin(8 * x)
        
The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, the reference solution ``func`` is defined as the following.

.. code-block:: python

    def func(x):
        summation = sum([np.sin(i * x) / i for i in range(1, 5)])
        return x + summation + np.sin(8 * x) / 8
        
Now, we have specified the geometry and PDE residual. We then define the PDE problem as the following.

.. code-block:: python

    data = dde.data.PDE(geom, pde, [], num_domain=64, solution=func, num_test=400)
    
The number 64 is the number of training residual points sampled inside the domain. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don’t have a reference solution. We use 400 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50.

.. code-block:: python

    layer_size = [1] + [50] * 3 + [1]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = dde.nn.FNN(layer_size, activation, initializer)
    
Next, we define the transformation of the output and apply it to the network. When :math:`x=0`, the boundary condition :math:`u(x = 0) = 0` is satisfied. When :math:`x=\pi`, the boundary condition :math:`u(x = \pi) = \pi` is satisfied. This demonstrates that both ends of the boundary constraint are hard conditions.

.. code-block:: python

    def  output_transform(x, y):
        return x * (np.pi - x) * y + x 

    net.apply_output_transform(output_transform)
    
Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate. We also implement a learning rate decay to reduce overfitting of the model.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-4, decay = ("inverse time", 1000, 0.3), metrics=["l2 relative error"])

We also compute the :math:`L^2` relative error as a metric during training.

We then train the model for 30000 iterations.

.. code-block:: python

    losshistory, train_state = model.train(iterations=30000)
    
Finally, we save and plot the best trained result and the loss history of the model.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
Complete code
--------------
.. literalinclude:: ../../../examples/pinn_forward/Poisson_Dirichlet_1d_exactBC.py
  :language: python
