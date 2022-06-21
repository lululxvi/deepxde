Allen-Cahn Equation
================

Problem Setup
--------------

We will solve an Allen-Cahn equation:

.. math:: \frac{\partial u}{\partial t} = d\frac{\partial^2u}{\partial x^2} + 5(u - u^3), \qquad x \in [-1, 1], \quad t \in [0, 1]

with initial and boundary conditions

.. math:: u(-1, t) = u(1, t) = -1, \quad u(x, 0) = x^2\cos(\pi x).

Implementation
--------------

This description goes through the implementation of a solver for the above described Allen-Cahn equation step-by-step.

First, the DeepXDE, NumPy (``np``), Scipy, and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from scipy.io import loadmat
    # Import tf if using backend tensorflow.compat.v1 or tensorflow
    from deepxde.backend import tf
    
We begin by defining a computational geometry and time domain. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime``.
    
.. code-block:: python
    
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    d = 0.001
        
Now, we express the PDE residual of the Allen-Cahn equation as follows

.. code-block:: python

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - d * dy_xx - 5 * (y - y**3)
        
The first argument to ``pde`` is a 2-dimensional vector where the first component(``x[:, 0]``) is :math:`x`-coordinate and the second component (``x[:, 1]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x, t)`, but here we use ``y`` as the name of the variable.

Next, we consider the initial conditions and boundary constraints, defining the transformation of the output as follows

.. code-block:: python

    def output_transform(x, y):
      return x[:, 0:1]**2 * tf.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y

Now, we have specified the geometry, PDE residual, boundary and initial conditions. We then define the TimePDE problem as

.. code-block:: python

    data = dde.data.TimePDE(geomtime, pde, [], num_domain=8000, num_boundary=400, num_initial=800)
    
The number 8000 is the number of training residual points sampled inside the domain, and the number 400 is the number of training points sampled on the boundary. We also include 800 initial residual points for the initial conditions.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:
    
.. code-block:: python

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

We add the ouput transformation layer:

.. code-block:: python

    net.apply_output_transform(output_transform)
    
Now that we have defined the neural network, we build a ``Model``, choose the optimizer and learning rate (``lr``), and train it for 40000 iterations:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(epochs=40000)
    
After we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    model.compile("L-BFGS")
    losshistory, train_state = model.train()

Complete Code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Allen_Cahn.py
  :language: python
