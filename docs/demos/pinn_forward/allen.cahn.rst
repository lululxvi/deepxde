Allen-Cahn equation
================

Problem setup
--------------

We will solve an Allen-Cahn equation:

.. math:: \frac{\partial u}{\partial t} = d\frac{\partial^2u}{\partial x^2} + 5(u - u^3), \quad x \in [-1, 1], \quad t \in [0, 1]

The initial condition is defined as the following:

.. math:: u(x, 0) = x^2\cos(\pi x)

And the boundary condition is defined:

.. math::  u(-1, t) = u(1, t) = -1

The reference solution is `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Allen_Cahn.mat>`_.

Implementation
--------------

This description goes through the implementation of a solver for the above described Allen-Cahn equation step-by-step.

First, the DeepXDE, NumPy (``np``), Scipy, and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from scipy.io import loadmat
    from deepxde.backend import tf
    
We then begin by defining a computational geometry and a time domain. We can use a built-in class ``Interval`` and ``TimeDomain``, and we can combine both of the domains using ``GeometryXTime``.
    
.. code-block:: python
    
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
       
Now, we express the PDE residual of the Allen-Cahn equation:

.. code-block:: python

    d = 0.001
    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - d * dy_xx - 5 * (y - y**3)
        
The first argument to ``pde`` is a 2-dimensional vector where the first component(``x[:, 0]``) is :math:`x`-coordinate and the second component (``x[:, 1]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x, t)`, but here we use ``y`` as the name of the variable.

Now that we have specified the geometry and PDE residual, we can define the ``TimePDE`` problem as the following:

.. code-block:: python

    data = dde.data.TimePDE(geomtime, pde, [], num_domain=8000, num_boundary=400, num_initial=800)
    
The parameter ``num_domain=8000`` is the number of training residual points sampled inside the domain, and the parameter ``num_boundary=400`` is the number of training points sampled on the boundary. We also include the parameter ``num_initial=800``, which represents the number of initial residual points for the initial conditions.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:
    
.. code-block:: python

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    
Next, we consider the initial conditions and boundary constraints, defining the transformation of the output and applying it to the network. In this case, :math:`x^2\cos(\pi x) + t(1 - x^2)y` is used. When :math:`t=0`, the initial condition :math:`x^2\cos(\pi x)` is satisfied. When :math:`x=1`` or :math:`x=-1`, the boundary condition :math:`y(-1, t) = y(1, t) = -1` is satisfied. This demonstrates that the initial condition and the boundary constraint are both hard conditions.

.. code-block:: python

    def output_transform(x, y):
        return x[:, 0:1]**2 * tf.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y

    net.apply_output_transform(output_transform)
    
Now that we have defined the neural network, we build a ``Model``, choose the optimizer and learning rate (``lr``), and train it for 40000 iterations:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=40000)
    
After we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    
We then save and plot the best trained result and the loss history of the model.

.. code-block:: python
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Next, we load and prepare the dataset with ``gen_testdata()``. Finally, we test the model and display a graph containing both training loss and testing loss over time. We also display a graph containing the predicted solution to the PDE.

.. code-block:: python

    def gen_testdata():
        data = loadmat("../dataset/Allen_Cahn.mat")

        t = data["t"]
        x = data["x"]
        u = data["u"]

        dt = dx = 0.01
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.flatten()[:, None]
        return X, y
    
    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    f = model.predict(X, operator=pde)
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

Complete Code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Allen_Cahn.py
  :language: python
