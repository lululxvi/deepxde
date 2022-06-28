Laplace equation on a disk
================================

Problem setup
--------------

We will solve a Laplace equation in a polar coordinate system:

.. math:: r\frac{dy}{dr} + r^2\frac{dy}{dr^2} + \frac{dy}{d\theta^2} = 0,  \qquad r \in [0, 1], \quad \theta \in [0, 2\pi]

with the Dirichlet boundary condition

.. math:: y(1,\theta) = \cos(\theta)

and the periodic boundary condition

.. math:: y(r, \theta +2\pi) = y(r, \theta).

The reference solution is :math:`y=r\cos(\theta)`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Laplace equation step-by-step.

First, the DeepXDE, NumPy (``np``), and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf

We begin by defining a computational geometry. We can use a built-in class ``Rectangle`` as follows

.. code-block:: python

    geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])

Next, we express the PDE residual of the Laplace equation:

.. code-block:: python

    def pde(x, y):
        dy_r = dde.grad.jacobian(y, x, i=0, j=0)
        dy_rr = dde.grad.hessian(y, x, i=0, j=0)
        dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
        return x[:, 0:1] * dy_r + x[:, 0:1] ** 2 * dy_rr + dy_thetatheta

The first argument to ``pde`` is 2-dimensional vector where the first component(``x[:,0:1]``) is :math:`r`-coordinate and the second componenet (``x[:,1:]``) is the :math:`\theta`-coordinate. The second argument is the network output, i.e., the solution :math:`y(r,\theta)`.

Next, we consider the Dirichlet boundary condition. We need to implement a function, which should return ``True`` for points inside the subdomain and ``False`` for the points outside. In our case, if the points satisfy :math:`r=1` and are on the whole boundary of the rectangle domain, then function ``boundary`` returns ``True``. Otherwise, it returns ``False``. (Note that because of rounding-off errors, it is often wise to use ``np.isclose`` to test whether two floating point values are equivalent.)

.. code-block:: python

    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

The argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=2` in this case. To facilitate the implementation of ``boundary``, a boolean ``on_boundary`` is used as the second argument. If the point :math:`(r,\theta)` (the first argument) is on the entire boundary of the rectangle geometry that created above, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``. 

Using a lambda funtion, the ``boundary`` we defined above can be passed to ``DirichletBC`` as the third argument. Thus, the Dirichlet boundary condition is

.. code-block:: python

    bc_rad = dde.icbc.DirichletBC(
        geom,
        lambda x: np.cos(x[:, 1:2]),
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
    )
   
Now, we have specified the geometry, PDE residual, and boundary condition. We then define the ``PDE`` problem as

.. code-block:: python

    data = dde.data.PDE(
        geom, pde, bc_rad, num_domain=2540, num_boundary=80, solution=solution
    )

The number 2540 is the number of training residual points sampled inside the domain, and the number 80 is the number of training points sampled on the boundary. The argument  ``solution`` is the reference solution to compute the error of our solution, and we define it as follows:

.. code-block:: python

    def solution(x):
        r, theta = x[:, 0:1], x[:, 1:]
        return r * np.cos(theta)

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:

.. code-block:: python
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

If we rewrite this problem in cartesian coordinates, the variables are in the form of :math:`[r\sin(\theta), r\cos(\theta)]`. We use them as features to satisfy the certain underlying physical constraints, so that the network is automatically periodic along the :math:`\theta` coordinate and the period is :math:`2\pi`.

.. code-block:: python

    def feature_transform(x):
        return tf.concat(
            [x[:, 0:1] * tf.sin(x[:, 1:2]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
        )

Then we apply ``feature_transform`` to the network inputs:

.. code-block:: python

    net.apply_feature_transform(feature_transform)

Now, we have the PDE problem and the network. We bulid a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    
   
We then train the model for 15000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=15000)
    
We also save and plot the best trained result and loss history.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Laplace_disk.py
  :language: python
