Inverse problem for the diffusion-reaction system
================

Problem setup
--------------

We will solve an inverse problem for diffusion-reaction systems for unknowns :math:`D` and :math:`k_f`:

.. math:: \frac{\partial C_A}{\partial t} = D\frac{\partial^2 C_A}{\partial x^2} - k_f C_A C_B^2,

.. math:: \frac{\partial C_B}{\partial t} = D\frac{\partial^2 C_B}{\partial x^2} - 2k_f C_A C_B^2

for :math:`x \in [0, 1]` and :math:`t \in [0, 10]` initial conditions 

.. math:: C_A(x, 0) = C_B(x, 0) = e^{-20x}

and boundary conditions 

.. math:: C_A(0, t) = C_B(0, t) = 1

.. math:: C_A(1, t) = C_B(1, t) = 0.

The training dataset is `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/reaction.npz>`_, and the expected values of :math:`D` and :math:`k_f` are :math:`2 \cdot 10^{-3}` and :math:`0.1`, respectively.

Implementation
--------------

We first import DeepXDE and numpy (``np``):

.. code-block:: python

    import deepxde as dde
    import numpy as np

Now, we define the unknown variables :math:`D` and :math:`k_f` with initial guesses of 1 and 0.05, respectively:

.. code-block:: python

    kf = dde.Variable(0.05)
    D = dde.Variable(1.0)
    
We define the computational geometries by using the built-in ``Interval`` and ``TimeDomain`` classes and combining them with ``GeometryXTime``:

.. code-block:: python

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 10)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

Now, we create the reaction-inverse PDE:

.. code-block:: python

    def pde(x, y):
        ca, cb = y[:, 0:1], y[:, 1:2]
        dca_t = dde.grad.jacobian(y, x, i=0, j=1)
        dca_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dcb_t = dde.grad.jacobian(y, x, i=1, j=1)
        dcb_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        eq_a = dca_t - 1e-3 * D * dca_xx + kf * ca * cb ** 2
        eq_b = dcb_t - 1e-3 * D * dcb_xx + 2 * kf * ca * cb ** 2
        return [eq_a, eq_b]

Here, the first parameter is the :math:`t`-coordinate, and it is represented by ``x``. The second parameter has :math:`C_A` and :math:`C_B`, which are represented by ``y``. Then, we use ``dde.grad.jacobian`` and ``dde.grad.hessian`` to represent the desired first and second order partial derivatives. 

Next, we consider the Dirichlet boundary conditions:

.. code-block:: python 

    def fun_bc(x):
        return 1 - x[:, 0:1]

Now, we define the initial conditions:

.. code-block:: python 

    def fun_init(x):
        return np.exp(-20 * x[:, 0:1]) 

Now that these are defined, we apply them:

.. code-block:: python

    bc_a = dde.icbc.DirichletBC(
        geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0
    )
    bc_b = dde.icbc.DirichletBC(
        geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
    )
    ic1 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
    ic2 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

Now, we generate the training data by getting it from `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/reaction.npz>`_:

.. code-block:: python

    def gen_traindata():
        data = np.load("dataset/reaction.npz")
        t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
        X, T = np.meshgrid(x, t)
        X = np.reshape(X, (-1, 1))
        T = np.reshape(T, (-1, 1))
        Ca = np.reshape(ca, (-1, 1))
        Cb = np.reshape(cb, (-1, 1))
        return np.hstack((X, T)), Ca, Cb

After generating the data, we organize it:

.. code-block:: python

    observe_x, Ca, Cb = gen_traindata()
    observe_y1 = dde.icbc.PointSetBC(observe_x, Ca, component=0)
    observe_y2 = dde.icbc.PointSetBC(observe_x, Cb, component=1)

Now, we can define the ``TimePDE`` problem as follows:

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_a, bc_b, ic1, ic2, observe_y1, observe_y2],
        num_domain=2000,
        num_boundary=100,
        num_initial=100,
        anchors=observe_x,
        num_test=50000,
    )

We have 2000 training residual points in the domain, 100 points on the boundary, 100 points for the initial conditions, and 50000 to test the PDE residual. ``anchors`` specifies the training points as well.

Now, we create the network:

.. code-block:: python

    net = dde.nn.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")

This network has two inputs, one for the :math:`t`-coordinate and one for the :math:`x`-coordinate, and three hidden layers with 20 neurons each. The output layer has two outputs, one for :math:`C_A` and one for :math:`C_B`. We also choose ``tanh`` to be the activation function, and the initializer is ``Glorot uniform``.

Now, we create the ``Model`` and specify the optimizer, learning rate, and ``external_trainable_variables``. We also output the values of :math:`D` and :math:`k_f` every 1000 iterations.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, external_trainable_variables=[kf, D])
    variable = dde.callbacks.VariableValue([kf, D], period=1000, filename="variables.dat")

Lastly, we train this network for 80000 iterations:

.. code-block:: python 
    
    losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Complete Code
--------------

.. literalinclude:: ../../../examples/pinn_inverse/reaction_inverse.py
  :language: python
