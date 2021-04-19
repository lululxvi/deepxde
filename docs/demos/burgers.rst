Burgers equation in 1D with Dirichlet boundary conditions
=========================================================

Problem setup
--------------

We will solve a Burgers equation:

.. math:: \frac{du}{dt} + u\frac{du}{dx} = \nu\frac{du}{dx^2}, \qquad x \in [-1, 1], \qquad t \in [0, 1]

with the Dirichlet boundary conditions

.. math:: u(-1,t)=u(1,t)=0, \quad u(x) = - \sin(\pi x).

The reference solution is <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz>

Implementation
--------------
1)We first define the ``Geometry`` , ``Time``, and combine both of them using ``GeometryXTime``

.. code-block:: python

   geom = dde.geometry.Interval(-1, 1)
   timedomain = dde.geometry.TimeDomain(0, 0.99)
   geomtime = dde.geometry.GeometryXTime(geom, timedomain)
      
2)Define the ``pde residual`` function of Poisson equation

.. code-block:: python

  def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
        
3)Define ``boundary conditions`` , ``Initail conditions`` and ``Data`` which is geometry + pde + BC + IC + training points using DeepXDE inbuilt functions as shown

.. code-block:: python
  
  bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
  ic = dde.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
  data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160)
    
4)We use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20

.. code-block:: python

  net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
     
5)Bulid the Model and Compile using the ``MODEL`` and ``Complile`` functions of DeepXDE as shown

.. code-block:: python

  model = dde.Model(data, net)
  model.compile("adam", lr=1e-3)
  

6)Train the model using ``Train`` function of deepXDE

.. code-block:: python

  model.train(epochs=15000)
  
7)Predict values and function "f" by using the ``Predict`` function of deepXDE 

.. code-block:: python 

   y_pred = model.predict(X)
   f = model.predict(X, operator=pde)

Complete code
--------------

.. literalinclude:: ../../examples/Burgers.py
  :language: python
